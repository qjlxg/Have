import os
import pandas as pd
import glob
import re
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# --- 交易逻辑配置 ---
DATA_DIR = 'fund_data'
ETF_LIST_FILE = 'ETF列表.txt'
MIN_TURNOVER = 5000000       # 500万成交额门槛
AVG_DAYS = 5                 
# ------------------

def get_target_mapping():
    if not os.path.exists(ETF_LIST_FILE): return {}
    mapping = {}
    for enc in ['utf-8', 'gbk', 'utf-16']:
        try:
            with open(ETF_LIST_FILE, 'r', encoding=enc) as f:
                for line in f:
                    line = line.strip()
                    if not line or "证券代码" in line: continue
                    match = re.search(r'(\d{6})\s+(.+)', line)
                    if match:
                        code, name = match.groups()
                        mapping[code] = name.strip()
            if mapping: return mapping
        except: continue
    return {}

def calculate_technical_indicators(df):
    tdf = df.iloc[::-1].copy()
    delta = tdf['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    tdf['RSI'] = 100 - (100 / (1 + (gain / loss)))
    low_9 = tdf['收盘'].rolling(9).min()
    high_9 = tdf['收盘'].rolling(9).max()
    rsv = (tdf['收盘'] - low_9) / (high_9 - low_9) * 100
    tdf['K'] = rsv.ewm(com=2, adjust=False).mean()
    tdf['D'] = tdf['K'].ewm(com=2, adjust=False).mean()
    tdf['J'] = 3 * tdf['K'] - 2 * tdf['D']
    res_tdf = tdf.iloc[::-1]
    df['RSI'], df['KDJ_J'] = res_tdf['RSI'], res_tdf['J']
    return df

def analyze_file(file_info):
    file_path, name_mapping = file_info
    try:
        code = re.search(r'(\d{6})', os.path.basename(file_path)).group(1)
        df = pd.read_csv(file_path)
        if len(df) < 30: return None
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期', ascending=False).reset_index(drop=True)
        last_price, last_turnover = df.loc[0, '收盘'], float(df.loc[0, '成交额'])
        if last_turnover < MIN_TURNOVER: return None
        df = calculate_technical_indicators(df)
        count = 0
        for i in range(len(df)):
            if float(df.loc[i, '涨跌幅']) < 0: count += 1
            else: break
        def get_chg(d): return round(((last_price - df.loc[d, '收盘']) / df.loc[d, '收盘']) * 100, 2) if len(df) > d else 0
        w_chg, m_chg, y_chg = get_chg(5), get_chg(20), get_chg(250)
        ma5 = df.loc[0:4, '收盘'].mean()
        bias = round(((last_price - ma5) / ma5) * 100, 2)
        rsi, j_val = df.loc[0, 'RSI'], df.loc[0, 'KDJ_J']
        vol_ratio = round(df.loc[0, '成交量'] / df.loc[1:5, '成交量'].mean(), 2)
        turnover = df.loc[0, '换手率'] if '换手率' in df.columns else 0
        score = 0
        if 3 <= count <= 5: score += 20
        if rsi < 35: score += 20
        if j_val < 0: score += 20
        if y_chg < -10: score += 15
        if bias < -2.5: score += 15
        if 0.4 < vol_ratio < 0.9: score += 10
        if score >= 85: signal, advice = "★★★ 一击必中", "多维共振极度超卖。建议重仓狙击/加仓。"
        elif 65 <= score < 85: signal, advice = "★★ 分批试错", "情绪接近底部。建议头仓入场。"
        elif 40 <= score < 65: signal, advice = "★ 试探观察", "跌势放缓。等待J线拐头。"
        else: signal, advice = "○ 择机等待", "指标平庸，观望为主。"
        return {
            '代码': code, '名称': name_mapping.get(code, "未知"), '信号强度': signal, '操作建议': advice, 
            '综合评分': score, '现价': last_price, '连跌天数': count, 'RSI': round(rsi, 2), 'KDJ_J': round(j_val, 2),
            '周幅%': w_chg, '月幅%': m_chg, '年幅%': y_chg, 'MA5偏离%': bias, '量比': vol_ratio, 
            '换手率%': round(turnover, 2), '成交额(万)': round(last_turnover/10000, 2), '日期': df.loc[0, '日期'].strftime('%Y-%m-%d')
        }
    except: return None

def summarize_performance(current_results):
    """新增：全自动胜率汇总功能"""
    current_prices = {item['代码']: item['现价'] for item in current_results}
    history_files = sorted(glob.glob(os.path.join('history', '**', 'report_*.csv'), recursive=True), reverse=True)
    
    perf_records = []
    if not history_files: return
    
    for h_file in history_files[:20]: # 追溯最近20个交易日
        h_df = pd.read_csv(h_file)
        # 统计当时评分 >= 65 的信号
        h_signals = h_df[h_df['综合评分'] >= 65]
        for _, row in h_signals.iterrows():
            code = str(row['代码']).zfill(6)
            if code in current_prices:
                gain = (current_prices[code] - row['现价']) / row['现价']
                perf_records.append({'信号': row['信号强度'], '收益': gain})
    
    if perf_records:
        pdf = pd.DataFrame(perf_records)
        summary_str = f"\n策略胜率自动汇总报告 ({datetime.now().strftime('%Y-%m-%d')})\n"
        summary_str += "="*50 + "\n"
        for sig in ["★★★ 一击必中", "★★ 分批试错"]:
            sub = pdf[pdf['信号'] == sig]
            if not sub.empty:
                win_rate = (sub['收益'] > 0).mean() * 100
                avg_return = sub['收益'].mean() * 100
                summary_str += f"{sig}: 样本数={len(sub)}, 胜率={win_rate:.2f}%, 平均涨幅={avg_return:.2f}%\n"
        summary_str += "="*50 + "\n"
        with open('Strategy_Backtest_Summary.txt', 'w', encoding='utf-8') as f:
            f.write(summary_str)
        print(summary_str)

def main():
    name_mapping = get_target_mapping()
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    tasks = [(f, name_mapping) for f in all_files]
    results = []
    with ProcessPoolExecutor() as executor:
        for res in executor.map(analyze_file, tasks):
            if res: results.append(res)
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values(by='综合评分', ascending=False)
    res_df.to_csv('investment_decision.csv', index=False, encoding='utf-8-sig')
    now = datetime.now()
    h_dir = os.path.join('history', now.strftime('%Y'), now.strftime('%m'))
    os.makedirs(h_dir, exist_ok=True)
    res_df.to_csv(os.path.join(h_dir, f"report_{now.strftime('%Y%m%d')}.csv"), index=False, encoding='utf-8-sig')
    summarize_performance(results) # 执行汇总

if __name__ == "__main__":
    main()
