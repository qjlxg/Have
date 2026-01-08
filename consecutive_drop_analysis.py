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
MIN_TURNOVER = 1000000       # 100万成交额门槛
AVG_DAYS = 5                 # 均线周期
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
    """
    计算 RSI 和 KDJ。注意：df 传入时是时间倒序(0是今天)
    为了计算准确，内部需要先转为正序。
    """
    # 转为正序计算指标
    tdf = df.iloc[::-1].copy()
    
    # 1. RSI (14日)
    delta = tdf['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    tdf['RSI'] = 100 - (100 / (1 + rs))

    # 2. KDJ (9, 3, 3)
    low_list = tdf['收盘'].rolling(9).min()
    high_list = tdf['收盘'].rolling(9).max()
    rsv = (tdf['收盘'] - low_list) / (high_list - low_list) * 100
    tdf['K'] = rsv.ewm(com=2, adjust=False).mean()
    tdf['D'] = tdf['K'].ewm(com=2, adjust=False).mean()
    tdf['J'] = 3 * tdf['K'] - 2 * tdf['D']
    
    # 转回倒序并同步到原df
    res_tdf = tdf.iloc[::-1]
    df['RSI'] = res_tdf['RSI']
    df['K'] = res_tdf['K']
    df['D'] = res_tdf['D']
    df['J'] = res_tdf['J']
    return df

def analyze_file(file_info):
    file_path, name_mapping = file_info
    try:
        code_match = re.search(r'(\d{6})', os.path.basename(file_path))
        if not code_match: return None
        code = code_match.group(1)
        
        df = pd.read_csv(file_path)
        if len(df) < 30: return None # 数据太少无法计算RSI/KDJ
        
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期', ascending=False).reset_index(drop=True)
        
        last_price = df.loc[0, '收盘']
        last_turnover = float(df.loc[0, '成交额'])
        if last_turnover < MIN_TURNOVER: return None
            
        # 计算技术指标
        df = calculate_technical_indicators(df)
        
        # 1. 连跌天数统计
        count, total_drop_pct = 0, 0.0
        for i in range(len(df)):
            change = float(df.loc[i, '涨跌幅'])
            if change < 0:
                count += 1
                total_drop_pct += change
            else: break
        
        # 2. 均线与量比
        ma5 = df.loc[0:AVG_DAYS-1, '收盘'].mean()
        bias = round(((last_price - ma5) / ma5) * 100, 2)
        avg_vol = df.loc[1:AVG_DAYS, '成交量'].mean()
        vol_ratio = round(df.loc[0, '成交量'] / avg_vol, 2) if avg_vol > 0 else 0
        
        # 3. 周、月、年幅 (原有功能还原)
        def get_period_change(days):
            if len(df) > days:
                prev = df.loc[days, '收盘']
                return round(((last_price - prev) / prev) * 100, 2)
            return None

        # 4. 获取最新指标值
        rsi_val = round(df.loc[0, 'RSI'], 2) if not pd.isna(df.loc[0, 'RSI']) else 0.0
        j_val = round(df.loc[0, 'J'], 2) if not pd.isna(df.loc[0, 'J']) else 0.0
        turnover = round(df.loc[0, '换手率'], 2) if '换手率' in df.columns else 0.0

        # 5. 决策逻辑 (集成全部指标)
        decision = "观察"
        # 黄金坑逻辑：连跌3-5天 + 低RSI(<35) + 低J值(<0) + 负偏离
        if 3 <= count <= 5 and bias < -2.0 and (rsi_val < 35 or j_val < 0):
            decision = "★★★ 极度超跌(黄金坑)"
        elif 3 <= count <= 5:
            decision = f"连跌{count}天(监控中)"
        elif rsi_val < 30:
            decision = "超卖预警"

        return {
            '代码': code, '名称': name_mapping.get(code, "未知"),
            '决策建议': decision, '现价': last_price,
            '连跌天数': count, 'RSI': rsi_val, 'KDJ_J': j_val, '换手率%': turnover,
            '连跌%': round(total_drop_pct, 2), 
            '周幅%': get_period_change(5), '月幅%': get_period_change(20), '年幅%': get_period_change(250),
            'MA5偏离%': bias, '量比': vol_ratio,
            '成交额(万)': round(last_turnover/10000, 2), '日期': df.loc[0, '日期'].strftime('%Y-%m-%d')
        }
    except Exception as e:
        return None

def track_performance(current_results):
    """回测模块：原有功能保持不变"""
    current_prices = {item['代码']: item['现价'] for item in current_results}
    history_files = sorted(glob.glob(os.path.join('history', '**', 'decision_*.csv'), recursive=True), reverse=True)
    perf_list = []
    if history_files:
        for h_file in history_files[:10]:
            try:
                h_df = pd.read_csv(h_file)
                h_date = h_df['日期'].iloc[0]
                h_targets = h_df[h_df['决策建议'].str.contains('★|连跌')]
                for _, row in h_targets.iterrows():
                    code = str(row['代码']).zfill(6)
                    if code in current_prices:
                        old_p, now_p = row['现价'], current_prices[code]
                        perf_list.append({
                            '决策日期': h_date, '代码': code, '名称': row['名称'],
                            '当时建议': row['决策建议'], '当时价': old_p,
                            '今日价': now_p, '盈亏%': round(((now_p - old_p) / old_p) * 100, 2)
                        })
            except: continue
    pd.DataFrame(perf_list).to_csv('performance_tracking.csv', index=False, encoding='utf-8-sig')

def main():
    name_mapping = get_target_mapping()
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    tasks = [(f, name_mapping) for f in all_files]

    results = []
    with ProcessPoolExecutor() as executor:
        for res in executor.map(analyze_file, tasks):
            if res: results.append(res)

    # 包含所有指标的完整输出
    cols = ['代码','名称','决策建议','现价','连跌天数','RSI','KDJ_J','换手率%','连跌%','周幅%','月幅%','年幅%','MA5偏离%','量比','成交额(万)','日期']
    res_df = pd.DataFrame(results, columns=cols)
    
    if not res_df.empty:
        res_df = res_df.sort_values(by=['决策建议', '连跌天数', 'RSI'], ascending=[False, False, True])
    
    res_df.to_csv('investment_decision.csv', index=False, encoding='utf-8-sig')
    
    # 历史归档
    now = datetime.now()
    h_dir = os.path.join('history', now.strftime('%Y'), now.strftime('%m'))
    os.makedirs(h_dir, exist_ok=True)
    h_file = os.path.join(h_dir, f"decision_{now.strftime('%Y%m%d_%H%M%S')}.csv")
    res_df.to_csv(h_file, index=False, encoding='utf-8-sig')
    
    track_performance(results)
    print(f"分析完成。已合并原有周/月/年幅与新指标。")

if __name__ == "__main__":
    main()
