import os
import pandas as pd
import glob
import re
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# --- 核心配置 ---
DATA_DIR = 'fund_data'
ETF_LIST_FILE = 'ETF列表.txt'
MIN_TURNOVER = 5000000       # 500万成交额门槛，过滤掉流动性差的僵尸品种
MIN_SCORE_SIGNAL = 85        # 一击必中触发阈值 (100分制)
HOLD_DAYS_BACKTEST = [3, 5, 10] # 回测统计周期

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

def calculate_tech(df):
    """计算核心技术指标，输入df需包含收盘、成交量、成交额"""
    # 转正序计算
    df = df.sort_values('日期').copy()
    
    # RSI (14日)
    delta = df['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))

    # KDJ (9, 3, 3)
    low_9 = df['收盘'].rolling(9).min()
    high_9 = df['收盘'].rolling(9).max()
    rsv = (df['收盘'] - low_9) / (high_9 - low_9) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    # 均线与量比
    df['MA5'] = df['收盘'].rolling(5).mean()
    df['V_MA5'] = df['成交量'].shift(1).rolling(5).mean()
    df['BIAS'] = (df['收盘'] - df['MA5']) / df['MA5'] * 100
    df['VOL_RATIO'] = df['成交量'] / df['V_MA5']
    
    return df.sort_values('日期', ascending=False).reset_index(drop=True)

def analyze_single_file(file_info):
    file_path, name_mapping = file_info
    try:
        code = re.search(r'(\d{6})', os.path.basename(file_path)).group(1)
        df = pd.read_csv(file_path)
        if len(df) < 30: return None
        
        df['日期'] = pd.to_datetime(df['日期'])
        df = calculate_tech(df)
        
        last = df.iloc[0]
        if float(last['成交额']) < MIN_TURNOVER: return None
            
        # 1. 连跌逻辑
        count = 0
        for i in range(len(df)):
            if float(df.loc[i, '涨跌幅']) < 0: count += 1
            else: break
        
        # 2. 空间涨跌
        def get_chg(d): return round(((last['收盘'] - df.loc[d, '收盘']) / df.loc[d, '收盘']) * 100, 2) if len(df) > d else 0
        w_chg, m_chg, y_chg = get_chg(5), get_chg(20), get_chg(250)
        
        # 3. 评分系统 (一击必中战法)
        score = 0
        if 3 <= count <= 5: score += 20      # 情绪冰点
        if last['RSI'] < 30: score += 20     # 指标超卖
        if last['J'] < 0: score += 20        # 拐点信号
        if y_chg < -15: score += 15          # 长期超跌
        if last['BIAS'] < -2.5: score += 15  # 短期乖离
        if 0.5 < last['VOL_RATIO'] < 0.9: score += 10 # 缩量企稳
        
        if score >= MIN_SCORE_SIGNAL:
            signal, advice = "★★★ 一击必中", "多维共振底部，建议果断分批介入。"
        elif score >= 65:
            signal, advice = "★★ 底部试错", "情绪接近极值，可轻仓建立头仓。"
        elif last['VOL_RATIO'] > 2.5 and last['涨跌幅'] < -2:
            signal, advice = "☢ 风险避让", "放量杀跌，恐有连环雷，管住手。"
        else:
            signal, advice = "○ 择机等待", "指标平庸，观望为宜。"

        return {
            '代码': code, '名称': name_mapping.get(code, "未知"), '信号强度': signal, '操作建议': advice,
            '综合评分': score, '现价': last['收盘'], '连跌天数': count, 'RSI': round(last['RSI'], 2), 
            'KDJ_J': round(last['J'], 2), '周幅%': w_chg, '月幅%': m_chg, '年幅%': y_chg, 
            'MA5偏离%': round(last['BIAS'], 2), '量比': round(last['VOL_RATIO'], 2), 
            '换手率%': round(last.get('换手率', 0), 2), '成交额(万)': round(last['成交额']/10000, 2), 
            '日期': last['日期'].strftime('%Y-%m-%d')
        }
    except: return None

def main():
    name_mapping = get_target_mapping()
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    tasks = [(f, name_mapping) for f in files]

    print(f"🚀 开始全自动复盘，扫描 {len(files)} 个标的...")
    results = []
    with ProcessPoolExecutor() as executor:
        for res in executor.map(analyze_single_file, tasks):
            if res: results.append(res)

    res_df = pd.DataFrame(results).sort_values(by='综合评分', ascending=False)
    res_df.to_csv('investment_decision.csv', index=False, encoding='utf-8-sig')
    
    # 历史归档
    now = datetime.now()
    h_dir = os.path.join('history', now.strftime('%Y'), now.strftime('%m'))
    os.makedirs(h_dir, exist_ok=True)
    res_df.to_csv(os.path.join(h_dir, f"report_{now.strftime('%Y%m%d')}.csv"), index=False, encoding='utf-8-sig')

    # 生成胜率汇总（基于历史report文件）
    history_files = sorted(glob.glob(os.path.join('history', '**', 'report_*.csv'), recursive=True), reverse=True)
    if len(history_files) > 1:
        perf = []
        cur_prices = {r['代码']: r['现价'] for r in results}
        for hf in history_files[1:11]: # 回溯最近10个历史档案
            h_df = pd.read_csv(hf)
            for _, row in h_df[h_df['综合评分'] >= 65].iterrows():
                c = str(row['代码']).zfill(6)
                if c in cur_prices:
                    perf.append({'信号': row['信号强度'], '收益': (cur_prices[c]-row['现价'])/row['现价']})
        if perf:
            pdf = pd.DataFrame(perf)
            with open('Strategy_Backtest_Summary.txt', 'w', encoding='utf-8') as f:
                f.write(f"策略实战胜率汇总 ({now.strftime('%Y-%m-%d')})\n")
                for s in ["★★★ 一击必中", "★★ 底部试错"]:
                    sub = pdf[pdf['信号'] == s]
                    if not sub.empty:
                        line = f"{s}: 样本={len(sub)}, 胜率={(sub['收益']>0).mean():.2%}, 平均收益={sub['收益'].mean():.2%}\n"
                        f.write(line); print(line)

    print(f"✅ 复盘完成！结果已存入 investment_decision.csv")

if __name__ == "__main__":
    main()
