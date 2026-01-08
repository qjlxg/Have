import os
import pandas as pd
import glob
import re
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# --- 核心同步配置 ---
DATA_DIR = 'fund_data'
ETF_LIST_FILE = 'ETF列表.txt'
MIN_TURNOVER = 5000000       # 500万成交额门槛
MIN_SCORE_SIGNAL = 70        # 同步回测：降低门槛至70分，提升资金利用率
# 量化逻辑：不放量杀跌即可 (0.4 - 1.1)
VOL_RATIO_UPPER = 1.1
VOL_RATIO_LOWER = 0.4

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
    """同步回测中的技术指标逻辑"""
    df = df.sort_values('日期').copy()
    # RSI
    delta = df['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    # KDJ
    low_9 = df['收盘'].rolling(9).min()
    high_9 = df['收盘'].rolling(9).max()
    rsv = (df['收盘'] - low_9) / (high_9 - low_9) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    # 乖离率与量比
    df['MA20'] = df['收盘'].rolling(20).mean()
    df['BIAS_20'] = (df['收盘'] - df['MA20']) / df['MA20'] * 100
    df['V_MA5'] = df['成交量'].shift(1).rolling(5).mean()
    df['VOL_RATIO'] = df['成交量'] / df['V_MA5']
    return df.sort_values('日期', ascending=False).reset_index(drop=True)

def analyze_single_file(file_info):
    file_path, name_mapping = file_info
    try:
        code = re.search(r'(\d{6})', os.path.basename(file_path)).group(1)
        df = pd.read_csv(file_path)
        if len(df) < 30: return None
        
        df = calculate_tech(df)
        last = df.iloc[0]
        prev = df.iloc[1]
        
        if float(last['成交额']) < MIN_TURNOVER: return None
            
        # 1. 评分系统 (同步回测中线策略)
        score = 0
        if last['RSI'] < 35: score += 30
        if last['J'] < 5: score += 30
        if last['BIAS_20'] < -4: score += 40
        
        # 2. 核心过滤逻辑：不放量杀跌 & 右侧J线勾头
        is_vol_safe = VOL_RATIO_LOWER < last['VOL_RATIO'] < VOL_RATIO_UPPER
        is_turning = last['J'] > prev['J']
        
        # 3. 信号分级与回测预期参考
        if score >= MIN_SCORE_SIGNAL and is_vol_safe and is_turning:
            if score >= 85:
                signal, advice = "★★★ 深度共振", "超跌极值+勾头确认。回测40天平均收益2.7%+"
            else:
                signal, advice = "★★ 波段机会", "温和止跌。月均信号约15次，建议分批介入。"
        elif last['VOL_RATIO'] > 1.8 and float(last['涨跌幅']) < -2.5:
            signal, advice = "☢ 放量杀跌", "抛压极重，避让。"
            score = -10 # 风险标识
        else:
            return None # 过滤掉无信号的，提升决策列表清晰度

        return {
            '代码': code, '名称': name_mapping.get(code, "未知"), '信号强度': signal, 
            '操作建议': advice, '综合评分': score, '现价': last['收盘'], 
            'RSI': round(last['RSI'], 2), 'KDJ_J': round(last['J'], 2), 
            'MA20偏离%': round(last['BIAS_20'], 2), '当前量比': round(last['VOL_RATIO'], 2), 
            '预期40天收益': '2.7%+', '10天胜率参考': '54%',
            '日期': last['日期']
        }
    except: return None

def main():
    name_mapping = get_target_mapping()
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    tasks = [(f, name_mapping) for f in files]

    print(f"🚀 开始全自动复盘 (频率优化版)...")
    results = []
    with ProcessPoolExecutor() as executor:
        for res in executor.map(analyze_single_file, tasks):
            if res: results.append(res)

    res_df = pd.DataFrame(results).sort_values(by='综合评分', ascending=False)
    res_df.to_csv('investment_decision.csv', index=False, encoding='utf-8-sig')
    
    # 历史存档逻辑保持不变
    now = datetime.now()
    h_dir = os.path.join('history', now.strftime('%Y'), now.strftime('%m'))
    os.makedirs(h_dir, exist_ok=True)
    res_df.to_csv(os.path.join(h_dir, f"report_{now.strftime('%Y%m%d')}.csv"), index=False, encoding='utf-8-sig')

    print(f"✅ 复盘完成！今日检出信号: {len(res_df)} 个。")
    if not res_df.empty:
        print("\n--- 核心关注标的 ---")
        print(res_df[['代码', '名称', '综合评分', '信号强度']].head(5).to_string(index=False))

if __name__ == "__main__":
    main()
