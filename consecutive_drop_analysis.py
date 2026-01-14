import os
import pandas as pd
import glob
import re
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# --- 核心科学配置 ---
DATA_DIR = 'fund_data'
ETF_LIST_FILE = 'ETF列表.txt'
PORTFOLIO_FILE = 'virtual_portfolio.csv'
MIN_TURNOVER = 5000000       
MIN_SCORE_SIGNAL = 65        # 科学阈值：过滤掉无意义的杂波
TARGET_PROFIT = 5.0          # 自动止盈线 %

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
    df = df.sort_values('日期').copy()
    # RSI (14)
    delta = df['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    # KDJ (9,3,3)
    low_9 = df['最低'].rolling(9).min()
    high_9 = df['最高'].rolling(9).max()
    rsv = (df['收盘'] - low_9) / (high_9 - low_9) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    # 均线与乖离
    df['MA5'] = df['收盘'].rolling(5).mean()
    df['MA20'] = df['收盘'].rolling(20).mean()
    df['BIAS_20'] = (df['收盘'] - df['MA20']) / df['MA20'] * 100
    # 量能
    df['V_MA5'] = df['成交量'].shift(1).rolling(5).mean()
    df['VOL_RATIO'] = df['成交量'] / df['V_MA5']
    return df.sort_values('日期', ascending=False).reset_index(drop=True)

def update_portfolio(new_signals):
    """更新账本：自动识别平仓、新开仓、更新收益"""
    cols = ['代码', '名称', '买入日期', '买入价', '当前价', '止损价', '持有天数', '当前收益%', '信号类型', '状态']
    if os.path.exists(PORTFOLIO_FILE):
        try:
            df_p = pd.read_csv(PORTFOLIO_FILE)
            if '状态' not in df_p.columns: df_p['状态'] = '持仓中'
        except: df_p = pd.DataFrame(columns=cols)
    else:
        df_p = pd.DataFrame(columns=cols)

    # 1. 处理新信号入场
    new_rows = []
    for s in new_signals:
        if s['综合评分'] >= MIN_SCORE_SIGNAL:
            code_str = str(s['代码']).zfill(6)
            is_holding = False
            if not df_p.empty:
                is_holding = ((df_p['代码'].astype(str).str.zfill(6) == code_str) & (df_p['状态'] == '持仓中')).any()
            if not is_holding:
                new_rows.append({
                    '代码': code_str, '名称': s['名称'], '买入日期': s['日期'],
                    '买入价': s['现价'], '当前价': s['现价'], '止损价': s['建议止损价'],
                    '持有天数': 0, '当前收益%': 0.0, '信号类型': s['信号强度'], '状态': '持仓中'
                })
    if new_rows:
        df_p = pd.concat([df_p, pd.DataFrame(new_rows)], ignore_index=True)

    # 2. 刷新持仓状态 (平仓判定)
    if not df_p.empty:
        files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        for idx, row in df_p.iterrows():
            if row['状态'] != '持仓中': continue
            code_str = str(row['代码']).zfill(6)
            target_file = [f for f in files if code_str in f]
            if target_file:
                df_t = pd.read_csv(target_file[0])
                last_p = df_t.iloc[-1]['收盘']
                last_d = df_t.iloc[-1]['日期']
                
                profit = round((last_p - row['买入价']) / row['买入价'] * 100, 2)
                d_days = (datetime.strptime(str(last_d), '%Y-%m-%d') - datetime.strptime(str(row['买入日期']), '%Y-%m-%d')).days
                
                df_p.at[idx, '当前价'] = last_p
                df_p.at[idx, '持有天数'] = d_days
                df_p.at[idx, '当前收益%'] = profit

                if last_p < row['止损价']: df_p.at[idx, '状态'] = '止损退出'
                elif profit >= TARGET_PROFIT: df_p.at[idx, '状态'] = '止盈退出'

    df_p.to_csv(PORTFOLIO_FILE, index=False, encoding='utf-8-sig')
    return df_p

def analyze_single_file(file_info):
    file_path, name_mapping = file_info
    try:
        code = re.search(r'(\d{6})', os.path.basename(file_path)).group(1)
        df = pd.read_csv(file_path)
        if len(df) < 40: return None
        df = calculate_tech(df)
        last, prev = df.iloc[0], df.iloc[1]
        if float(last['成交额']) < MIN_TURNOVER: return None

        # --- 核心分流评分逻辑 ---
        ls_score = 0  # 左侧分
        rs_score = 0  # 右侧分
        
        # A. 左侧超跌维度
        if last['RSI'] < 30: ls_score += 40
        elif last['RSI'] < 40: ls_score += 25
        if last['J'] < 5: ls_score += 40
        elif last['J'] < 15: ls_score += 20
        if last['BIAS_20'] < -5: ls_score += 20

        # B. 右侧趋势维度
        is_uptrend = last['收盘'] > last['MA20'] and last['收盘'] > last['MA5']
        if is_uptrend and last['VOL_RATIO'] > 1.2: rs_score = 70
        
        # --- 最终信号判定 ---
        status, advice, final_score, sl = "无", "观望", 0, 0
        
        # 优先触发高胜率超跌信号
        if ls_score >= 80 and last['J'] > prev['J']:
            status, advice, final_score = "★★★ 五星金底", "超跌反弹高胜率区，建议分批介入。", ls_score
            sl = last['最低']
        # 其次触发动能趋势信号
        elif rs_score >= 70:
            status, advice, final_score = "🚀 趋势主升", "右侧动能强劲，跌破5日线止损。", rs_score
            sl = round(last['MA5'], 3)
        # 风险提示
        elif last['RSI'] > 82:
            status, advice, final_score = "☢ 极致超买", "情绪过热，随时回调，禁止开仓。", -20
            sl = round(last['MA5'], 3)
        else: return None

        return {
            '代码': code, '名称': name_mapping.get(code, "未知"), '信号强度': status,
            '操作建议': advice, '综合评分': final_score, '建议止损价': sl,
            '现价': last['收盘'], 'RSI': round(last['RSI'], 2), '日期': last['日期']
        }
    except: return None

def main():
    name_mapping = get_target_mapping()
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    tasks = [(f, name_mapping) for f in files]

    print(f"🚀 启动科学分流复盘系统...")
    results = []
    with ProcessPoolExecutor() as executor:
        for res in executor.map(analyze_single_file, tasks):
            if res: results.append(res)

    res_df = pd.DataFrame(results).sort_values(by='综合评分', ascending=False) if results else pd.DataFrame()
    res_df.to_csv('investment_decision.csv', index=False, encoding='utf-8-sig')
    portfolio_df = update_portfolio(results)
    
    # 历史归档
    now = datetime.now()
    h_dir = os.path.join('history', now.strftime('%Y'), now.strftime('%m'))
    os.makedirs(h_dir, exist_ok=True)
    res_df.to_csv(os.path.join(h_dir, f"report_{now.strftime('%Y%m%d')}.csv"), index=False, encoding='utf-8-sig')

    print(f"✅ 复盘完成！")
    if not portfolio_df.empty:
        active = portfolio_df[portfolio_df['状态'] == '持仓中']
        closed = portfolio_df[portfolio_df['状态'] != '持仓中']
        print
