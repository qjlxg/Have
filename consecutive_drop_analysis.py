import os
import pandas as pd
import glob
import re
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# --- 配置 ---
DATA_DIR = 'fund_data'
ETF_LIST_FILE = 'ETF列表.txt'
PORTFOLIO_FILE = 'virtual_portfolio.csv'
MIN_TURNOVER = 5000000       
MIN_SCORE_SIGNAL = 60        

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
    delta = df['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    low_9 = df['最低'].rolling(9).min()
    high_9 = df['最高'].rolling(9).max()
    rsv = (df['收盘'] - low_9) / (high_9 - low_9) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    df['MA5'] = df['收盘'].rolling(5).mean()
    df['MA20'] = df['收盘'].rolling(20).mean()
    df['BIAS_20'] = (df['收盘'] - df['MA20']) / df['MA20'] * 100
    df['V_MA5'] = df['成交量'].shift(1).rolling(5).mean()
    df['VOL_RATIO'] = df['成交量'] / df['V_MA5']
    return df.sort_values('日期', ascending=False).reset_index(drop=True)

def update_portfolio(new_signals):
    """更新虚拟持仓账本，修复FutureWarning并确保文件写入"""
    cols = ['代码', '名称', '买入日期', '买入价', '当前价', '持有天数', '当前收益%', '信号类型']
    
    if os.path.exists(PORTFOLIO_FILE):
        try:
            df_p = pd.read_csv(PORTFOLIO_FILE)
        except:
            df_p = pd.DataFrame(columns=cols)
    else:
        df_p = pd.DataFrame(columns=cols)

    # 1. 存入新信号
    new_entries = []
    for s in new_signals:
        if s['综合评分'] >= MIN_SCORE_SIGNAL:
            code_str = str(s['代码']).zfill(6)
            exists = False
            if not df_p.empty:
                exists = ((df_p['代码'].astype(str).str.zfill(6) == code_str) & 
                          (df_p['买入日期'] == s['日期'])).any()
            
            if not exists:
                new_entries.append({
                    '代码': code_str, '名称': s['名称'], '买入日期': s['日期'],
                    '买入价': s['现价'], '当前价': s['现价'], '持有天数': 0,
                    '当前收益%': 0.0, '信号类型': s['信号强度']
                })
    
    if new_entries:
        df_p = pd.concat([df_p, pd.DataFrame(new_entries)], ignore_index=True)

    # 2. 刷新价格 (使用 fund_data 里的最新价)
    if not df_p.empty:
        files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        for idx, row in df_p.iterrows():
            code_str = str(row['代码']).zfill(6)
            target_file = [f for f in files if code_str in f]
            if target_file:
                df_temp = pd.read_csv(target_file[0])
                if not df_temp.empty:
                    last_row = df_temp.iloc[-1]
                    last_price = last_row['收盘']
                    last_date = last_row['日期']
                    
                    d1 = datetime.strptime(str(row['买入日期']), '%Y-%m-%d')
                    d2 = datetime.strptime(str(last_date), '%Y-%m-%d')
                    
                    df_p.at[idx, '当前价'] = last_price
                    df_p.at[idx, '持有天数'] = (d2 - d1).days
                    df_p.at[idx, '当前收益%'] = round((last_price - row['买入价']) / row['买入价'] * 100, 2)

    df_p.to_csv(PORTFOLIO_FILE, index=False, encoding='utf-8-sig')
    return df_p

# ... analyze_single_file 保持不变 ...

def analyze_single_file(file_info):
    file_path, name_mapping = file_info
    try:
        code = re.search(r'(\d{6})', os.path.basename(file_path)).group(1)
        df = pd.read_csv(file_path)
        if len(df) < 40: return None
        df = calculate_tech(df)
        last = df.iloc[0]; prev = df.iloc[1]
        if float(last['成交额']) < MIN_TURNOVER: return None

        score_oversold = 0
        if last['RSI'] < 38: score_oversold += 35
        if last['J'] < 10: score_oversold += 35
        if last['BIAS_20'] < -3: score_oversold += 30
        
        is_strong = last['RSI'] > 65 and last['收盘'] > last['MA5']
        
        if score_oversold >= 70 and last['J'] > prev['J']:
            sig, adv, score, sl = "★★★ 超跌反弹", "底部确认，波段持有。", score_oversold, last['最低']
        elif last['RSI'] > 80:
            sig, adv, score, sl = "☢ 极致超买", "博傻阶段，严守5日线。", -20, round(last['MA5'], 3)
        elif is_strong:
            sig, adv, score, sl = "🚀 趋势主升", "动能强。5日线为止损线。", 65, round(last['MA5'], 3)
        else:
            return None

        return {
            '代码': code, '名称': name_mapping.get(code, "未知"), '信号强度': sig,
            '操作建议': adv, '综合评分': score, '建议止损价': sl,
            '现价': last['收盘'], 'RSI': round(last['RSI'], 2), 'KDJ_J': round(last['J'], 2),
            'MA20偏离%': round(last['BIAS_20'], 2), '当前量比': round(last['VOL_RATIO'], 2),
            '日期': last['日期']
        }
    except: return None

def main():
    name_mapping = get_target_mapping()
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    tasks = [(f, name_mapping) for f in files]

    print(f"🚀 正在扫描全市场标的并更新账本...")
    results = []
    with ProcessPoolExecutor() as executor:
        for res in executor.map(analyze_single_file, tasks):
            if res: results.append(res)

    if not results:
        res_df = pd.DataFrame(columns=['代码', '名称', '信号强度', '操作建议', '综合评分', '建议止损价', '现价', '日期'])
    else:
        res_df = pd.DataFrame(results).sort_values(by='综合评分', ascending=False)
    
    res_df.to_csv('investment_decision.csv', index=False, encoding='utf-8-sig')
    portfolio_df = update_portfolio(results)
    
    now = datetime.now()
    h_dir = os.path.join('history', now.strftime('%Y'), now.strftime('%m'))
    os.makedirs(h_dir, exist_ok=True)
    res_df.to_csv(os.path.join(h_dir, f"report_{now.strftime('%Y%m%d')}.csv"), index=False, encoding='utf-8-sig')

    print(f"✅ 复盘完成。")
    if not portfolio_df.empty:
        total = len(portfolio_df)
        wins = len(portfolio_df[portfolio_df['当前收益%'] > 0])
        print(f"\n📊 账本统计 | 信号总数: {total} | 胜率: {wins/total*100:.1f}% | 平均浮盈: {portfolio_df['当前收益%'].mean():.2f}%")
        print("\n📈 表现最好的历史信号 (Top 5):")
        print(portfolio_df.sort_values(by='当前收益%', ascending=False).head(5)[['代码','名称','买入日期','持有天数','当前收益%']].to_string(index=False))

    if not res_df.empty:
        print("\n🔥 今日信号摘要:")
        print(res_df[['代码', '名称', '信号强度', '现价', '建议止损价']].head(5).to_string(index=False))

if __name__ == "__main__":
    main()
