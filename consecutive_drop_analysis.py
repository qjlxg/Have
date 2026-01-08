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
AVG_DAYS = 5                 # MA5和量比周期
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

def calculate_indicators(df):
    """计算 RSI, KDJ 和 换手率"""
    # 1. RSI (14日)
    delta = df['收盘'].diff(-1) # 这里的diff需要注意顺序，df是倒序
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 2. KDJ (9, 3, 3)
    # 因为df是倒序[0是今天]，计算KDJ需要临时正序
    temp_df = df.iloc[::-1].copy()
    low_list = temp_df['收盘'].rolling(9).min()
    high_list = temp_df['收盘'].rolling(9).max()
    rsv = (temp_df['收盘'] - low_list) / (high_list - low_list) * 100
    temp_df['K'] = rsv.ewm(com=2).mean()
    temp_df['D'] = temp_df['K'].ewm(com=2).mean()
    temp_df['J'] = 3 * temp_df['K'] - 2 * temp_df['D']
    
    # 将计算结果转回原df
    df['K'], df['D'], df['J'] = temp_df['K'][::-1], temp_df['D'][::-1], temp_df['J'][::-1]

    # 3. 换手率 (假设CSV中有'换手率'列，若无则设为0)
    if '换手率' not in df.columns:
        df['换手率'] = 0 
    
    return df

def analyze_file(file_info):
    file_path, name_mapping = file_info
    try:
        code_match = re.search(r'(\d{6})', os.path.basename(file_path))
        if not code_match: return None
        code = code_match.group(1)
        
        # 加载数据
        df = pd.read_csv(file_path)
        if len(df) < 30: return None
        
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期', ascending=False).reset_index(drop=True)
        
        # 基础过滤
        last_price = df.loc[0, '收盘']
        last_turnover = float(df.loc[0, '成交额'])
        if last_turnover < MIN_TURNOVER: return None
            
        # 计算指标
        df = calculate_indicators(df)
        
        # 连跌统计
        count, total_drop_pct = 0, 0.0
        for i in range(len(df)):
            if float(df.loc[i, '涨跌幅']) < 0:
                count += 1
                total_drop_pct += float(df.loc[i, '涨跌幅'])
            else: break
        
        # MA5 & 量比
        ma5 = df.loc[0:AVG_DAYS-1, '收盘'].mean()
        bias = round(((last_price - ma5) / ma5) * 100, 2)
        avg_vol = df.loc[1:AVG_DAYS, '成交量'].mean()
        vol_ratio = round(df.loc[0, '成交量'] / avg_vol, 2) if avg_vol > 0 else 0
        
        # 提取最新指标
        rsi_val = round(df.loc[0, 'RSI'], 2)
        j_val = round(df.loc[0, 'J'], 2)
        turnover = round(df.loc[0, '换手率'], 2)

        # 决策逻辑升级：加入RSI和J值过滤
        decision = "观察"
        # 极度超跌：连跌+RSI超卖(<30)+J值超卖(<0)+偏离MA5
        if 3 <= count <= 5 and bias < -2.5 and (rsi_val < 35 or j_val < 0):
            decision = "★★★ 极度超跌(黄金坑)"
        elif 3 <= count <= 5:
            decision = f"连跌{count}天(监控中)"
        elif rsi_val < 30:
            decision = "超卖区域(博反弹)"

        return {
            '代码': code, '名称': name_mapping.get(code, "未知"),
            '决策建议': decision, '现价': last_price,
            '连跌天数': count, 'RSI': rsi_val, 'KDJ_J': j_val, '换手率%': turnover,
            '连跌%': round(total_drop_pct, 2), 'MA5偏离%': bias, '量比': vol_ratio,
            '成交额(万)': round(last_turnover/10000, 2), '日期': df.loc[0, '日期'].strftime('%Y-%m-%d')
        }
    except: return None

def track_performance(current_results):
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
                        perf_list.append({
                            '决策日期': h_date, '代码': code, '名称': row['名称'],
                            '当时建议': row['决策建议'], '当时价': row['现价'],
                            '今日价': current_prices[code], 
                            '盈亏%': round(((current_prices[code] - row['现价']) / row['现价']) * 100, 2)
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

    res_df = pd.DataFrame(results)
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
    print(f"分析完成。RSI/KDJ已加入精选逻辑。")

if __name__ == "__main__":
    main()
