import os
import pandas as pd
import glob
import re
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# --- 核心优化配置 ---
DATA_DIR = 'fund_data'
HOLD_DAYS = [3, 5, 10]    
MIN_SCORE_THRESHOLD = 80  # 稍微调低总分
STOP_LOSS = -0.05         # 5%硬止损

def calculate_backtest_indicators(df):
    df = df.sort_values('日期').copy()
    delta = df['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    low_9 = df['收盘'].rolling(9).min()
    high_9 = df['收盘'].rolling(9).max()
    rsv = (df['收盘'] - low_9) / (high_9 - low_9) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    df['MA5'] = df['收盘'].rolling(5).mean()
    df['BIAS'] = (df['收盘'] - df['MA5']) / df['MA5'] * 100
    return df

def run_single_backtest(file_path):
    trades = []
    try:
        code = re.search(r'(\d{6})', os.path.basename(file_path)).group(1)
        df = pd.read_csv(file_path)
        if len(df) < 100: return []
        df = calculate_backtest_indicators(df)
        
        for i in range(10, len(df) - 10):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            
            score = 0
            if row['RSI'] < 35: score += 25
            if row['J'] < 5: score += 25
            if row['BIAS'] < -2.0: score += 25
            if all(df.iloc[i-j]['涨跌幅'] < 0 for j in range(2)): score += 25
            
            # --- 核心优化：右侧确认 (J线向上拐头 或 今日收红线) ---
            is_turning_up = row['J'] > prev['J'] and row['收盘'] > row['开盘']
            
            if score >= MIN_SCORE_THRESHOLD and is_turning_up:
                buy_price = row['收盘']
                res = {'代码': code, '日期': row['日期']}
                for d in HOLD_DAYS:
                    period_data = df.iloc[i+1 : i+d+1]
                    if (period_data['最低'].min() - buy_price) / buy_price <= STOP_LOSS:
                        res[f'{d}日收益%'] = STOP_LOSS * 100
                    else:
                        res[f'{d}日收益%'] = round((df.iloc[i+d]['收盘'] - buy_price) / buy_price * 100, 2)
                trades.append(res)
    except: pass
    return trades

def main():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    all_trades = []
    with ProcessPoolExecutor() as executor:
        for result in executor.map(run_single_backtest, files):
            all_trades.extend(result)
            
    if all_trades:
        res_df = pd.DataFrame(all_trades)
        summary = []
        for d in HOLD_DAYS:
            col = f'{d}日收益%'
            win_rate = (res_df[col] > 0).mean() * 100
            avg_ret = res_df[col].mean()
            summary.append({'周期': f'持有{d}天', '信号数': len(res_df), '胜率%': round(win_rate, 2), '平均收益%': round(avg_ret, 2)})
        print(pd.DataFrame(summary).to_string(index=False))

if __name__ == "__main__":
    main()
