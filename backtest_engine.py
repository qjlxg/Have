import os
import pandas as pd
import glob
import re
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# --- 逻辑修正配置 ---
DATA_DIR = 'fund_data'
HOLD_DAYS = [3, 5, 10]    
MIN_SCORE_THRESHOLD = 85  # 门槛微调回85，增加样本量
MIN_TURNOVER = 5000000    
VOL_RATIO_UPPER = 1.0     # 缩量上限：不高于均量
VOL_RATIO_LOWER = 0.5     # 缩量下限：不能低于50%，防止流动性枯竭陷阱
STOP_LOSS = -0.05         # 强制止损逻辑：-5%

def calculate_backtest_indicators(df):
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
    # Vol & Bias
    df['MA5'] = df['收盘'].rolling(5).mean()
    df['BIAS'] = (df['收盘'] - df['MA5']) / df['MA5'] * 100
    df['V_MA5'] = df['成交量'].shift(1).rolling(5).mean()
    df['VOL_RATIO'] = df['成交量'] / df['V_MA5']
    df['Y_CHG'] = df['收盘'].pct_change(250) * 100
    return df

def run_single_backtest(file_path):
    trades = []
    try:
        code = re.search(r'(\d{6})', os.path.basename(file_path)).group(1)
        df = pd.read_csv(file_path)
        if len(df) < 260: return []
        df = calculate_backtest_indicators(df)
        
        for i in range(30, len(df) - max(HOLD_DAYS)):
            row = df.iloc[i]
            if row['成交额'] < MIN_TURNOVER: continue
            
            score = 0
            if all(df.iloc[i-j]['涨跌幅'] < 0 for j in range(3)): score += 20
            if row['RSI'] < 30: score += 20
            if row['J'] < 0: score += 20
            if row['Y_CHG'] < -15: score += 20
            if row['BIAS'] < -2.5: score += 20
            
            # 修正后的量比逻辑：温和缩量 (0.5 - 1.0 之间)
            is_vol_ok = VOL_RATIO_LOWER < row['VOL_RATIO'] < VOL_RATIO_UPPER
            
            if score >= MIN_SCORE_THRESHOLD and is_vol_ok:
                buy_price = row['收盘']
                res = {'代码': code, '买入日期': row['日期'], '评分': score}
                for d in HOLD_DAYS:
                    # 模拟真实持仓：期间如果跌破止损线，按止损价算
                    period_low = df.iloc[i+1 : i+d+1]['最低'].min()
                    if (period_low - buy_price) / buy_price <= STOP_LOSS:
                        res[f'{d}日收益%'] = STOP_LOSS * 100
                    else:
                        sell_price = df.iloc[i + d]['收盘']
                        res[f'{d}日收益%'] = round((sell_price - buy_price) / buy_price * 100, 2)
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
        res_df.to_csv('backtest_detail.csv', index=False, encoding='utf-8-sig')
        summary = []
        for d in HOLD_DAYS:
            col = f'{d}日收益%'
            win_rate = (res_df[col] > 0).mean() * 100
            avg_ret = res_df[col].mean()
            summary.append({'周期': f'持有{d}天', '信号数': len(res_df), '胜率%': round(win_rate, 2), '平均收益%': round(avg_ret, 2)})
        pd.DataFrame(summary).to_csv('backtest_summary.csv', index=False, encoding='utf-8-sig')
        print(pd.DataFrame(summary).to_string(index=False))
    else:
        print("未发现信号")

if __name__ == "__main__":
    main()
