import os
import pandas as pd
import glob
import re
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# --- 严苛策略配置 ---
DATA_DIR = 'fund_data'
HOLD_DAYS = [3, 5, 10]    
MIN_SCORE_THRESHOLD = 90  # 提升至90分：追求极高胜率
MIN_TURNOVER = 5000000    
VOL_RATIO_LIMIT = 0.7     # 极致缩量：成交量需低于5日均值的70%

def calculate_backtest_indicators(df):
    df = df.sort_values('日期').copy()
    
    # RSI (14)
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
    
    # BIAS & Volume
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
            
            # 基础过滤
            if row['成交额'] < MIN_TURNOVER: continue
            
            # --- 极致评分逻辑 ---
            score = 0
            # 1. 连跌3天
            if all(df.iloc[i-j]['涨跌幅'] < 0 for j in range(3)): score += 20
            # 2. RSI < 30
            if row['RSI'] < 30: score += 20
            # 3. J < 0
            if row['J'] < 0: score += 20
            # 4. 年跌幅 > 15%
            if row['Y_CHG'] < -15: score += 20
            # 5. MA5负乖离 > 2.5%
            if row['BIAS'] < -2.5: score += 20
            
            # --- 增加硬性条件：极致缩量 ---
            is_volume_shrink = row['VOL_RATIO'] < VOL_RATIO_LIMIT
            
            # 触发买入：必须满90分 且 属于地量
            if score >= MIN_SCORE_THRESHOLD and is_volume_shrink:
                buy_price = row['收盘']
                res = {'代码': code, '买入日期': row['日期'], '评分': score, '量比': round(row['VOL_RATIO'], 2)}
                for d in HOLD_DAYS:
                    sell_price = df.iloc[i + d]['收盘']
                    res[f'{d}日收益%'] = round((sell_price - buy_price) / buy_price * 100, 2)
                trades.append(res)
    except: pass
    return trades

def main():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"🚀 启动【极致缩量+90分】严选回测...")
    
    all_trades = []
    with ProcessPoolExecutor() as executor:
        for result in executor.map(run_single_backtest, files):
            all_results = result # 变量名微调确保一致
            all_trades.extend(all_results)
            
    if all_trades:
        res_df = pd.DataFrame(all_trades)
        res_df.to_csv('backtest_detail.csv', index=False, encoding='utf-8-sig')
        
        summary = []
        for d in HOLD_DAYS:
            col = f'{d}日收益%'
            win_rate = (res_df[col] > 0).mean() * 100
            avg_ret = res_df[col].mean()
            summary.append({
                '周期': f'持有{d}天',
                '严选信号数': len(res_df),
                '胜率%': round(win_rate, 2),
                '平均收益%': round(avg_ret, 2)
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv('backtest_summary.csv', index=False, encoding='utf-8-sig')
        print("\n" + "="*20 + " 严选回测统计 " + "="*20)
        print(summary_df.to_string(index=False))
        print("="*54)
    else:
        print("❌ 门槛过高，当前历史数据中未发现符合‘极致缩量90分’的信号。")

if __name__ == "__main__":
    main()
