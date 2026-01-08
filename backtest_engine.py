import os
import pandas as pd
import glob
import re
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# --- 回测参数配置 ---
DATA_DIR = 'fund_data'
HOLD_DAYS = [3, 5, 10]    # 模拟持仓周期
MIN_SCORE_THRESHOLD = 80  # 触发买入的最低评分
MIN_TURNOVER = 5000000    # 过滤成交额过低的日期

def calculate_backtest_tech(df):
    """为历史每一天计算技术指标 (正序)"""
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
    df['MA5'] = df['收盘'].rolling(5).mean()
    df['BIAS'] = (df['收盘'] - df['MA5']) / df['MA5'] * 100
    df['V_MA5'] = df['成交量'].shift(1).rolling(5).mean()
    df['VOL_RATIO'] = df['成交量'] / df['V_MA5']
    # 250日年涨跌幅 (空间因子)
    df['Y_CHG'] = df['收盘'].pct_change(250) * 100
    return df

def process_file_backtest(file_path):
    """模拟单只ETF的历史交易"""
    trades = []
    try:
        code = re.search(r'(\d{6})', os.path.basename(file_path)).group(1)
        df = pd.read_csv(file_path)
        if len(df) < 260: return []
        
        df = calculate_backtest_tech(df)
        
        # 模拟历史：从第30天开始，直到留出最长持仓周期
        for i in range(30, len(df) - max(HOLD_DAYS)):
            row = df.iloc[i]
            
            # 过滤成交额
            if row['成交额'] < MIN_TURNOVER: continue
            
            # 评分逻辑
            score = 0
            # 情绪：是否连跌3天
            if all(df.iloc[i-j]['涨跌幅'] < 0 for j in range(3)): score += 20
            if row['RSI'] < 35: score += 20
            if row['J'] < 0: score += 20
            if row['Y_CHG'] < -15: score += 20
            if row['BIAS'] < -2.5: score += 20
            
            if score >= MIN_SCORE_THRESHOLD:
                buy_price = row['收盘']
                res = {'代码': code, '日期': row['日期'], '评分': score}
                for d in HOLD_DAYS:
                    sell_price = df.iloc[i + d]['收盘']
                    res[f'{d}日收益%'] = round((sell_price - buy_price) / buy_price * 100, 2)
                trades.append(res)
    except: pass
    return trades

def main():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"开始并行回测 {len(files)} 个文件...")
    
    all_trades = []
    with ProcessPoolExecutor() as executor:
        for result in executor.map(process_file_backtest, files):
            all_trades.extend(result)
            
    if all_trades:
        res_df = pd.DataFrame(all_trades)
        res_df.to_csv('backtest_detail.csv', index=False, encoding='utf-8-sig')
        
        # 统计汇总
        stats = []
        for d in HOLD_DAYS:
            col = f'{d}日收益%'
            win_rate = (res_df[col] > 0).mean() * 100
            avg_ret = res_df[col].mean()
            stats.append({'持仓周期': f'{d}天', '信号次数': len(res_df), '胜率%': f'{win_rate:.2f}', '平均收益%': f'{avg_ret:.2f}'})
        
        stats_df = pd.DataFrame(stats)
        stats_df.to_csv('backtest_summary.csv', index=False, encoding='utf-8-sig')
        print("\n" + "="*20 + " 回测概览 " + "="*20)
        print(stats_df.to_string(index=False))
    else:
        print("未发现任何符合条件的交易信号。")

if __name__ == "__main__":
    main()
