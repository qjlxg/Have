import os
import pandas as pd
import glob
import re
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# --- 回测配置 ---
DATA_DIR = 'fund_data'
HOLD_DAYS = [3, 5, 10]  # 测试不同持仓周期的收益
MIN_SCORE = 80          # 仅回测高分“一击必中”信号

def calculate_indicators(df):
    """为回测准备技术指标 (正序处理)"""
    df = df.sort_values('日期').copy()
    # RSI
    delta = df['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    # KDJ
    low_9 = df['收盘'].rolling(9).min()
    high_9 = df['收盘'].rolling(9).max()
    rsv = (td := df['收盘'] - low_9) / (high_9 - low_9) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    # 均线与偏离
    df['MA5'] = df['收盘'].rolling(5).mean()
    df['BIAS'] = (df['收盘'] - df['MA5']) / df['MA5'] * 100
    # 量比
    df['V_MA5'] = df['成交量'].shift(1).rolling(5).mean()
    df['VOL_RATIO'] = df['成交量'] / df['V_MA5']
    return df

def analyze_single_file(file_path):
    """单个文件的回测逻辑，供并行调用"""
    trades = []
    try:
        code = re.search(r'(\d{6})', os.path.basename(file_path)).group(1)
        df = pd.read_csv(file_path)
        if len(df) < 260: return []
        
        df = calculate_indicators(df)
        
        # 遍历历史 (跳过前30天指标稳定期)
        for i in range(30, len(df) - max(HOLD_DAYS)):
            score = 0
            # 简化版评分战法 (连跌 + RSI + J + BIAS + 缩量)
            # 检查过去3天是否连跌
            if i >= 3 and all(df.iloc[i-j]['涨跌幅'] < 0 for j in range(3)): score += 20
            if df.iloc[i]['RSI'] < 35: score += 20
            if df.iloc[i]['J'] < 0: score += 20
            if df.iloc[i]['BIAS'] < -2.5: score += 20
            if 0.4 < df.iloc[i]['VOL_RATIO'] < 0.9: score += 20
            
            if score >= MIN_SCORE:
                buy_price = df.iloc[i]['收盘']
                trade = {'代码': code, '买入日期': df.iloc[i]['日期']}
                for d in HOLD_DAYS:
                    sell_price = df.iloc[i + d]['收盘']
                    trade[f'{d}日收益%'] = round((sell_price - buy_price) / buy_price * 100, 2)
                trades.append(trade)
    except: pass
    return trades

def main():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"🚀 启动并行回测，目标文件数: {len(files)}")
    
    all_results = []
    with ProcessPoolExecutor() as executor:
        for result in executor.map(analyze_single_file, files):
            all_results.extend(result)
    
    if all_results:
        res_df = pd.DataFrame(all_results)
        res_df.to_csv('backtest_detail.csv', index=False, encoding='utf-8-sig')
        
        # 生成统计摘要
        summary = []
        for d in HOLD_DAYS:
            col = f'{d}日收益%'
            win_rate = (res_df[col] > 0).mean() * 100
            avg_ret = res_df[col].mean()
            summary.append({
                '持仓周期': f'{d}天',
                '总触发次数': len(res_df),
                '胜率%': round(win_rate, 2),
                '平均收益%': round(avg_ret, 2)
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv('backtest_summary.csv', index=False, encoding='utf-8-sig')
        print("\n" + "="*30 + "\n回测统计报告\n" + "="*30)
        print(summary_df.to_string(index=False))
        print("="*30)
    else:
        print("未发现符合条件的交易记录。")

if __name__ == "__main__":
    main()
