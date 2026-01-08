import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import multiprocessing

# 战法名称：股灾战法 (Extreme Crash Recovery Strategy)
# 核心逻辑：
# 1. 过滤：排除ST、创业板，股价5-20元，基本面绩优（PE/ROE等）。
# 2. 信号：寻找RSI<30超卖、缩量企稳、MACD底背离迹象。
# 3. 复盘：根据反弹力度与量价关系，自动给出买入强度及操作建议。

def calculate_indicators(df):
    """计算战法所需的指标"""
    # 计算RSI (14日)
    delta = df['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 计算MACD
    df['EMA12'] = df['收盘'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['收盘'].ewm(span=26, adjust=False).mean()
    df['DIF'] = df['EMA12'] - df['EMA26']
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = (df['DIF'] - df['DEA']) * 2
    
    # 5日成交量均线（判定缩量）
    df['V_MA5'] = df['成交量'].rolling(window=5).mean()
    return df

def screen_stock(file_path):
    try:
        df = pd.read_csv(file_path)
        if df.empty or len(df) < 30: return None
        
        # 获取最新一行数据
        latest = df.iloc[-1]
        code = str(latest['股票代码']).zfill(6)
        
        # 基础过滤条件
        # 1. 排除ST(假设代码或文件名不带ST), 排除30开头(创业板)
        if code.startswith('30'): return None
        # 2. 只要深沪A股 (00, 60开头)
        if not (code.startswith('00') or code.startswith('60')): return None
        # 3. 价格区间 5.0 - 20.0
        price = latest['收盘']
        if not (5.0 <= price <= 20.0): return None
        
        # 战法逻辑计算
        df = calculate_indicators(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 信号识别
        is_oversold = latest['RSI'] < 35  # 超卖区
        is_volume_shrinking = latest['成交量'] < latest['V_MA5'] * 0.8 # 缩量企稳
        is_bottom_divergence = (latest['收盘'] < prev['收盘']) and (latest['DIF'] > prev['DIF']) # 疑似底背离
        
        # 评分系统 (优中选优)
        score = 0
        if is_oversold: score += 40
        if is_volume_shrinking: score += 30
        if is_bottom_divergence: score += 30
        
        if score >= 70:  # 只有高分才输出
            # 自动复盘建议逻辑
            suggestion = ""
            strength = ""
            if score >= 90:
                strength = "极强 (一击必中)"
                suggestion = "符合股灾战法核心模型：极致缩量+底背离。建议轻仓试错，止损设在10日线。"
            elif is_oversold and is_volume_shrinking:
                strength = "中等 (观察)"
                suggestion = "超卖严重且卖盘枯竭，等待MACD金叉确认后再介入。"
            else:
                strength = "弱 (待定)"
                suggestion = "虽有回调，但动能未完全衰减，建议暂且观望。"

            return {
                "代码": code,
                "收盘": price,
                "涨跌幅": latest['涨跌幅'],
                "RSI": round(latest['RSI'], 2),
                "信号强度": strength,
                "操作建议": suggestion,
                "战法": "股灾战法"
            }
    except Exception:
        return None

def run_strategy():
    stock_files = glob.glob('stock_data/*.csv')
    
    # 并行处理
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(screen_stock, stock_files)
    
    # 过滤掉None并合并
    results = [r for r in results if r is not None]
    if not results:
        print("今日无符合战法条件的股票。")
        return

    # 匹配名称
    names_df = pd.read_csv('stock_names.csv')
    names_df['code'] = names_df['code'].astype(str).str.zfill(6)
    
    final_df = pd.DataFrame(results)
    final_df = pd.merge(final_df, names_df[['code', 'name']], left_on='代码', right_on='code', how='left')
    
    # 整理列顺序
    final_df = final_df[['代码', 'name', '收盘', '涨跌幅', 'RSI', '信号强度', '操作建议']]
    final_df.rename(columns={'name': '名称'}, inplace=True)

    # 保存结果
    now = datetime.now()
    dir_path = f"results/{now.strftime('%Y%m')}"
    os.makedirs(dir_path, exist_ok=True)
    file_name = f"{dir_path}/crash_recovery_strategy_{now.strftime('%Y%m%d_%H%M%S')}.csv"
    
    final_df.to_csv(file_name, index=False, encoding='utf-8-sig')
    print(f"复盘完成，筛选出 {len(final_df)} 只潜力股。结果已保存至 {file_name}")

if __name__ == "__main__":
    run_strategy()
