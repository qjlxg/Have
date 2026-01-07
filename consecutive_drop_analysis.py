import os
import pandas as pd
import glob
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# 配置路径
DATA_DIR = 'fund_data'
ETF_LIST_FILE = 'ETF列表.txt'

def get_target_codes():
    """读取目标ETF列表，兼容不同编码"""
    if not os.path.exists(ETF_LIST_FILE):
        print(f"File not found: {ETF_LIST_FILE}")
        return None
    for enc in ['utf-8', 'gbk', 'utf-16']:
        try:
            with open(ETF_LIST_FILE, 'r', encoding=enc) as f:
                # 提取代码，去除 .csv 后缀和空格
                return set(line.strip().replace('.csv', '') for line in f if line.strip())
        except:
            continue
    return None

def analyze_file(file_path):
    """核心逻辑：统计连续下跌天数和幅度"""
    try:
        # 读取数据，指定日期解析
        df = pd.read_csv(file_path)
        if df.empty:
            return None
        
        # 统一表头格式（去除空格）
        df.columns = [c.strip() for c in df.columns]
        df['日期'] = pd.to_datetime(df['日期'])
        
        # 按日期倒序（最新日期在第一行）
        df = df.sort_values('日期', ascending=False).reset_index(drop=True)
        
        count = 0
        total_drop_pct = 0.0
        
        # 遍历数据：从最新日期开始
        for i in range(len(df)):
            try:
                # 强制转为浮点数
                change = float(df.loc[i, '涨跌幅'])
                if change < 0:
                    count += 1
                    total_drop_pct += change
                else:
                    # 只要不小于0（上涨或平盘）即停止统计
                    break
            except (ValueError, TypeError):
                break
        
        # 只有在最新状态是下跌时才返回结果
        if count > 0:
            code = os.path.basename(file_path).replace('.csv', '')
            return {
                '基金代码': code,
                '连续下跌天数': count,
                '累计跌幅(%)': round(total_drop_pct, 2),
                '最近交易日': df.loc[0, '日期'].strftime('%Y-%m-%d')
            }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return None

def main():
    target_codes = get_target_codes()
    # 获取 fund_data 目录下所有 csv
    all_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    
    # 筛选出在列表中的文件
    if target_codes:
        tasks = [f for f in all_files if os.path.basename(f).replace('.csv', '') in target_codes]
    else:
        tasks = all_files

    if not tasks:
        print("No matching CSV files found.")
        return

    # 并行处理加速
    results = []
    with ProcessPoolExecutor() as executor:
        for res in executor.map(analyze_file, tasks):
            if res:
                results.append(res)

    if results:
        res_df = pd.DataFrame(results).sort_values('连续下跌天数', ascending=False)
        res_df.to_csv('temp_result.csv', index=False, encoding='utf-8-sig')
        print(f"Success: Found {len(results)} matching funds.")
    else:
        print("No funds found with consecutive drops.")

if __name__ == "__main__":
    main()
