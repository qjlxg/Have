import os
import pandas as pd
import glob
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# 配置路径
DATA_DIR = 'fund_data'
ETF_LIST_FILE = 'ETF列表.txt'
OUTPUT_BASE = 'results'

def get_target_codes():
    if not os.path.exists(ETF_LIST_FILE):
        return None
    with open(ETF_LIST_FILE, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

def analyze_file(file_path):
    try:
        df = pd.read_csv(file_path)
        if df.empty or len(df) < 2:
            return None
        
        # 确保按日期降序排列（最新在前）
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期', ascending=False).reset_index(drop=True)
        
        count = 0
        total_drop_pct = 0.0
        
        # 计算连续下跌（今日收盘 < 昨日收盘）
        # 注意：CSV中 涨跌幅 字段已经是百分比，如 -1.3 表示 -1.3%
        for i in range(len(df) - 1):
            change_pct = df.loc[i, '涨跌幅']
            if change_pct < 0:
                count += 1
                total_drop_pct += change_pct
            else:
                break
        
        if count > 0:
            code = os.path.basename(file_path).replace('.csv', '')
            return {
                '基金代码': code,
                '连续下跌天数': count,
                '累计跌幅(%)': round(total_drop_pct, 2),
                '最后交易日': df.loc[0, '日期'].strftime('%Y-%m-%d')
            }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return None

def main():
    target_codes = get_target_codes()
    files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    
    # 筛选匹配 ETF列表 的文件
    if target_codes:
        files = [f for f in files if os.path.basename(f).replace('.csv', '') in target_codes]

    results = []
    # 使用进程池并行处理
    with ProcessPoolExecutor() as executor:
        for res in executor.map(analyze_file, files):
            if res:
                results.append(res)

    if results:
        res_df = pd.DataFrame(results).sort_values('连续下跌天数', ascending=False)
        # 生成输出内容
        return res_df
    return None

if __name__ == "__main__":
    analysis_res = main()
    if analysis_res is not None:
        # 结果由 Workflow 处理保存路径，此处直接打印或存为临时文件
        analysis_res.to_csv('temp_result.csv', index=False, encoding='utf-8-sig')
        print("Analysis completed successfully.")
