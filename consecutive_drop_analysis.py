import os
import pandas as pd
import glob
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# 配置路径
DATA_DIR = 'fund_data'
ETF_LIST_FILE = 'ETF列表.txt'

def get_target_codes():
    if not os.path.exists(ETF_LIST_FILE):
        print(f"警告: 找不到 {ETF_LIST_FILE}")
        return None
    # 尝试多种编码读取列表
    for enc in ['utf-8', 'gbk', 'utf-16']:
        try:
            with open(ETF_LIST_FILE, 'r', encoding=enc) as f:
                codes = set(line.strip().replace('.csv', '') for line in f if line.strip())
                print(f"成功读取列表，包含 {len(codes)} 个目标")
                return codes
        except:
            continue
    return None

def analyze_file(file_path):
    try:
        # 只读取必要的列以加快速度
        df = pd.read_csv(file_path, usecols=['日期', '涨跌幅'], parse_dates=['日期'])
        if df.empty or len(df) < 1:
            return None
        
        # 按日期倒序排列
        df = df.sort_values('日期', ascending=False).reset_index(drop=True)
        
        count = 0
        total_drop_pct = 0.0
        
        # 计算逻辑：从最新的一天开始往回数
        for i in range(len(df)):
            change_pct = float(df.loc[i, '涨跌幅'])
            if change_pct < 0:
                count += 1
                total_drop_pct += change_pct
            else:
                # 遇到非负（上涨或平盘）立即停止
                break
        
        # 只有连续下跌（至少1天）才记录
        if count > 0:
            code = os.path.basename(file_path).replace('.csv', '')
            return {
                '代码': code,
                '连续下跌天数': count,
                '累计跌幅(%)': round(total_drop_pct, 2),
                '最后交易日': df.loc[0, '日期'].strftime('%Y-%m-%d')
            }
    except Exception as e:
        pass # 忽略格式错误的csv
    return None

def main():
    target_codes = get_target_codes()
    # 查找 fund_data 下的所有 csv
    all_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    
    # 并行筛选匹配的文件
    tasks = []
    if target_codes:
        tasks = [f for f in all_files if os.path.basename(f).replace('.csv', '') in target_codes]
    else:
        tasks = all_files

    if not tasks:
        print("没有找到匹配待处理的文件。")
        return

    results = []
    with ProcessPoolExecutor() as executor:
        for res in executor.map(analyze_file, tasks):
            if res:
                results.append(res)

    if results:
        res_df = pd.DataFrame(results).sort_values(['连续下跌天数', '累计跌幅(%)'], ascending=[False, True])
        res_df.to_csv('temp_result.csv', index=False, encoding='utf-8-sig')
        print(f"分析完成，发现 {len(results)} 个连续下跌标的。")
    else:
        print("未发现符合连续下跌条件的标的。")

if __name__ == "__main__":
    main()
