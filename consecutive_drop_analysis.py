import os
import pandas as pd
import glob
import re
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# 配置路径
DATA_DIR = 'fund_data'
ETF_LIST_FILE = 'ETF列表.txt'

def get_target_codes():
    """从包含简称的文本中提取纯数字代码"""
    if not os.path.exists(ETF_LIST_FILE):
        print(f"警告: 未找到 {ETF_LIST_FILE}")
        return None
    
    target_codes = set()
    for enc in ['utf-8', 'gbk', 'utf-16']:
        try:
            with open(ETF_LIST_FILE, 'r', encoding=enc) as f:
                for line in f:
                    # 使用正则匹配 6 位数字
                    match = re.search(r'(\d{6})', line)
                    if match:
                        target_codes.add(match.group(1))
            if target_codes:
                print(f"成功加载目标列表，共计 {len(target_codes)} 个标的")
                return target_codes
        except:
            continue
    return None

def analyze_file(file_path):
    """核心分析逻辑"""
    try:
        # 只读取关键列，提升速度
        df = pd.read_csv(file_path, usecols=['日期', '涨跌幅'])
        if df.empty:
            return None
        
        # 转换日期并倒序排列（最新日期在前）
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期', ascending=False).reset_index(drop=True)
        
        count = 0
        total_drop_pct = 0.0
        
        # 从第一行（最新日期）开始循环
        for i in range(len(df)):
            try:
                # 转换涨跌幅为浮点数（处理可能存在的空值或异常字符串）
                change = float(df.loc[i, '涨跌幅'])
                if change < 0:
                    count += 1
                    total_drop_pct += change
                else:
                    # 遇到上涨或平盘，中断循环
                    break
            except:
                break
        
        # 仅返回有连续下跌记录的数据
        if count > 0:
            # 从文件名提取 6 位代码
            code_match = re.search(r'(\d{6})', os.path.basename(file_path))
            code = code_match.group(1) if code_match else "Unknown"
            
            return {
                '代码': code,
                '连续下跌天数': count,
                '累计跌幅(%)': round(total_drop_pct, 2),
                '最后交易日': df.loc[0, '日期'].strftime('%Y-%m-%d')
            }
    except Exception:
        return None
    return None

def main():
    target_codes = get_target_codes()
    # 查找 fund_data 下的所有 csv
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    if not all_files:
        print(f"错误: 在 {DATA_DIR} 目录下没有找到任何 CSV 文件。")
        return

    # 匹配过滤
    tasks = []
    if target_codes:
        for f in all_files:
            file_name_code = re.search(r'(\d{6})', os.path.basename(f))
            if file_name_code and file_name_code.group(1) in target_codes:
                tasks.append(f)
    else:
        tasks = all_files

    print(f"开始并行处理 {len(tasks)} 个匹配的 CSV 文件...")

    results = []
    # 使用并行计算加速
    with ProcessPoolExecutor() as executor:
        for res in executor.map(analyze_file, tasks):
            if res:
                results.append(res)

    if results:
        res_df = pd.DataFrame(results).sort_values(by=['连续下跌天数', '累计跌幅(%)'], ascending=[False, True])
        res_df.to_csv('temp_result.csv', index=False, encoding='utf-8-sig')
        print(f"分析成功: 发现 {len(results)} 个连续下跌标的。")
    else:
        print("分析完成: 未发现符合连续下跌条件的标的。")

if __name__ == "__main__":
    main()
