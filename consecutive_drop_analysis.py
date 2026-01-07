import os
import pandas as pd
import glob
import re
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# 配置路径
DATA_DIR = 'fund_data'
ETF_LIST_FILE = 'ETF列表.txt'

def get_target_mapping():
    """从文本中提取代码和简称的映射关系"""
    if not os.path.exists(ETF_LIST_FILE):
        print(f"警告: 未找到 {ETF_LIST_FILE}")
        return {}
    
    mapping = {}
    for enc in ['utf-8', 'gbk', 'utf-16']:
        try:
            with open(ETF_LIST_FILE, 'r', encoding=enc) as f:
                for line in f:
                    line = line.strip()
                    if not line or "证券代码" in line:
                        continue
                    # 匹配 6 位数字代码和紧随其后的名称（支持空格或制表符分隔）
                    match = re.search(r'(\d{6})\s+(.+)', line)
                    if match:
                        code, name = match.groups()
                        mapping[code] = name.strip()
            if mapping:
                print(f"成功加载目标列表，共计 {len(mapping)} 个标的")
                return mapping
        except:
            continue
    return {}

def analyze_file(file_info):
    """核心分析逻辑"""
    file_path, name_mapping = file_info
    try:
        # 提取文件名中的 6 位代码
        code_match = re.search(r'(\d{6})', os.path.basename(file_path))
        if not code_match:
            return None
        code = code_match.group(1)
        
        # 只读取日期和涨跌幅列
        df = pd.read_csv(file_path, usecols=['日期', '涨跌幅'])
        if df.empty:
            return None
        
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期', ascending=False).reset_index(drop=True)
        
        count = 0
        total_drop_pct = 0.0
        
        for i in range(len(df)):
            try:
                change = float(df.loc[i, '涨跌幅'])
                if change < 0:
                    count += 1
                    total_drop_pct += change
                else:
                    break
            except:
                break
        
        if count > 0:
            return {
                '代码': code,
                '名称': name_mapping.get(code, "未知名称"), # 从映射表中取名
                '连续下跌天数': count,
                '累计跌幅(%)': round(total_drop_pct, 2),
                '最后交易日': df.loc[0, '日期'].strftime('%Y-%m-%d')
            }
    except Exception:
        return None
    return None

def main():
    name_mapping = get_target_mapping()
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    if not all_files:
        print(f"错误: 在 {DATA_DIR} 目录下没有找到任何 CSV 文件。")
        return

    # 匹配过滤：只处理在列表中的文件
    tasks = []
    target_codes = set(name_mapping.keys())
    for f in all_files:
        code_match = re.search(r'(\d{6})', os.path.basename(f))
        if code_match and code_match.group(1) in target_codes:
            # 将文件路径和映射表作为元组传给并行函数
            tasks.append((f, name_mapping))

    print(f"开始并行处理 {len(tasks)} 个匹配的 CSV 文件...")

    results = []
    with ProcessPoolExecutor() as executor:
        for res in executor.map(analyze_file, tasks):
            if res:
                results.append(res)

    if results:
        # 排序：按天数倒序，跌幅由大到小
        res_df = pd.DataFrame(results).sort_values(by=['连续下跌天数', '累计跌幅(%)'], ascending=[False, True])
        # 调整列顺序，让名称排在代码后面
        res_df = res_df[['代码', '名称', '连续下跌天数', '累计跌幅(%)', '最后交易日']]
        res_df.to_csv('temp_result.csv', index=False, encoding='utf-8-sig')
        print(f"分析成功: 发现 {len(results)} 个连续下跌标的。")
    else:
        print("分析完成: 未发现符合连续下跌条件的标的。")

if __name__ == "__main__":
    main()
