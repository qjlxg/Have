import os
import pandas as pd
import glob
import re
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# --- 配置区 ---
DATA_DIR = 'fund_data'
ETF_LIST_FILE = 'ETF列表.txt'
MIN_TURNOVER = 1000000  # 过滤门槛：日成交额低于 100 万(元)的直接排除
# --------------

def get_target_mapping():
    if not os.path.exists(ETF_LIST_FILE):
        return {}
    mapping = {}
    for enc in ['utf-8', 'gbk', 'utf-16']:
        try:
            with open(ETF_LIST_FILE, 'r', encoding=enc) as f:
                for line in f:
                    line = line.strip()
                    if not line or "证券代码" in line: continue
                    match = re.search(r'(\d{6})\s+(.+)', line)
                    if match:
                        code, name = match.groups()
                        mapping[code] = name.strip()
            if mapping: return mapping
        except: continue
    return {}

def analyze_file(file_info):
    file_path, name_mapping = file_info
    try:
        code_match = re.search(r'(\d{6})', os.path.basename(file_path))
        if not code_match: return None
        code = code_match.group(1)
        
        # 加载数据，增加成交量和成交额列
        df = pd.read_csv(file_path, usecols=['日期', '涨跌幅', '成交量', '成交额'])
        if df.empty: return None
        
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期', ascending=False).reset_index(drop=True)
        
        # 1. 过滤退市风险/流动性风险：检查最新一天的成交额
        last_turnover = float(df.loc[0, '成交额'])
        if last_turnover < MIN_TURNOVER:
            return None
            
        # 2. 计算连续下跌
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
            except: break
        
        if count > 0:
            return {
                '代码': code,
                '名称': name_mapping.get(code, "未知"),
                '连续下跌天数': count,
                '累计跌幅(%)': round(total_drop_pct, 2),
                '成交额(元)': round(last_turnover, 2),
                '成交量(手)': df.loc[0, '成交量'],
                '最后交易日': df.loc[0, '日期'].strftime('%Y-%m-%d')
            }
    except Exception: return None
    return None

def main():
    name_mapping = get_target_mapping()
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not all_files: return

    tasks = []
    target_codes = set(name_mapping.keys())
    for f in all_files:
        code_match = re.search(r'(\d{6})', os.path.basename(f))
        if code_match and code_match.group(1) in target_codes:
            tasks.append((f, name_mapping))

    results = []
    with ProcessPoolExecutor() as executor:
        for res in executor.map(analyze_file, tasks):
            if res: results.append(res)

    if results:
        res_df = pd.DataFrame(results).sort_values(by=['连续下跌天数', '累计跌幅(%)'], ascending=[False, True])
        # 重新排序列，确保直观
        res_df = res_df[['代码', '名称', '连续下跌天数', '累计跌幅(%)', '成交额(元)', '成交量(手)', '最后交易日']]
        res_df.to_csv('temp_result.csv', index=False, encoding='utf-8-sig')
        print(f"分析完成: 筛选出 {len(results)} 个活跃且连续下跌的标的。")
    else:
        print("未发现符合条件的标的（可能已被成交额过滤）。")

if __name__ == "__main__":
    main()
