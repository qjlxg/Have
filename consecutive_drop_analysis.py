import os
import pandas as pd
import glob
import re
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# --- 交易逻辑配置 ---
DATA_DIR = 'fund_data'
ETF_LIST_FILE = 'ETF列表.txt'
MIN_TURNOVER = 1000000       # 硬性过滤：日成交额低于 100 万(元)
AVG_DAYS = 5                 # 计算 MA5 和量比的周期
# ------------------

def get_target_mapping():
    if not os.path.exists(ETF_LIST_FILE): return {}
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
        
        df = pd.read_csv(file_path, usecols=['日期', '收盘', '成交量', '成交额', '涨跌幅', '振幅'])
        if len(df) < 10: return None
        
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期', ascending=False).reset_index(drop=True)
        
        # 1. 基础过滤：流动性
        last_turnover = float(df.loc[0, '成交额'])
        if last_turnover < MIN_TURNOVER: return None
            
        # 2. 连续下跌（天数 & 累跌）
        count = 0
        total_drop_pct = 0.0
        for i in range(len(df)):
            change = float(df.loc[i, '涨跌幅'])
            if change < 0:
                count += 1
                total_drop_pct += change
            else: break
        
        if count == 0: return None

        # 3. 均线与量比
        ma5 = df.loc[0:AVG_DAYS-1, '收盘'].mean()
        last_price = df.loc[0, '收盘']
        bias = round(((last_price - ma5) / ma5) * 100, 2)
        avg_vol = df.loc[1:AVG_DAYS, '成交量'].mean()
        vol_ratio = round(df.loc[0, '成交量'] / avg_vol, 2) if avg_vol > 0 else 0
        
        # 4. 新增：多周期涨跌幅统计 (计算方式：(现价 - N天前价) / N天前价)
        def get_period_change(days):
            if len(df) > days:
                prev_price = df.loc[days, '收盘']
                return round(((last_price - prev_price) / prev_price) * 100, 2)
            return None

        week_change = get_period_change(5)    # 周
        month_change = get_period_change(20)  # 月
        year_change = get_period_change(250)  # 年（约250个交易日）

        # 5. 自动决策逻辑优化
        decision = "观察"
        if bias < -3 and vol_ratio < 0.8 and count >= 3:
            decision = "★★★ 极度超跌(买入建议)"
        elif bias < -1.5 and vol_ratio < 1.0:
            decision = "★★ 缩量回调(关注)"
        elif vol_ratio > 1.5 and last_price < ma5:
            decision = "放量下跌(风险)"

        return {
            '代码': code,
            '名称': name_mapping.get(code, "未知"),
            '连跌天数': count,
            '连跌%': round(total_drop_pct, 2),
            '周幅%': week_change,
            '月幅%': month_change,
            '年幅%': year_change,
            'MA5偏离%': bias,
            '量比': vol_ratio,
            '成交额(万)': round(last_turnover / 10000, 2),
            '决策建议': decision,
            '日期': df.loc[0, '日期'].strftime('%Y-%m-%d')
        }
    except Exception: return None

def main():
    name_mapping = get_target_mapping()
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    tasks = [(f, name_mapping) for f in all_files]

    results = []
    with ProcessPoolExecutor() as executor:
        for res in executor.map(analyze_file, tasks):
            if res: results.append(res)

    if results:
        res_df = pd.DataFrame(results).sort_values(
            by=['决策建议', 'MA5偏离%', '连跌天数'], 
            ascending=[False, True, False]
        )
        # 调整列顺序，更直观
        res_df = res_df[['代码', '名称', '决策建议', '连跌天数', '连跌%', '周幅%', '月幅%', '年幅%', 'MA5偏离%', '量比', '成交额(万)', '日期']]
        res_df.to_csv('investment_decision.csv', index=False, encoding='utf-8-sig')
        print(f"分析完成，已生成多周期决策报告。")

if __name__ == "__main__":
    main()
