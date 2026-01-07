import os
import pandas as pd
import glob
import re
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# --- 交易逻辑配置 ---
DATA_DIR = 'fund_data'
ETF_LIST_FILE = 'ETF列表.txt'
MIN_TURNOVER = 1000000       
AVG_DAYS = 5                 
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
        
        last_price = df.loc[0, '收盘']
        last_turnover = float(df.loc[0, '成交额'])
        if last_turnover < MIN_TURNOVER: return None
            
        count, total_drop_pct = 0, 0.0
        for i in range(len(df)):
            change = float(df.loc[i, '涨跌幅'])
            if change < 0:
                count += 1
                total_drop_pct += change
            else: break
        
        ma5 = df.loc[0:AVG_DAYS-1, '收盘'].mean()
        bias = round(((last_price - ma5) / ma5) * 100, 2)
        avg_vol = df.loc[1:AVG_DAYS, '成交量'].mean()
        vol_ratio = round(df.loc[0, '成交量'] / avg_vol, 2) if avg_vol > 0 else 0
        
        def get_period_change(days):
            if len(df) > days:
                prev = df.loc[days, '收盘']
                return round(((last_price - prev) / prev) * 100, 2)
            return None

        decision = "观察"
        if count >= 3 and bias < -3 and vol_ratio < 0.8:
            decision = "★★★ 极度超跌(买入建议)"
        elif bias < -1.5 and vol_ratio < 1.0:
            decision = "★★ 缩量回调(关注)"
        elif vol_ratio > 1.5 and last_price < ma5:
            decision = "放量下跌(风险)"

        return {
            '代码': code, '名称': name_mapping.get(code, "未知"),
            '决策建议': decision, '现价': last_price,
            '连跌天数': count, '连跌%': round(total_drop_pct, 2),
            '周幅%': get_period_change(5), '月幅%': get_period_change(20), '年幅%': get_period_change(250),
            'MA5偏离%': bias, '量比': vol_ratio, '成交额(万)': round(last_turnover/10000, 2),
            '日期': df.loc[0, '日期'].strftime('%Y-%m-%d')
        }
    except: return None

def track_performance(current_results):
    """回测逻辑：确保即使没有数据也创建文件"""
    current_prices = {item['代码']: item['现价'] for item in current_results}
    history_files = sorted(glob.glob(os.path.join('history', '**', 'decision_*.csv'), recursive=True), reverse=True)
    
    performance_list = []
    if history_files:
        for h_file in history_files[:5]:
            try:
                h_df = pd.read_csv(h_file)
                h_date = h_df['日期'].iloc[0]
                h_targets = h_df[h_df['决策建议'].str.contains('★')]
                for _, row in h_targets.iterrows():
                    code = str(row['代码']).zfill(6)
                    if code in current_prices:
                        old_price = row['现价']
                        now_price = current_prices[code]
                        performance_list.append({
                            '决策日期': h_date, '代码': code, '名称': row['名称'],
                            '当时建议': row['决策建议'], '建议时价格': old_price,
                            '今日价格': now_price, '盈亏幅%': round(((now_price - old_price) / old_price) * 100, 2)
                        })
            except: continue

    # 重点改进：无论是否有数据，都生成文件，防止 YML 报错
    perf_df = pd.DataFrame(performance_list if performance_list else [], 
                          columns=['决策日期', '代码', '名称', '当时建议', '建议时价格', '今日价格', '盈亏幅%'])
    perf_df.to_csv('performance_tracking.csv', index=False, encoding='utf-8-sig')
    print(f"性能跟踪已更新，共计 {len(performance_list)} 条记录。")

def main():
    name_mapping = get_target_mapping()
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    tasks = [(f, name_mapping) for f in all_files]

    results = []
    with ProcessPoolExecutor() as executor:
        for res in executor.map(analyze_file, tasks):
            if res: results.append(res)

    if results:
        res_df = pd.DataFrame(results).sort_values(by=['决策建议', 'MA5偏离%'], ascending=[False, True])
        res_df.to_csv('investment_decision.csv', index=False, encoding='utf-8-sig')
        
        # 存档历史
        now = datetime.now()
        history_dir = os.path.join('history', now.strftime('%Y'), now.strftime('%m'))
        os.makedirs(history_dir, exist_ok=True)
        history_file = os.path.join(history_dir, f"decision_{now.strftime('%Y%m%d_%H%M%S')}.csv")
        res_df.to_csv(history_file, index=False, encoding='utf-8-sig')
        
        # 运行跟踪
        track_performance(results)
    else:
        # 如果今天没结果，也要创建一个空的决策表，防止 YML git add 失败
        pd.DataFrame(columns=['代码','名称','决策建议']).to_csv('investment_decision.csv', index=False)
        print("未发现符合条件的标的。")

if __name__ == "__main__":
    main()
