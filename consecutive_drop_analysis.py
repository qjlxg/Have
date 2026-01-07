import os
import pandas as pd
import glob
import re
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# --- 交易逻辑配置 ---
DATA_DIR = 'fund_data'        # 存放ETF历史数据的目录
ETF_LIST_FILE = 'ETF列表.txt'   # 存放代码与名称对应关系的文本
MIN_TURNOVER = 1000000         # 流动性门槛：日成交额低于100万的标的直接过滤
AVG_DAYS = 5                   # 计算均线(MA5)和成交量比值的参考周期
# ------------------

def get_target_mapping():
    """解析ETF列表文件，建立 {代码: 名称} 的映射字典"""
    if not os.path.exists(ETF_LIST_FILE): return {}
    mapping = {}
    # 尝试多种常见编码读取
    for enc in ['utf-8', 'gbk', 'utf-16']:
        try:
            with open(ETF_LIST_FILE, 'r', encoding=enc) as f:
                for line in f:
                    line = line.strip()
                    if not line or "证券代码" in line: continue
                    # 使用正则匹配：6位数字代码 + 空格 + 名称
                    match = re.search(r'(\d{6})\s+(.+)', line)
                    if match:
                        code, name = match.groups()
                        mapping[code] = name.strip()
            if mapping: return mapping
        except: continue
    return {}

def analyze_file(file_info):
    """
    核心分析函数：计算连跌、多周期涨幅、偏离度并给出决策建议
    """
    file_path, name_mapping = file_info
    try:
        # 从文件名提取6位基金代码
        code_match = re.search(r'(\d{6})', os.path.basename(file_path))
        if not code_match: return None
        code = code_match.group(1)
        
        # 读取CSV数据，仅加载必要的列
        df = pd.read_csv(file_path, usecols=['日期', '收盘', '成交量', '成交额', '涨跌幅', '振幅'])
        if len(df) < 10: return None # 数据量太少则跳过
        
        # 日期预处理，按时间由新到旧排序
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期', ascending=False).reset_index(drop=True)
        
        # 1. 流动性过滤：如果最后交易日成交额不足，视为不可交易
        last_price = df.loc[0, '收盘']
        last_turnover = float(df.loc[0, '成交额'])
        if last_turnover < MIN_TURNOVER: return None
            
        # 2. 连续下跌天数统计 (从今天往回数，直到遇到涨的日子)
        count, total_drop_pct = 0, 0.0
        for i in range(len(df)):
            change = float(df.loc[i, '涨跌幅'])
            if change < 0:
                count += 1
                total_drop_pct += change
            else: break
        
        # 只要没有连跌，就跳过
        if count == 0: return None

        # 3. 技术指标计算
        # MA5 均线
        ma5 = df.loc[0:AVG_DAYS-1, '收盘'].mean()
        # MA5 偏离度 (BIAS): (现价 - 均价) / 均价
        bias = round(((last_price - ma5) / ma5) * 100, 2)
        # 量比: 今日成交量 / 过去5日平均成交量 (用于判断是否缩量)
        avg_vol = df.loc[1:AVG_DAYS, '成交量'].mean()
        vol_ratio = round(df.loc[0, '成交量'] / avg_vol, 2) if avg_vol > 0 else 0
        
        # 4. 多周期涨幅计算
        def get_period_change(days):
            if len(df) > days:
                prev = df.loc[days, '收盘']
                return round(((last_price - prev) / prev) * 100, 2)
            return None

        # 5. 自动决策逻辑 (重点：加入3-5天连跌统计)
        decision = "观察"
        # 黄金坑逻辑：连跌3-5天 + 严重偏离均线 + 缩量
        if 3 <= count <= 5 and bias < -2.5 and vol_ratio < 0.9:
            decision = "★★★ 极度超跌(黄金坑建议)"
        elif bias < -1.5 and vol_ratio < 1.0:
            decision = "★★ 缩量回调(关注)"
        elif vol_ratio > 1.5 and last_price < ma5:
            decision = "放量下跌(高风险)"

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
    """
    回测逻辑：读取历史存档，对比当日价格，统计决策胜率
    """
    current_prices = {item['代码']: item['现价'] for item in current_results}
    history_files = sorted(glob.glob(os.path.join('history', '**', 'decision_*.csv'), recursive=True), reverse=True)
    
    performance_list = []
    if history_files:
        for h_file in history_files[:10]: # 对比最近10次的历史记录
            try:
                h_df = pd.read_csv(h_file)
                h_date = h_df['日期'].iloc[0]
                # 找出历史文件中带有星号标记的建议
                h_targets = h_df[h_df['决策建议'].str.contains('★')]
                for _, row in h_targets.iterrows():
                    code = str(row['代码']).zfill(6)
                    if code in current_prices:
                        old_p, now_p = row['现价'], current_prices[code]
                        performance_list.append({
                            '决策日期': h_date, '代码': code, '名称': row['名称'],
                            '当时建议': row['决策建议'], '当时价': old_p,
                            '今日价': now_p, '盈亏%': round(((now_p - old_p) / old_p) * 100, 2)
                        })
            except: continue

    # 哪怕没有数据也生成CSV，防止GitHub Action add时报错
    perf_df = pd.DataFrame(performance_list if performance_list else [], 
                          columns=['决策日期', '代码', '名称', '当时建议', '当时价', '今日价', '盈亏%'])
    perf_df.to_csv('performance_tracking.csv', index=False, encoding='utf-8-sig')
    print(f"\n[回测统计] 匹配到 {len(performance_list)} 条历史记录。")

def main():
    name_mapping = get_target_mapping()
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    tasks = [(f, name_mapping) for f in all_files]

    results = []
    # 使用多进程并行处理文件以提高速度
    with ProcessPoolExecutor() as executor:
        for res in executor.map(analyze_file, tasks):
            if res: results.append(res)

    # 即使今日无数据，也生成空表头文件
    columns = ['代码','名称','决策建议','现价','连跌天数','连跌%','周幅%','月幅%','年幅%','MA5偏离%','量比','成交额(万)','日期']
    res_df = pd.DataFrame(results if results else [], columns=columns)
    
    if results:
        res_df = res_df.sort_values(by=['决策建议', 'MA5偏离%'], ascending=[False, True])
    
    # 保存今日决策报告
    res_df.to_csv('investment_decision.csv', index=False, encoding='utf-8-sig')
    
    # 历史归档：按年月建立文件夹保存备份
    now = datetime.now()
    history_dir = os.path.join('history', now.strftime('%Y'), now.strftime('%m'))
    os.makedirs(history_dir, exist_ok=True)
    history_file = os.path.join(history_dir, f"decision_{now.strftime('%Y%m%d_%H%M%S')}.csv")
    res_df.to_csv(history_file, index=False, encoding='utf-8-sig')
    
    # 执行历史表现跟踪
    track_performance(results)
    
    # 终端输出统计汇总
    count_3_5 = len(res_df[(res_df['连跌天数'] >= 3) & (res_df['连跌天数'] <= 5)])
    print(f"--- 分析报告完毕 ---")
    print(f"1. 连跌3-5天的标的数量: {count_3_5}")
    print(f"2. 极度超跌建议数量: {len(res_df[res_df['决策建议'].str.contains('极度')])}")
    print(f"3. 报告已保存至 history 文件夹")

if __name__ == "__main__":
    main()
