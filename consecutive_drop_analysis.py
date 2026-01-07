import os
import pandas as pd
import glob
import re
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# --- 交易逻辑配置 ---
DATA_DIR = 'fund_data'        # 存放ETF历史CSV数据的目录
ETF_LIST_FILE = 'ETF列表.txt'   # 存放代码与名称对应关系的文本
MIN_TURNOVER = 1000000         # 流动性过滤：日成交额低于100万(元)的排除
AVG_DAYS = 5                   # 计算MA5均线和量比的参考周期
# ------------------

def get_target_mapping():
    """解析ETF列表文件，建立 {代码: 名称} 的字典"""
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
    """
    核心分析：计算连跌天数、多周期跌幅、MA5偏离度及自动决策
    """
    file_path, name_mapping = file_info
    try:
        # 获取文件名中的6位代码
        code_match = re.search(r'(\d{6})', os.path.basename(file_path))
        if not code_match: return None
        code = code_match.group(1)
        
        # 读取CSV，仅加载必要的列以节省内存
        df = pd.read_csv(file_path, usecols=['日期', '收盘', '成交量', '成交额', '涨跌幅', '振幅'])
        if len(df) < 10: return None
        
        # 确保数据按日期从新到旧排列
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期', ascending=False).reset_index(drop=True)
        
        last_price = df.loc[0, '收盘']
        last_turnover = float(df.loc[0, '成交额'])
        
        # 1. 流动性过滤：剔除成交额太小的标的
        if last_turnover < MIN_TURNOVER: return None
            
        # 2. 统计连续下跌天数
        count, total_drop_pct = 0, 0.0
        for i in range(len(df)):
            change = float(df.loc[i, '涨跌幅'])
            if change < 0:
                count += 1
                total_drop_pct += change
            else: break # 一旦遇到涨，统计结束

        # 3. 计算技术指标：MA5偏离度与量比
        ma5 = df.loc[0:AVG_DAYS-1, '收盘'].mean()
        bias = round(((last_price - ma5) / ma5) * 100, 2)
        avg_vol = df.loc[1:AVG_DAYS, '成交量'].mean()
        vol_ratio = round(df.loc[0, '成交量'] / avg_vol, 2) if avg_vol > 0 else 0
        
        # 4. 计算周、月、年涨跌幅
        def get_period_change(days):
            if len(df) > days:
                prev = df.loc[days, '收盘']
                return round(((last_price - prev) / prev) * 100, 2)
            return None

        # 5. 自动决策逻辑（加入连跌3-5天的统计和标记）
        decision = "观察"
        # 核心买入信号：连跌3-5天 + 严重偏离均线 + 缩量
        if 3 <= count <= 5 and bias < -2.5 and vol_ratio < 0.9:
            decision = "★★★ 极度超跌(黄金坑)"
        # 连跌监控信号：虽然还没到买入标准，但已连跌3-5天，需要在表格中高亮
        elif 3 <= count <= 5:
            decision = f"连跌{count}天(监控中)"
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
    """回测模块：读取历史CSV，对比现价计算历史建议的胜率"""
    current_prices = {item['代码']: item['现价'] for item in current_results}
    history_files = sorted(glob.glob(os.path.join('history', '**', 'decision_*.csv'), recursive=True), reverse=True)
    
    perf_list = []
    if history_files:
        for h_file in history_files[:10]:
            try:
                h_df = pd.read_csv(h_file)
                h_date = h_df['日期'].iloc[0]
                # 跟踪所有当时被标注为“买入”或“连跌监控”的标的
                h_targets = h_df[h_df['决策建议'].str.contains('★|连跌')]
                for _, row in h_targets.iterrows():
                    code = str(row['代码']).zfill(6)
                    if code in current_prices:
                        old_p, now_p = row['现价'], current_prices[code]
                        perf_list.append({
                            '决策日期': h_date, '代码': code, '名称': row['名称'],
                            '当时建议': row['决策建议'], '当时价': old_p,
                            '今日价': now_p, '盈亏%': round(((now_p - old_p) / old_p) * 100, 2)
                        })
            except: continue

    # 哪怕为空也生成文件，避免 Actions 报错
    columns = ['决策日期', '代码', '名称', '当时建议', '当时价', '今日价', '盈亏%']
    pd.DataFrame(perf_list if perf_list else [], columns=columns).to_csv('performance_tracking.csv', index=False, encoding='utf-8-sig')

def main():
    name_mapping = get_target_mapping()
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    tasks = [(f, name_mapping) for f in all_files]

    results = []
    # 多进程并行执行分析任务
    with ProcessPoolExecutor() as executor:
        for res in executor.map(analyze_file, tasks):
            if res: results.append(res)

    # 转换结果为表格
    cols = ['代码','名称','决策建议','现价','连跌天数','连跌%','周幅%','月幅%','年幅%','MA5偏离%','量比','成交额(万)','日期']
    res_df = pd.DataFrame(results if results else [], columns=cols)
    
    if not res_df.empty:
        # 优先级排序：先排买入建议，再排连跌天数多的
        res_df = res_df.sort_values(by=['决策建议', '连跌天数', 'MA5偏离%'], ascending=[False, False, True])
    
    # 1. 保存最新的决策表
    res_df.to_csv('investment_decision.csv', index=False, encoding='utf-8-sig')
    
    # 2. 存档到历史目录 (history/年/月/...)
    now = datetime.now()
    h_dir = os.path.join('history', now.strftime('%Y'), now.strftime('%m'))
    os.makedirs(h_dir, exist_ok=True)
    h_file = os.path.join(h_dir, f"decision_{now.strftime('%Y%m%d_%H%M%S')}.csv")
    res_df.to_csv(h_file, index=False, encoding='utf-8-sig')
    
    # 3. 运行回测跟踪模块
    track_performance(results)
    
    # 统计汇总并打印到控制台
    monitored = res_df[res_df['决策建议'].str.contains('连跌')]
    print(f"\n--- 运行报告 ---")
    print(f"累计分析标的数量: {len(results)}")
    print(f"连跌3-5天(监控中)的数量: {len(monitored)}")
    print(f"黄金坑建议数量: {len(res_df[res_df['决策建议'].str.contains('黄金坑')])}")

if __name__ == "__main__":
    main()
