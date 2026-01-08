import os
import pandas as pd
import glob
import re
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# --- 交易逻辑配置 ---
DATA_DIR = 'fund_data'
ETF_LIST_FILE = 'ETF列表.txt'
# 提高门槛：成交额 > 500万确保是一线主力品种，流动性差的（容易被操控）直接剔除
MIN_TURNOVER = 5000000       
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

def calculate_technical_indicators(df):
    """计算 RSI 和 KDJ，确保时间序列正确"""
    tdf = df.iloc[::-1].copy() # 转为正序计算指标
    
    # RSI (14日)
    delta = tdf['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    tdf['RSI'] = 100 - (100 / (1 + (gain / loss)))

    # KDJ (9, 3, 3)
    low_9 = tdf['收盘'].rolling(9).min()
    high_9 = tdf['收盘'].rolling(9).max()
    rsv = (tdf['收盘'] - low_9) / (high_9 - low_9) * 100
    tdf['K'] = rsv.ewm(com=2, adjust=False).mean()
    tdf['D'] = tdf['K'].ewm(com=2, adjust=False).mean()
    tdf['J'] = 3 * tdf['K'] - 2 * tdf['D']
    
    res_tdf = tdf.iloc[::-1] # 转回倒序
    df['RSI'], df['KDJ_J'] = res_tdf['RSI'], res_tdf['J']
    return df

def analyze_file(file_info):
    file_path, name_mapping = file_info
    try:
        code = re.search(r'(\d{6})', os.path.basename(file_path)).group(1)
        df = pd.read_csv(file_path)
        if len(df) < 30: return None
        
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期', ascending=False).reset_index(drop=True)
        
        last_price = df.loc[0, '收盘']
        last_turnover = float(df.loc[0, '成交额'])
        if last_turnover < MIN_TURNOVER: return None
            
        df = calculate_technical_indicators(df)
        
        # 1. 连跌计算
        count, total_drop = 0, 0.0
        for i in range(len(df)):
            if float(df.loc[i, '涨跌幅']) < 0:
                count += 1
                total_drop += float(df.loc[i, '涨跌幅'])
            else: break
        
        # 2. 空间指标（原有功能保留）
        def get_chg(d): return round(((last_price - df.loc[d, '收盘']) / df.loc[d, '收盘']) * 100, 2) if len(df) > d else 0
        w_chg, m_chg, y_chg = get_chg(5), get_chg(20), get_chg(250)
        
        # 3. 动能与偏离
        ma5 = df.loc[0:4, '收盘'].mean()
        bias = round(((last_price - ma5) / ma5) * 100, 2)
        rsi, j_val = df.loc[0, 'RSI'], df.loc[0, 'KDJ_J']
        vol_ratio = round(df.loc[0, '成交量'] / df.loc[1:5, '成交量'].mean(), 2)
        turnover = df.loc[0, '换手率'] if '换手率' in df.columns else 0

        # --- 全自动复盘评分系统 (优中选优) ---
        score = 0
        if 3 <= count <= 5: score += 20      # 情绪维度：连跌3-5天是变盘点
        if rsi < 30: score += 20             # 动能维度：RSI进入超卖区
        if j_val < 0: score += 20            # 动能维度：KDJ J线杀出负值
        if y_chg < -10: score += 15          # 空间维度：年线下跌10%以上属于低位
        if bias < -2.5: score += 15          # 空间维度：短线偏离MA5太远
        if 0.4 < vol_ratio < 0.8: score += 10 # 量价维度：明显的缩量地量企稳

        # --- 自动生成操作建议 ---
        if score >= 85:
            signal, advice = "★★★ 一击必中", "多维共振极度超卖。历史高胜率区域，建议重仓狙击/加仓。"
        elif 65 <= score < 85:
            signal, advice = "★★ 分批试错", "情绪接近底部。建议头仓入场，若继续杀跌则在低位摊薄。"
        elif 40 <= score < 65:
            signal, advice = "★ 试探观察", "跌势放缓。暂无共振信号，建议加入自选，等待J线拐头。"
        elif vol_ratio > 2.0 and last_price < ma5:
            signal, advice = "☢ 暂时放弃", "放量大跌说明恐慌盘未出净。切勿接飞刀，建议至少观察3天。"
        else:
            signal, advice = "○ 择机等待", "指标平庸，无明显多空博弈点，建议休息保持现金流。"

        return {
            '代码': code, '名称': name_mapping.get(code, "未知"),
            '信号强度': signal, '操作建议': advice, '综合评分': score,
            '现价': last_price, '连跌天数': count, 'RSI': round(rsi, 2), 'KDJ_J': round(j_val, 2),
            '周幅%': w_chg, '月幅%': m_chg, '年幅%': y_chg, 'MA5偏离%': bias,
            '量比': vol_ratio, '换手率%': round(turnover, 2), '成交额(万)': round(last_turnover/10000, 2),
            '日期': df.loc[0, '日期'].strftime('%Y-%m-%d')
        }
    except: return None

def main():
    name_mapping = get_target_mapping()
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    tasks = [(f, name_mapping) for f in all_files]

    results = []
    with ProcessPoolExecutor() as executor:
        for res in executor.map(analyze_file, tasks):
            if res: results.append(res)

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        # 只输出有信号的标的，或按评分降序排列，优加选优
        res_df = res_df.sort_values(by='综合评分', ascending=False)
    
    # 保存结果
    res_df.to_csv('investment_decision.csv', index=False, encoding='utf-8-sig')
    
    # 历史归档
    now = datetime.now()
    h_dir = os.path.join('history', now.strftime('%Y'), now.strftime('%m'))
    os.makedirs(h_dir, exist_ok=True)
    res_df.to_csv(os.path.join(h_dir, f"report_{now.strftime('%Y%m%d')}.csv"), index=False, encoding='utf-8-sig')

    # 控制台复盘总结
    top_picks = res_df[res_df['综合评分'] >= 65]
    print(f"\n{'='*25} AI 自动复盘报告 {'='*25}")
    if not top_picks.empty:
        print(f"今日共锁定 {len(top_picks)} 个优选目标：")
        print(top_picks[['代码', '名称', '信号强度', '操作建议', '综合评分']].to_string(index=False))
    else:
        print("今日暂无强力信号。空仓也是一种战斗，耐心等待黄金坑。")
    print(f"{'='*67}\n")

if __name__ == "__main__":
    main()
