import os
import pandas as pd
import glob
import re
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# --- 核心配置 ---
DATA_DIR = 'fund_data'
ETF_LIST_FILE = 'ETF列表.txt'
MIN_TURNOVER = 5000000       # 500万成交额门槛
MIN_SCORE_SIGNAL = 70        # 超跌评分门槛

def get_target_mapping():
    """读取ETF代码与名称映射"""
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

def calculate_tech(df):
    """计算核心技术指标序列"""
    df = df.sort_values('日期').copy()
    
    # RSI (14)
    delta = df['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # KDJ (9,3,3)
    low_9 = df['最低'].rolling(9).min()
    high_9 = df['最高'].rolling(9).max()
    rsv = (df['收盘'] - low_9) / (high_9 - low_9) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    # 均线系统与乖离率
    df['MA5'] = df['收盘'].rolling(5).mean()
    df['MA20'] = df['收盘'].rolling(20).mean()
    df['BIAS_20'] = (df['收盘'] - df['MA20']) / df['MA20'] * 100
    
    # 量比
    df['V_MA5'] = df['成交量'].shift(1).rolling(5).mean()
    df['VOL_RATIO'] = df['成交量'] / df['V_MA5']
    
    return df.sort_values('日期', ascending=False).reset_index(drop=True)

def analyze_single_file(file_info):
    """单文件分析决策逻辑"""
    file_path, name_mapping = file_info
    try:
        code = re.search(r'(\d{6})', os.path.basename(file_path)).group(1)
        df = pd.read_csv(file_path)
        if len(df) < 40: return None
        
        df = calculate_tech(df)
        last = df.iloc[0]
        prev = df.iloc[1]
        
        if float(last['成交额']) < MIN_TURNOVER: return None

        # --- 初始化变量防止KeyError ---
        signal_type = "无信号"
        advice = "观望"
        final_score = 0
        stop_loss_price = 0

        # --- 逻辑 A：超跌反弹确认 (你的核心策略) ---
        score_oversold = 0
        if last['RSI'] < 38: score_oversold += 35
        if last['J'] < 10: score_oversold += 35
        if last['BIAS_20'] < -3: score_oversold += 30
        
        # --- 逻辑 B：主升浪/趋势识别 ---
        is_strong_trend = last['RSI'] > 65 and last['收盘'] > last['MA5']
        
        # --- 判定优先级 ---
        # 1. 超跌勾头 (最高优)
        if score_oversold >= MIN_SCORE_SIGNAL and last['J'] > prev['J']:
            signal_type = "★★★ 超跌反弹"
            advice = "超跌修复确认。建议40天波段，止损参考今日最低价。"
            final_score = score_oversold
            stop_loss_price = last['最低']

        # 2. 极致超买 (风险提示)
        elif last['RSI'] > 80:
            signal_type = "☢ 极致超买"
            advice = "处于博傻阶段！若要追涨，必须严格执行5日线跌破清仓。"
            final_score = -20
            stop_loss_price = round(last['MA5'], 3)

        # 3. 主升浪趋势
        elif is_strong_trend:
            signal_type = "🚀 趋势主升"
            advice = "动能强劲。追涨建议：5日线不破持股，跌破止损。"
            final_score = 65
            stop_loss_price = round(last['MA5'], 3)
            
        else:
            return None # 非信号区间，不输出

        return {
            '代码': code,
            '名称': name_mapping.get(code, "未知"),
            '信号强度': signal_type,
            '操作建议': advice,
            '综合评分': final_score,
            '建议止损价': stop_loss_price,
            '现价': last['收盘'],
            'RSI': round(last['RSI'], 2),
            'KDJ_J': round(last['J'], 2),
            'MA20偏离%': round(last['BIAS_20'], 2),
            '当前量比': round(last['VOL_RATIO'], 2),
            '日期': last['日期']
        }
    except Exception as e:
        return None

def main():
    name_mapping = get_target_mapping()
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    tasks = [(f, name_mapping) for f in files]

    print(f"🚀 正在扫描 {len(files)} 个标的...")
    results = []
    with ProcessPoolExecutor() as executor:
        for res in executor.map(analyze_single_file, tasks):
            if res: results.append(res)

 
    if not results:
        print("💡 今日全市场未检出显著信号。")
        res_df = pd.DataFrame(columns=['代码', '名称', '信号强度', '操作建议', '综合评分', '建议止损价', '现价', '日期'])
    else:
        res_df = pd.DataFrame(results).sort_values(by='综合评分', ascending=False)

    # 保存结果
    res_df.to_csv('investment_decision.csv', index=False, encoding='utf-8-sig')
    
    # 历史归档
    now = datetime.now()
    h_dir = os.path.join('history', now.strftime('%Y'), now.strftime('%m'))
    os.makedirs(h_dir, exist_ok=True)
    res_df.to_csv(os.path.join(h_dir, f"report_{now.strftime('%Y%m%d')}.csv"), index=False, encoding='utf-8-sig')

    print(f"✅ 复盘完成！检出标的数: {len(res_df)}")
    if not res_df.empty:
        print("\n" + "="*30)
        print(res_df[['代码', '名称', '信号强度', '现价', '建议止损价']].head(10).to_string(index=False))
        print("="*30)

if __name__ == "__main__":
    main()
