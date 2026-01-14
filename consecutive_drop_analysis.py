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
PORTFOLIO_FILE = 'virtual_portfolio.csv'  # 虚拟持仓账本
MIN_TURNOVER = 5000000       
MIN_SCORE_SIGNAL = 65        

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

def update_portfolio(new_signals):
    """更新虚拟持仓账本：存入新信号，更新旧信号收益"""
    if not os.path.exists(PORTFOLIO_FILE):
        df_p = pd.DataFrame(columns=['代码', '名称', '买入日期', '买入价', '当前价', '持有天数', '当前收益%', '信号类型'])
    else:
        df_p = pd.read_csv(PORTFOLIO_FILE)

    # 1. 存入今日新信号 (仅存入 评分为正且非极致超买的信号)
    for s in new_signals:
        if s['综合评分'] >= MIN_SCORE_SIGNAL:
            # 避免同一天重复记录同一代码
            if not ((df_p['代码'] == int(s['代码'])) & (df_p['买入日期'] == s['日期'])).any():
                new_row = {
                    '代码': s['代码'], '名称': s['名称'], '买入日期': s['日期'],
                    '买入价': s['现价'], '当前价': s['现价'], '持有天数': 0,
                    '当前收益%': 0.0, '信号类型': s['信号强度']
                }
                df_p = pd.concat([df_p, pd.DataFrame([new_row])], ignore_index=True)

    # 2. 更新账本中所有记录的实时状态
    # 注意：这里需要再次读取最新数据来更新旧记录
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    for idx, row in df_p.iterrows():
        code_str = str(int(row['代码'])).zfill(6)
        target_file = [f for f in files if code_str in f]
        if target_file:
            df_temp = pd.read_csv(target_file[0])
            last_price = df_temp.iloc[-1]['收盘']
            last_date = df_temp.iloc[-1]['日期']
            
            # 计算持有天数 (自然日)
            d1 = datetime.strptime(str(row['买入日期']), '%Y-%m-%d')
            d2 = datetime.strptime(str(last_date), '%Y-%m-%d')
            hold_days = (d2 - d1).days
            
            df_p.at[idx, '当前价'] = last_price
            df_p.at[idx, '持有天数'] = hold_days
            df_p.at[idx, '当前收益%'] = round((last_price - row['买入价']) / row['买入价'] * 100, 2)

    df_p.to_csv(PORTFOLIO_FILE, index=False, encoding='utf-8-sig')
    return df_p

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

        signal_type, advice, final_score, stop_loss_price = "无信号", "观望", 0, 0

        # --- 逻辑 A：超跌反弹确认 ---
        score_oversold = 0
        if last['RSI'] < 38: score_oversold += 35
        if last['J'] < 10: score_oversold += 35
        if last['BIAS_20'] < -3: score_oversold += 30
        
        # --- 逻辑 B：主升浪/趋势识别 ---
        is_strong_trend = last['RSI'] > 65 and last['收盘'] > last['MA5']
        
        if score_oversold >= MIN_SCORE_SIGNAL and last['J'] > prev['J']:
            signal_type, advice, final_score = "★★★ 超跌反弹", "底部勾头确认。建议40天波段。", score_oversold
            stop_loss_price = last['最低']
        elif last['RSI'] > 80:
            signal_type, advice, final_score = "☢ 极致超买", "博傻阶段，严守5日线。", -20
            stop_loss_price = round(last['MA5'], 3)
        elif is_strong_trend:
            signal_type, advice, final_score = "🚀 趋势主升", "动能强。5日线不破持股。", 65
            stop_loss_price = round(last['MA5'], 3)
        else:
            return None

        return {
            '代码': code, '名称': name_mapping.get(code, "未知"), '信号强度': signal_type,
            '操作建议': advice, '综合评分': final_score, '建议止损价': stop_loss_price,
            '现价': last['收盘'], 'RSI': round(last['RSI'], 2), 'KDJ_J': round(last['J'], 2),
            'MA20偏离%': round(last['BIAS_20'], 2), '当前量比': round(last['VOL_RATIO'], 2),
            '日期': last['日期']
        }
    except: return None

def main():
    name_mapping = get_target_mapping()
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    tasks = [(f, name_mapping) for f in files]

    print(f"🚀 正在扫描全市场标的并更新账本...")
    results = []
    with ProcessPoolExecutor() as executor:
        for res in executor.map(analyze_single_file, tasks):
            if res: results.append(res)

    # 1. 保存今日决策报告
    if not results:
        res_df = pd.DataFrame(columns=['代码', '名称', '信号强度', '操作建议', '综合评分', '建议止损价', '现价', '日期'])
    else:
        res_df = pd.DataFrame(results).sort_values(by='综合评分', ascending=False)
    
    res_df.to_csv('investment_decision.csv', index=False, encoding='utf-8-sig')
    
    # 2. 更新虚拟持仓账本并统计
    portfolio_df = update_portfolio(results)
    
    print(f"✅ 今日复盘完成。")
    
    # 3. 输出虚拟账本统计信息
    if not portfolio_df.empty:
        print("\n📊 --- 虚拟持仓账本历史表现统计 ---")
        # 统计整体胜率
        total_signals = len(portfolio_df)
        win_signals = len(portfolio_df[portfolio_df['当前收益%'] > 0])
        avg_return = portfolio_df['当前收益%'].mean()
        win_rate = (win_signals / total_signals * 100) if total_signals > 0 else 0
        
        print(f"累计信号总数: {total_signals} | 整体胜率: {win_rate:.2f}% | 平均浮盈: {avg_return:.2f}%")
        
        # 输出近期表现最好的5个历史信号
        print("\n📈 表现最好的历史信号 (Top 5):")
        top_5 = portfolio_df.sort_values(by='当前收益%', ascending=False).head(5)
        print(top_5[['代码', '名称', '买入日期', '持有天数', '当前收益%']].to_string(index=False))

    # 4. 展示今日高分信号
    if not res_df.empty:
        print("\n🔥 今日核心信号提示:")
        print(res_df[['代码', '名称', '信号强度', '现价', '建议止损价']].head(5).to_string(index=False))

if __name__ == "__main__":
    main()
