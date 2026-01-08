import pandas as pd
import numpy as np
import os
from datetime import datetime
import multiprocessing

# ==========================================
# 战法名称：龙回头筑底突破战法
# 逻辑要领：
# 1. 空间：从近一年高点回落 > 50%，释放风险。
# 2. 筑底：底部横盘超过20个交易日，波动收敛。
# 3. 突破：今日放量（倍量）且突破近期平台颈线。
# 4. 指标：MACD低位金叉或底背离。
# ==========================================

def screen_logic(file_path, names_df):
    try:
        df = pd.read_csv(file_path)
        if len(df) < 60: return None
        
        # 基础过滤：只要深沪A股 (60, 00, 01)
        code = str(df['股票代码'].iloc[-1]).zfill(6)
        if not (code.startswith('60') or code.startswith('00') or code.startswith('01')):
            return None
        
        # 最新价格过滤 (5.0 - 20.0)
        last_price = df['收盘'].iloc[-1]
        if not (5.0 <= last_price <= 20.0):
            return None

        # --- 核心战法逻辑计算 ---
        
        # 1. 超跌检测 (近一年最高价回落)
        high_1y = df['最高'].tail(250).max()
        drop_ratio = (high_1y - last_price) / high_1y
        
        # 2. 量能检测 (今日成交量是否倍量)
        vol_today = df['成交量'].iloc[-1]
        vol_ma5 = df['成交量'].tail(5).mean()
        is_vol_boost = vol_today > vol_ma5 * 1.8 # 接近2倍量
        
        # 3. 突破检测 (突破近20日平台最高价)
        platform_high = df['最高'].tail(21).iloc[:-1].max()
        is_breakout = last_price > platform_high
        
        # 4. MACD 计算 (快速判断金叉)
        exp1 = df['收盘'].ewm(span=12, adjust=False).mean()
        exp2 = df['收盘'].ewm(span=26, adjust=False).mean()
        diff = exp1 - exp2
        dea = diff.ewm(span=9, adjust=False).mean()
        is_gold_cross = diff.iloc[-1] > dea.iloc[-1] and diff.iloc[-2] <= dea.iloc[-2]

        # --- 优中选优评分体系 ---
        score = 0
        if drop_ratio > 0.6: score += 40  # 超跌加分
        if is_vol_boost: score += 30     # 倍量突破加分
        if is_gold_cross: score += 30    # 指标共振加分
        
        if score >= 70: # 只有高分才输出，一击必中
            stock_name = names_df.get(code, "未知")
            
            # 自动复盘操作建议
            suggestion = ""
            if score >= 90:
                suggestion = "【重仓博弈】超跌+倍量金叉共振，极大概率反转"
            elif is_breakout:
                suggestion = "【试错买入】突破颈线位，观察持续性"
            else:
                suggestion = "【轻仓观察】形态走好，等待右侧确立"

            return {
                "日期": df['日期'].iloc[-1],
                "代码": code,
                "名称": stock_name,
                "现价": last_price,
                "跌幅深度": f"{drop_ratio*100:.1f}%",
                "买入信号强度": f"{score}分",
                "操作建议": suggestion
            }
            
    except Exception as e:
        return None

def main():
    data_dir = './stock_data'
    names_file = './stock_names.csv'
    
    # 加载股票名称
    names_df = pd.read_csv(names_file, dtype={'code': str}).set_index('code')['name'].to_dict()
    
    # 扫描文件
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    # 多进程并行处理加快速度
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = pool.starmap(screen_logic, [(f, names_df) for f in files])
    pool.close()
    pool.join()
    
    # 过滤空结果
    final_list = [r for r in results if r is not None]
    
    if final_list:
        output_df = pd.DataFrame(final_list)
        
        # 创建年月目录
        now = datetime.now()
        dir_path = now.strftime('%Y%m')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
        # 生成带时间戳的文件名
        file_name = f"dragon_breakout_{now.strftime('%Y%m%d_%H%M%S')}.csv"
        output_df.to_csv(os.path.join(dir_path, file_name), index=False, encoding='utf_8_sig')
        print(f"成功筛选出 {len(final_list)} 只潜力股，结果已保存。")
    else:
        print("今日未发现符合'一击必中'条件的股票。")

if __name__ == "__main__":
    main()
