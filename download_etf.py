import pandas as pd
import akshare as ak
import os
from concurrent.futures import ThreadPoolExecutor

# 创建存储目录
SAVE_DIR = "fund_data"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def download_fund_data(fund_code):
    """下载单个基金的历史行情并保存为CSV"""
    try:
        fund_code = str(fund_code).strip().zfill(6)
        print(f"正在下载: {fund_code}")
        
        # 使用东方财富接口获取历史行情 
        # 默认获取所有历史数据，包含日期、开盘、收盘、最高、最低等 [cite: 3541, 3543, 3544]
        df = ak.fund_etf_hist_em(symbol=fund_code, period="daily", adjust="qfq")
        
        if not df.empty:
            file_path = os.path.join(SAVE_DIR, f"{fund_code}.csv")
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            return f"{fund_code} 下载成功"
        else:
            return f"{fund_code} 数据为空"
    except Exception as e:
        return f"{fund_code} 下载失败: {str(e)}"

def main():
    # 读取 etf.txt 中的基金代码 
    if not os.path.exists("etf.txt"):
        print("未找到 etf.txt 文件")
        return

    # 假设 etf.txt 第一行为 'code'，后续为代码 
    codes_df = pd.read_csv("etf.txt")
    fund_codes = codes_df['code'].unique().tolist()

    # 使用线程池并行下载
    print(f"开始并行下载 {len(fund_codes)} 只基金数据...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(download_fund_data, fund_codes))
    
    for res in results:
        print(res)

if __name__ == "__main__":
    main()