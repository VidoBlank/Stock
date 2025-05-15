import tushare as ts
import pandas as pd
import os
import time
from datetime import datetime, timedelta
import concurrent.futures
from config import Config

ts.set_token(Config.TUSHARE_TOKEN)
pro = ts.pro_api()

# 需要抓取的接口配置（新增关键接口）
API_CONFIG = {
    # 基础行情
    'daily': {'fields': 'ts_code,trade_date,open,high,low,close,vol,amount,pct_chg'},
    'daily_basic': {'fields': 'ts_code,trade_date,total_mv,turnover_rate'},
    
    # 资金相关
    'moneyflow': {'fields': 'ts_code,trade_date,buy_lg_amount,sell_lg_amount'},
    'block_trade': {'fields': 'ts_code,trade_date,price,vol,amount'},
    
    # 财务指标
    'fina_indicator_vip': {'fields': 'ts_code,end_date,roe,grossprofit_margin,debt_to_assets'},
    'express': {'fields': 'ts_code,end_date,net_profit_yoy'},
    'forecast': {'fields': 'ts_code,ann_date,end_date,type'},
    
    # 市场参考数据
    'stk_limit': {'fields': 'ts_code,trade_date,up_limit,down_limit'},
    'limit_list_d': {'fields': 'ts_code,trade_date'},
    'suspend_d': {'fields': 'ts_code,suspend_date'},
    
    # 股东股权
    'stk_holdernumber': {'fields': 'ts_code,end_date,holder_num'},
    'stk_holdertrade': {'fields': 'ts_code,ann_date,holder_name,change_vol,change_type'},
    'pledge_stat': {'fields': 'ts_code,end_date,pledge_ratio'},
    'pledge_detail': {'fields': 'ts_code,ann_date,pledge_amount'},
    
    # 基础信息
    'stock_basic': {'fields': 'ts_code,name,industry,list_date,market'},
    'concept_detail': {'fields': 'id,concept_name,ts_code'},
    'index_weight': {'fields': 'index_code,con_code,weight'},
    'index_member': {'fields': 'index_code,con_code'},
    
    # 指数数据
    'index_daily': {'fields': 'ts_code,trade_date,close'},
    
    # 特色数据
    'share_float': {'fields': 'ts_code,float_date,float_share'},
    'anns_d': {'fields': 'ts_code,ann_date,title,content'},
    
    # 同花顺概念板块数据（新添加的配置）
    'ths_index': {'fields': 'ts_code,name,count,exchange,list_date,type'}
}


# fetch_a_shares_data.py
def fetch_single_api(api_name, date):
    """抓取单个接口数据"""
    file_path = f'history_data/{api_name}/{date}.parquet'
    if os.path.exists(file_path):
        print(f"✅ 数据已存在，跳过抓取：{api_name} {date}")
        return (api_name, date, pd.read_parquet(file_path))

    try:
        api_method = getattr(pro, api_name)
        # 处理需要特殊参数的接口
        params = {}
        if api_name in ['index_weight', 'index_member']:
            params = {'trade_date': date}
        elif api_name == 'share_float':
            params = {'ann_date': date}
            
        df = api_method(**params, **API_CONFIG[api_name])
        
        # === 新增字段校验 ===
        required_fields = API_CONFIG[api_name]['fields'].split(',')
        missing = [f for f in required_fields if f not in df.columns]
        if missing:
            print(f"❌ {api_name} {date} 字段缺失：{missing}，跳过保存")
            return (api_name, date, None)
        # ===================
            
        return (api_name, date, df)
    except Exception as e:
        print(f"抓取{api_name}失败：{str(e)} 日期：{date}")
        return (api_name, date, None)

def save_data(api_name, date, data):
    """保存数据到按接口分日期目录"""
    if data is None or data.empty:
        print(f"警告：{api_name} 数据为空，跳过保存")
        return
    dir_path = f'history_data/{api_name}'
    os.makedirs(dir_path, exist_ok=True)
    file_path = f'{dir_path}/{date}.parquet'

    if os.path.exists(file_path):
        print(f"✅ 文件已存在，跳过保存：{file_path}")
        return

    data.to_parquet(file_path)
    print(f"✅ 已保存{api_name} {date}数据，共{len(data)}条")

def fetch_history_data(start_date, end_date):
    """抓取指定日期范围内所有必要数据"""
    print(f"开始抓取从 {start_date} 到 {end_date} 的数据...")
    dates = pd.date_range(start=start_date, end=end_date).strftime('%Y%m%d').tolist()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for api_name in API_CONFIG.keys():
            for date in dates:
                futures.append(executor.submit(fetch_single_api, api_name, date))
        
        for future in concurrent.futures.as_completed(futures):
            api_name, date, df = future.result()
            if df is not None:
                save_data(api_name, date, df)
            time.sleep(0.5)

if __name__ == '__main__':
    fetch_history_data('20240101', '20240301')