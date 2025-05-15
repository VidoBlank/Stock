import tushare as ts
import pandas as pd
import json
import time
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, List
from config import Config

# 配置
ts.set_token(Config.TUSHARE_TOKEN)
pro = ts.pro_api()

CONCEPT_STOCK_MAP_FILE = "concept_to_stock_map.json"
CONCEPT_NAME_TO_CODE_FILE = "concept_name_to_code.json"

def build_concept_to_stock_map(trade_date: str = None):
    """构建概念板块到股票的映射关系"""
    if trade_date is None:
        trade_date = datetime.now().strftime('%Y%m%d')
    
    concept_map = defaultdict(list)
    concept_name_to_code = {}
    
    # 获取所有概念板块信息
    concepts = pro.tdx_index(
        trade_date=trade_date, 
        idx_type='概念板块', 
        fields='ts_code,name,idx_count'
    )
    
    print(f"日期 {trade_date} 获取到概念板块数量: {len(concepts)}")
    
    # 统计信息
    success_count = 0
    empty_count = 0
    error_count = 0
    
    for _, row in tqdm(concepts.iterrows(), total=len(concepts), desc="构建概念-股票映射"):
        concept_code = row['ts_code']
        concept_name = row['name']
        expected_count = row['idx_count']
        
        # 保存概念名称到代码的映射
        concept_name_to_code[concept_name] = concept_code
        
        try:
            # 使用正确的接口 tdx_member 获取成分股
            members = pro.tdx_member(
                trade_date=trade_date,
                ts_code=concept_code
            )
            
            if len(members) == 0:
                empty_count += 1
                continue
            
            # 获取成分股代码
            for _, member in members.iterrows():
                stock_code = member['con_code']
                stock_name = member['con_name']
                
                if stock_code and str(stock_code) != 'nan':
                    concept_map[concept_name].append(stock_code)
            
            actual_count = len(concept_map[concept_name])
            if actual_count != expected_count:
                print(f"注意: {concept_name} 预期成分数 {expected_count}, 实际获取 {actual_count}")
            
            success_count += 1
            
            # 添加延时避免频繁请求
            time.sleep(0.1)
                
        except Exception as e:
            error_count += 1
            print(f"❌ 获取概念成分失败：{concept_name} ({concept_code}) - {e}")
    
    # 统计结果
    print(f"\n=== 构建结果 ===")
    print(f"交易日期: {trade_date}")
    print(f"总概念数: {len(concepts)}")
    print(f"成功获取成分股: {success_count}")
    print(f"无成分股数据: {empty_count}")
    print(f"获取失败: {error_count}")
    print(f"有效概念数: {len(concept_map)}")
    print(f"总股票数: {sum(len(stocks) for stocks in concept_map.values())}")
    
    # 保存概念到股票映射
    with open(CONCEPT_STOCK_MAP_FILE, 'w', encoding='utf-8') as f:
        json.dump(concept_map, f, ensure_ascii=False, indent=2)
    
    # 保存概念名称到代码映射
    with open(CONCEPT_NAME_TO_CODE_FILE, 'w', encoding='utf-8') as f:
        json.dump(concept_name_to_code, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 概念映射已保存到: {CONCEPT_STOCK_MAP_FILE}")
    print(f"✅ 概念名称映射已保存到: {CONCEPT_NAME_TO_CODE_FILE}")
    
    return concept_map, concept_name_to_code

if __name__ == "__main__":
    # 执行构建
    build_concept_to_stock_map()