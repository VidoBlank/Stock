

import tushare as ts
from config import Config as AppConfig
import pandas as pd
import json
import os
import requests
import gradio as gr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pandas.tseries.holiday import get_calendar
import logging
from functools import lru_cache
import numpy as np
import concurrent.futures
from typing import List, Tuple, Dict, Optional
from typing import Optional, Tuple, Dict, Any
import time
import threading
from tqdm import tqdm
import concurrent.futures
import traceback
import sys
import appy
from collections import defaultdict
from typing import Set  # 添加到文件顶部导入部分
sys.stdout.reconfigure(encoding='utf-8')
import io
import functools  # 添加此行

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from fetch_a_shares_data import API_CONFIG


# 强制 stdout/stderr 使用 UTF-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 新增配置项
IS_BACKTEST = False  # 默认非回测模式
CURRENT_TRADE_DATE = None  # 回测时设置当前日期


# ===== 全局定义 =====

STRATEGY_GROUPS = {
    # ===== 核心趋势策略 =====
    "趋势型": [
        "均线突破（5/20/30日）",  # 合并原"站上X日均线"系列
        "均线多头排列", 
        "MACD零轴共振",          # 合并"MACD金叉"和"MACD零轴金叉"
        "趋势突破确认"            # 合并"突破压力位"+"突破5日线放量"
    ],
    
    # ===== 量价动能策略 =====
    "动量型": [
        "量价齐升",              # 合并"温和放量"+"突然放量"
        "主力资金共振",          # 合并"龙虎榜异动"+"主力净流入"+"融资净买入"
        "OBV动量引擎"           # 升级原"OBV上穿均线"
    ],
    
    # ===== 底部反转策略 =====
    "反转型": [
        "超跌反弹（RSI+BOLL）",  # 合并"RSI超卖回升"+"BOLL下轨反弹"
        "底部反转确认"           # 新增复合信号
    ],
    
    # ===== 对冲策略 =====
    "市场中性型": [
        "行业超额收益（RS改进版）", 
        "波动率套利",
        "量价背离"
    ],
    
    # ===== 风险控制 =====
    "风险型": [
        "趋势破位（MA60+MACD死叉）",  # 合并原两项
        "高位滞涨风险"              # 新增
    ],
    
    # ===== 新增: 穿线型策略 =====
    "穿线型": [
        "一阳穿三线",             # 新增技术指标
    ]
}

# ===== 策略权重定义 =====
STRATEGY_WEIGHTS = {
    # === 趋势型 ===
    "均线突破（5/20/30日）": 25,  
    "均线多头排列": 28,             
    "MACD零轴共振": 32,             
    "趋势突破确认": 35,            
    
    # === 动量型 ===
    "量价齐升": 35,                 
    "主力资金共振": 38,            
    "OBV动量引擎": 37,             
    
    # === 反转型 ===
    "超跌反弹（RSI+BOLL）": 35,   
    "底部反转确认": 40,
    
    # === 风险型 ===
    "趋势破位（MA60+MACD死叉）": 25,  
    "高位滞涨风险": 20,
    
    # === 穿线型 ===
    "一阳穿三线": 45,              
}


# ===== 板块/市场映射 =====
MARKET_SECTORS = {
    "主板": ["主板"],
    "创业板": ["创业板"],
    "科创板": ["科创板"],
    "中证白酒": [],
    "中证消费": [],
    "国证ETF": [],
    "中证500": [],
    "深证100": [],
    "北证50": [],
    "科创50": [],
    "沪深300": [],
    "上证50": []
}

STRATEGY_TYPE_WEIGHTS = {
    "趋势型": 1.0,  
    "动量型": 1.0,
    "反转型": 1.0,   
    "风险型": -1.0,  
    "市场中性型": 1.0,
    "穿线型": 1.0
}




# 动态扣分系数配置
RISK_PENALTY_MULTIPLIER = {
    "趋势破位（MA60+MACD死叉）": 1.0,  # 重大风险加倍扣分
    "高位滞涨风险": 1.0
}

# 在函数外部定义强信号列表
STRONG_SIGNALS = ["趋势突破确认", "主力资金共振", "底部反转确认"]



POTENTIAL_SIGNALS = [
    "均线多头排列", 
    "BOLL中轨支撑", 
    "MACD金叉", 
    "突破前夕"
]
KEY_BUY_SIGNALS = [
    "回踩30日线确认",
    "突破5日线放量",
    "MACD零轴金叉",
    "均线多头排列"  # 新增
]





# =====各个可用接口及其调用方法 =====
from typing import List
from datetime import datetime, timedelta

# Helper：对单个接口的日期参数做回退尝试，直到拿到非空结果或用尽天数
def _fetch_with_fallback(
    api_func,
    date_field: str,
    base_date: datetime,
    max_back_days: int,
    extra_params: dict
) -> pd.DataFrame:
    for delta in range(max_back_days + 1):
        d = (base_date - timedelta(days=delta)).strftime('%Y%m%d')
        params = extra_params.copy()
        params[date_field] = d
        df = safe_api_call(api_func, **params)
        if not df.empty:
            return df
    return pd.DataFrame()
# 【行业&概念】初始化批量缓存
def initialize_industry_and_concept(ts_codes: List[str]):
    global industry_cache, concept_cache
    logger.info("🚀 初始化行业与概念缓存...")

    # 1. 全量获取行业信息（不需要回退）
    try:
        info_df = safe_api_call(
            pro.stock_basic,
            exchange='',
            list_status='L',
            fields='ts_code,industry'
        )
        if not info_df.empty:
            industry_cache.update(
                dict(zip(
                    info_df['ts_code'],
                    info_df['industry'].fillna('未知行业')
                ))
            )
            logger.info(f"✅ 行业信息缓存完成，共 {len(industry_cache)} 条")
        else:
            logger.warning("⚠️ 行业信息为空")
    except Exception as e:
        logger.warning(f"行业批量获取失败: {e}")

    # 2. 概念批量缓存：分批（200/批）查询，避免单次传参过长导致接口返回空
    batch_size = 200
    for i in range(0, len(ts_codes), batch_size):
        batch = ts_codes[i : i + batch_size]
        ts_str = ",".join(batch)
        try:
            df = safe_api_call(
                pro.concept_detail,
                ts_code=ts_str,
                fields='ts_code,concept_name'
            )
            if not df.empty:
                grouped = df.groupby('ts_code')['concept_name'].apply(list)
                concept_cache.update(grouped.to_dict())
        except Exception as e:
            logger.warning(f"概念批量获取失败（批次 {i // batch_size + 1}）: {e}")

    logger.info(f"✅ 概念信息缓存完成，覆盖股票数：{len(concept_cache)}")
        
# 【市值打分】小市值加分，大市值扣分        
def apply_market_cap_penalty(ts_code: str, score: float) -> float:
    try:
        df = safe_api_call(pro.daily_basic, ts_code=ts_code, trade_date=datetime.today().strftime('%Y%m%d'), fields='ts_code,total_mv')
        if df.empty:
            return score
        market_cap = df.iloc[0]['total_mv'] / 10000  # 换算为亿元

        if market_cap < 30:
            logger.debug(f"{ts_code} 小市值（{market_cap:.1f}亿），加5分")
            return score + 5
        elif market_cap > 1000:
            penalty = min(10, (market_cap - 1000) * 0.01)
            logger.debug(f"{ts_code} 超大市值（{market_cap:.1f}亿），扣{penalty:.1f}分")
            return score - penalty
        return score
    except Exception as e:
        logger.warning(f"{ts_code} 市值扣分失败: {str(e)}")
        return score
  
    


# 【财务评分】根据ROE、毛利率、负债率等打分    
def evaluate_financials(ts_code: str) -> float:
    try:
        # 获取当前年份的最后一天（12月31日）
        current_year = datetime.today().year
        period = f"{current_year - 1}1231"  # 使用去年的12月31日作为报告期

        # 使用 fina_indicator_vip 获取财务数据
        df = safe_api_call(pro.fina_indicator_vip, 
                           ts_code=ts_code, 
                           start_date='20230101',  # 设置起始日期为2023年1月1日
                           end_date=datetime.today().strftime('%Y%m%d'),  # 截止到今天
                           period=period, 
                           fields='roe,roe_dt,grossprofit_margin,netprofit_yoy,debt_to_assets')
        
        # 如果没有数据，记录并返回 0
        if df.empty:
            logger.warning(f"{ts_code} 未获取到财务数据")
            return 0
        
        # 打印调试信息，查看返回的数据内容
        logger.debug(f"{ts_code} 获取到的财务数据: {df.head()}")
        
        data = df.iloc[0]  # 获取第一条数据
        score = 0

        # 确保财务数据为有效数字，否则设为 0
        roe = data['roe'] if isinstance(data['roe'], (int, float)) else 0
        grossprofit_margin = data['grossprofit_margin'] if isinstance(data['grossprofit_margin'], (int, float)) else 0
        netprofit_yoy = data['netprofit_yoy'] if isinstance(data['netprofit_yoy'], (int, float)) else 0
        debt_to_assets = data['debt_to_assets'] if isinstance(data['debt_to_assets'], (int, float)) else 0

        # 打分逻辑
        if roe > 25:
            score += 10  
        elif roe > 20:
            score += 7
        elif roe > 15:
            score += 5
        elif roe > 12:
            score += 3

        if grossprofit_margin > 40:
            score += 7  
        elif grossprofit_margin > 30:
            score += 5
        elif grossprofit_margin > 20:
            score += 3

        if netprofit_yoy > 30:
            score += 7  
        elif netprofit_yoy > 20:
            score += 5
        elif netprofit_yoy > 10:
            score += 3
        elif netprofit_yoy > 0:
            score += 2
        else:
            score -= 10  # 负增长

        if debt_to_assets < 30:
            score += 5  
        elif debt_to_assets < 40:
            score += 3
        elif debt_to_assets > 70:
            score -= 10
        
        # 新增获取现金流量数据
        cash_flow = safe_api_call(pro.cashflow_vip, ts_code=ts_code, 
                                  fields="n_cashflow_act, free_cashflow")  # 替换字段
        
        # 新增盈利能力质量评分
        profit_quality = 0
        if not cash_flow.empty:
            if cash_flow['n_cashflow_act'].iloc[-1] > 1.2:  # 使用 n_cashflow_act 替代 ocf_to_operate_income
                profit_quality += 8
            if cash_flow['free_cashflow'].iloc[-1] > 0:
                profit_quality += 6
                
        # 新增估值指标
        valuation = safe_api_call(pro.daily_basic, ts_code=ts_code,
                                  fields="pe_ttm,pb")
        val_score = 0
        if not valuation.empty:
            if valuation['pe_ttm'].iloc[-1] < 15:
                val_score += 5
            if valuation['pb'].iloc[-1] < 1.5:
                val_score += 5
        
        # 将财务得分与新增的盈利能力质量和估值指标得分合并
        total_score = score + profit_quality + val_score
        logger.debug(f"{ts_code} 财务健康得分: {total_score}")
        
        return total_score

    except Exception as e:
        logger.debug(f"{ts_code} 财务指标获取失败: {str(e)}")
        return 0






    
# 【龙虎榜】机构席位数据缓存    
def initialize_top_inst():
    global top_inst_cache
    logger.info("🚨 初始化龙虎榜机构席位数据...")

    try:
        df = safe_api_call(pro.top_inst, trade_date=datetime.today().strftime('%Y%m%d'))
        if not df.empty:
            top_inst_cache = set(df['ts_code'].unique())
            logger.info(f"✅ 机构席位数据缓存完成：{len(top_inst_cache)} 支股票")
        else:
            top_inst_cache = set()   # 数据为空时，确保是空集合
            logger.warning("⚠️ 机构席位数据为空")
    except Exception as e:
        top_inst_cache = set()       # 异常时也清空，防止残留旧数据
        logger.warning(f"机构席位获取失败: {str(e)}")
        
# 【龙虎榜】命中机构席位加分
def check_top_inst(ts_code: str) -> float:
    return 10 if ts_code in top_inst_cache else 0

# 【资金流向】近5日主力资金净流入得分    
def initialize_moneyflow_scores(ts_codes: List[str]):
    global moneyflow_scores
    logger.info("🚀 初始化资金流向得分缓存...")

    start_date = (datetime.today() - timedelta(days=5)).strftime('%Y%m%d')
    try:
        # 不传 ts_code，获取近5日全市场数据
        df = safe_api_call(pro.moneyflow, start_date=start_date)
        if df.empty:
            logger.warning("⚠️ 资金流向数据为空")
            return

        # 过滤目标股票
        df = df[df['ts_code'].isin(ts_codes)]

        # 分组汇总
        grouped = df.groupby('ts_code')['net_mf_amount'].sum()

        # 计算得分
        raw_scores = {}  # 用于存储原始资金面得分
        for ts_code in ts_codes:
            net_inflow = grouped.get(ts_code, 0)
            if net_inflow > 0:
                bonus = min(5, net_inflow / 1000)
            else:
                bonus = max(-5, net_inflow / 1000)
            raw_scores[ts_code] = bonus  # 存储原始得分

        # 标准化得分：min-max 标准化
        min_score = min(raw_scores.values())
        max_score = max(raw_scores.values())

        # 设置目标得分范围 [0, 10]
        target_min = 0
        target_max = 10

        # 对每只股票的资金面得分进行标准化
        for ts_code in ts_codes:
            raw_score = raw_scores[ts_code]
            # 使用min-max标准化公式将得分映射到[0, 10]的范围
            normalized_score = (raw_score - min_score) / (max_score - min_score) * (target_max - target_min) + target_min
            
            # 将标准化后的得分四舍五入为整数
            moneyflow_scores[ts_code] = round(normalized_score)  # 四舍五入为整数
        
        logger.info(f"✅ 资金流向缓存完成，覆盖 {len(moneyflow_scores)} 支股票")
    except Exception as e:
        logger.warning(f"资金流向批量获取失败: {str(e)}")

        
# 【资金流向】返回单票资金得分        
def evaluate_moneyflow(ts_code: str) -> float:
    return moneyflow_scores.get(ts_code, 0)


# 【涨跌停】批量获取涨跌停价格区间
def initialize_stk_limit(ts_codes: List[str]):
    logger.info("🚨 初始化涨跌停价数据...")
    try:
        df = safe_api_call(pro.stk_limit, trade_date=datetime.today().strftime('%Y%m%d'))
        if not df.empty:
            filtered = df[df['ts_code'].isin(ts_codes)]
            stk_limit_cache.update(filtered.set_index('ts_code')[['up_limit', 'down_limit']].to_dict('index'))
            logger.info(f"✅ 涨跌停价缓存完成：{len(stk_limit_cache)} 支股票")
        else:
            logger.warning("⚠️ 涨跌停数据为空")
    except Exception as e:
        logger.warning(f"涨跌停价获取失败: {str(e)}")

# 【大宗交易】统计近30日活跃度，适当加分        
def initialize_block_trade(ts_codes: List[str]):
    logger.info("🚨 初始化大宗交易数据...")
    try:
        start_date = (datetime.today() - timedelta(days=30)).strftime('%Y%m%d')
        df = safe_api_call(pro.block_trade, start_date=start_date)
        if not df.empty:
            filtered = df[df['ts_code'].isin(ts_codes)]
            grouped = filtered.groupby('ts_code').size()
            block_trade_cache.update(grouped.to_dict())
            logger.info(f"✅ 大宗交易缓存完成：{len(block_trade_cache)} 支股票")
        else:
            logger.warning("⚠️ 大宗交易数据为空")
    except Exception as e:
        logger.warning(f"大宗交易获取失败: {str(e)}")


def save_to_local_cache(data, filename):
    """将数据保存到本地缓存文件"""
    with open(filename, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_from_local_cache(filename):
    """从本地缓存文件加载数据"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def fetch_stock_data(ts_code, trade_date_str, retries=3, timeout=30):
    """获取每个概念板块的数据，并处理超时"""
    for attempt in range(retries):
        try:
            stock_data = StockAnalyzer.pro.daily(ts_code=ts_code, trade_date=trade_date_str, fields='ts_code,close', timeout=timeout)
            if not stock_data.empty:
                return stock_data.to_dict(orient='records')
            else:
                logger.warning(f"获取 {ts_code} 的数据为空")
                return None
        except Exception as e:
            logger.warning(f"获取 {ts_code} 的数据时出错: {e}")
            if attempt < retries - 1:
                logger.info(f"重试 {ts_code} (尝试 {attempt + 1} 次)")
                time.sleep(1)
    return None

def get_concept_trends(trade_date: str):
    """
    获取同花顺概念板块数据并计算每个概念板块的趋势。
    """
    if IS_BACKTEST:  # 如果是回测模式，跳过获取同花顺概念数据
        logger.info("回测模式下，跳过获取同花顺概念数据")
        return pd.DataFrame()  # 返回一个空的DataFrame

    cache_filename = f'concept_data_{trade_date}.json'  # 使用日期作为文件名的一部分来区分不同的缓存

    # 尝试从本地缓存加载数据
    cached_data = load_from_local_cache(cache_filename)
    if cached_data:
        logger.info(f"加载本地缓存数据，共{len(cached_data)}个概念板块")
        return cached_data

    try:
        # 将 trade_date 转换为字符串格式
        trade_date_str = trade_date.strftime('%Y%m%d') if isinstance(trade_date, datetime) else trade_date

        # 获取所有概念板块数据（一次性获取，尽量避免分页）
        df = StockAnalyzer.pro.ths_index(type='N')  # 获取概念指数，type='N'表示概念指数

        if df is not None and not df.empty:
            df = df[['ts_code', 'name', 'count']]  # 只提取股票代码、板块名称和成分股数
            logger.info(f"获取同花顺概念板块数据成功，共{len(df)}个板块")

            # 将所有概念板块的涨幅数据一次性获取
            concept_data = []

            # 使用并发处理获取每个板块的涨幅数据
            def fetch_stock_data_for_row(row):
                """封装 row 获取数据的逻辑"""
                ts_code = row['ts_code']
                result = fetch_stock_data(ts_code, trade_date_str)
                return result

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_ts_code = {executor.submit(fetch_stock_data_for_row, row): row for _, row in df.iterrows()}

                # 使用 tqdm 显示进度条
                for future in tqdm(concurrent.futures.as_completed(future_to_ts_code), total=len(future_to_ts_code), desc="获取概念板块涨幅数据"):
                    result = future.result()
                    if result:
                        concept_data.append(result)

            logger.info(f"成功获取 {len(concept_data)} 个概念板块的涨幅数据")

            # 缓存数据到本地
            save_to_local_cache(concept_data, cache_filename)

            # 转换为 DataFrame
            concept_data_df = pd.DataFrame(concept_data)
            return concept_data_df
        else:
            logger.warning("无法获取同花顺概念板块数据")
    except Exception as e:
        logger.error(f"获取同花顺概念板块数据出错: {str(e)}")
    return pd.DataFrame()

def calculate_concept_trend_score(concept_data: pd.DataFrame, trade_date: str):
    """
    根据概念板块的数据，计算每个概念的趋势评分。
    """
    total_pct_change = 0
    valid_count = 0

    # 确保 concept_data 是 DataFrame 格式
    if isinstance(concept_data, list):
        concept_data = pd.DataFrame(concept_data)

    trade_date_str = trade_date.strftime('%Y%m%d') if isinstance(trade_date, datetime) else trade_date

    for index, row in tqdm(concept_data.iterrows(), total=len(concept_data), desc="计算概念板块趋势得分"):
        concept_ts_code = row['ts_code']
        concept_name = row['name']
        
        # 检查是否有缓存的趋势评分
        cached_score = load_trend_score_from_cache(concept_name)
        if cached_score:
            logger.info(f"加载缓存的趋势评分：{concept_name} - {cached_score['trend_score']}")
            total_pct_change += cached_score['trend_score']
            valid_count += 1
            continue
        
        # 获取板块中每只股票的涨幅
        try:
            stock_data = StockAnalyzer.pro.daily(ts_code=concept_ts_code, trade_date=trade_date_str, fields='ts_code,close')
            if not stock_data.empty:
                pct_change = stock_data['pct_chg'].mean()  # 计算平均涨幅
                total_pct_change += pct_change
                valid_count += 1
                # 缓存计算的趋势评分
                save_trend_score_to_cache(concept_name, pct_change)
        except Exception as e:
            logger.warning(f"获取 {concept_name} 板块的涨幅数据出错: {str(e)}")

    # 返回计算后的趋势评分
    if valid_count > 0:
        avg_pct_change = total_pct_change / valid_count
        logger.info(f"概念板块平均涨幅: {avg_pct_change:.2f}%")
        return avg_pct_change
    return 0.0


def save_trend_score_to_cache(concept_name, trend_score):
    """保存趋势评分到缓存"""
    filename = f'trend_score_{concept_name}.json'
    data = {'concept_name': concept_name, 'trend_score': trend_score}
    save_to_local_cache(data, filename)

def load_trend_score_from_cache(concept_name):
    """加载缓存的趋势评分"""
    filename = f'trend_score_{concept_name}.json'
    return load_from_local_cache(filename)






SECTOR_SCORE_CACHE = "sector_scores_cache.json"
CONCEPT_STOCK_MAP_FILE = "concept_to_stock_map.json"
CONCEPT_NAME_TO_CODE_FILE = "concept_name_to_code.json"

# === 获取概念板块热度评分 ===
def get_sector_strength_scores(trade_date: str) -> Dict[str, float]:
    """根据通达信板块数据打分每个概念板块"""
        # 回测模式下，跳过板块评分
    if IS_BACKTEST:
        logger.info(f"🔙 回测模式：跳过通达信板块评分")
        return {}
    logger.info(f"🔍 开始获取 {trade_date} 的板块数据...")
    
    try:
        # 使用 safe_api_call 确保一致的错误处理
        df = safe_api_call(
            pro.tdx_daily,
            trade_date=trade_date,
            fields="ts_code,3day,5day,bm_buy_ratio,turnover_rate,idx_type"
        )
        
        if df.empty:
            logger.warning(f"⚠️ {trade_date} 无板块数据返回")
            return {}
            
        logger.info(f"📊 原始板块数据数量: {len(df)}")
        
        # 检查返回的字段
        logger.info(f"📊 返回的字段: {df.columns.tolist()}")
        
        # 检查idx_type字段是否存在
        if 'idx_type' not in df.columns:
            logger.error(f"❌ 返回数据缺少 idx_type 字段，尝试其他方法")
            # 尝试获取所有通达信概念板块代码（以880开头的代码）
            df_filtered = df[df['ts_code'].str.startswith('880')]
            logger.info(f"📊 使用代码前缀筛选，找到 {len(df_filtered)} 个板块")
        else:
            logger.info(f"📊 包含的板块类型: {df['idx_type'].unique()}")
            # 筛选概念板块
            df_filtered = df[df['idx_type'] == '概念板块']
            logger.info(f"📊 筛选后概念板块数量: {len(df_filtered)}")
        
        if df_filtered.empty:
            logger.warning("⚠️ 没有找到概念板块数据")
            return {}
            
    except Exception as e:
        logger.error(f"❌ 板块评分获取失败：{e}")
        logger.error(f"错误详情: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"堆栈信息:\n{traceback.format_exc()}")
        return {}
    
    scores = {}
    for _, row in df_filtered.iterrows():
        try:
            # 检查必要字段是否存在
            required_fields = ['3day', '5day', 'bm_buy_ratio', 'turnover_rate']
            missing_fields = [f for f in required_fields if f not in row or pd.isna(row[f])]
            
            if missing_fields:
                logger.warning(f"⚠️ 板块 {row['ts_code']} 缺少字段: {missing_fields}")
                continue
            
            score = (
                row['3day'] * 0.2 +
                row['5day'] * 0.3 +
                row['bm_buy_ratio'] * 0.3 +
                row['turnover_rate'] * 0.2
            )
            scores[row['ts_code']] = round(score, 2)
        except Exception as e:
            logger.warning(f"⚠️ 计算板块 {row['ts_code']} 评分失败: {e}")
            continue
    
    logger.info(f"✅ 成功计算 {len(scores)} 个板块的评分")
    if scores:
        # 打印评分最高的5个板块
        top_5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"🏆 评分TOP5板块: {top_5}")
    
    # 可选：写入缓存
    try:
        with open(SECTOR_SCORE_CACHE, 'w', encoding='utf-8') as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 板块评分已缓存到 {SECTOR_SCORE_CACHE}")
    except Exception as e:
        logger.warning(f"⚠️ 板块评分缓存失败: {e}")
    
    return scores

# === 股票所属概念映射表 ===
def load_concept_to_stock_map() -> Dict[str, List[str]]:
    logger.info(f"📚 开始加载概念股票映射文件: {CONCEPT_STOCK_MAP_FILE}")
    
    if os.path.exists(CONCEPT_STOCK_MAP_FILE):
        try:
            with open(CONCEPT_STOCK_MAP_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"✅ 成功加载 {len(data)} 个概念的股票映射")
            
            # 打印一些统计信息
            total_stocks = sum(len(stocks) for stocks in data.values())
            logger.info(f"📊 总计包含 {total_stocks} 个股票映射关系")
            
            # 打印前3个概念的信息
            for i, (concept, stocks) in enumerate(data.items()):
                if i >= 3:
                    break
                logger.debug(f"  概念样例 {i+1}: {concept} -> {len(stocks)} 只股票")
            
            return data
        except Exception as e:
            logger.error(f"❌ 加载概念股票映射失败: {e}")
            return {}
    else:
        logger.warning(f"⚠️ 概念股票映射文件不存在: {CONCEPT_STOCK_MAP_FILE}")
        return {}

def load_concept_name_to_code() -> Dict[str, str]:
    """加载概念名称到代码的映射"""
    logger.info(f"🔗 开始加载概念代码映射文件: {CONCEPT_NAME_TO_CODE_FILE}")
    
    if os.path.exists(CONCEPT_NAME_TO_CODE_FILE):
        try:
            with open(CONCEPT_NAME_TO_CODE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"✅ 成功加载 {len(data)} 个概念代码映射")
            
            # 打印前5个映射样例
            sample_items = list(data.items())[:5]
            for name, code in sample_items:
                logger.debug(f"  映射样例: {name} -> {code}")
            
            return data
        except Exception as e:
            logger.error(f"❌ 加载概念代码映射失败: {e}")
            return {}
    else:
        logger.warning(f"⚠️ 概念名称映射文件不存在: {CONCEPT_NAME_TO_CODE_FILE}")
        return {}

def get_stock_concepts(stock_code: str, concept_map: Dict[str, List[str]]) -> List[str]:
    """获取某个股票所属概念名称列表"""
    result = []
    for concept, stocks in concept_map.items():
        if stock_code in stocks:
            result.append(concept)
    
    if result:
        logger.debug(f"🏷️ {stock_code} 所属概念: {result}")
    else:
        logger.debug(f"⚠️ {stock_code} 未找到所属概念")
    
    return result

def inject_sector_score(final_score: float, stock_code: str, 
                        concept_name_to_ts_code: Dict[str, str], 
                        sector_scores: Dict[str, float], 
                        concept_map: Dict[str, List[str]],
                        weight: float = 0.3) -> float:
    """对单只股票 final_score 添加板块热度加成"""
    # 回测模式下直接返回原分数
    if IS_BACKTEST:
        logger.debug(f"🔙 回测模式：{stock_code} 跳过板块热度加分")
        return final_score
    
    # 如果没有板块评分数据，也直接返回
    if not sector_scores:
        logger.debug(f"⚠️ 无板块评分数据，{stock_code} 跳过板块热度加分")
        return final_score
    logger.debug(f"💉 开始计算 {stock_code} 的板块热度加分...")
    
    concepts = get_stock_concepts(stock_code, concept_map)
    if not concepts:
        logger.debug(f"⚠️ {stock_code} 无所属概念，板块加分为0")
        return final_score
    
    logger.debug(f"🏷️ {stock_code} 所属概念: {concepts}")
    
    score_boost = 0.0
    detail_boosts = []
    
    for c in concepts:
        ts_code = concept_name_to_ts_code.get(c)
        logger.debug(f"🔍 概念 '{c}' 对应代码: {ts_code}")
        
        if ts_code:
            sector_score = sector_scores.get(ts_code, 0.0)
            logger.debug(f"📊 {c} ({ts_code}) 板块评分: {sector_score}")
            score_boost += sector_score
            detail_boosts.append(f"{c}({sector_score})")
        else:
            logger.debug(f"⚠️ 概念 '{c}' 未找到对应代码")
    
    weighted_boost = score_boost * weight
    logger.debug(f"💰 {stock_code} 板块热度总分: {score_boost:.2f} × {weight} = {weighted_boost:.2f}")
    logger.debug(f"📊 各板块贡献: {', '.join(detail_boosts)}")
    
    return final_score + weighted_boost








# 新增复合信号检测函数
def detect_composite_signals(df: pd.DataFrame) -> dict:
    signals = {}
    
    # 业绩突破+量价齐升
    earnings_growth = (df['revenue_yoy'].iloc[-1] > 0.3) & (df['netprofit_yoy'].iloc[-1] > 0.5)
    volume_spike = df['vol'].iloc[-1] > df['vol'].rolling(20).mean().iloc[-1] * 1.5
    signals["业绩突破+量价齐升"] = earnings_growth & volume_spike
    
    # 机构增持+估值修复
    inst_holding = (df['holder_num'].iloc[-1] < df['holder_num'].iloc[-2]) & \
                  (df['inst_holding_ratio'].iloc[-1] > df['inst_holding_ratio'].iloc[-2])
    valuation_improve = (df['pe_ttm'].iloc[-1] < df['pe_ttm'].rolling(20).mean().iloc[-1]) & \
                       (df['pb'].iloc[-1] < df['pb'].rolling(20).mean().iloc[-1])
    signals["机构增持+估值修复"] = inst_holding & valuation_improve
    
    return signals










def initialize_share_float_data(ts_codes: List[str], days_ahead: int = 30):
    """统计未来 days_ahead 天内将要解禁的股份总量"""
    logger.info(f"🚨 初始化限售解禁数据（未来 {days_ahead} 天）...")
    batch_size = 200
    today = datetime.today()
    end_date = (today + timedelta(days=days_ahead)).strftime('%Y%m%d')
    today_str = today.strftime('%Y%m%d')

    # 预填充默认值
    for code in ts_codes:
        share_float_cache.setdefault(code, 0)

    for i in range(0, len(ts_codes), batch_size):
        batch = ts_codes[i:i + batch_size]
        ts_str = ",".join(batch)

        try:
            # 获取今天公告的所有解禁计划
            df = safe_api_call(pro.share_float, ts_code=ts_str, ann_date=today_str)

            if df.empty or 'ts_code' not in df.columns:
                logger.warning(f"⚠️ share_float 批次 {i // batch_size + 1} 数据为空或字段缺失")
                continue

            # 统计未来 N 天内将解禁的总股数
            valid_codes = df['ts_code'].unique()
            for code in batch:
                if code in valid_codes:
                    sub_df = df[(df['ts_code'] == code) &
                                (df['float_date'] >= today_str) &
                                (df['float_date'] <= end_date)]
                    future_unlock = sub_df['float_share'].sum()
                    share_float_cache[code] = future_unlock
                    logger.info(f"✅ {code} 未来{days_ahead}天将解禁：{future_unlock:.1f} 万股")
                else:
                    share_float_cache[code] = 0

        except Exception as e:
            logger.error(f"❌ share_float 批次 {i // batch_size + 1} 处理失败：{str(e)}")
            for code in batch:
                share_float_cache[code] = 0

    logger.info(f"✅ 限售解禁数据初始化完成 | 有效股票数：{len([v for v in share_float_cache.values() if v > 0])}")


def evaluate_share_float(ts_code: str) -> float:
    """未来解禁评分"""
    try:
        unlock_amount = share_float_cache.get(ts_code, 0)

        if not isinstance(unlock_amount, (int, float)):
            logger.warning(f"{ts_code} 解禁数据异常：{unlock_amount}，按0处理")
            return 0

        # 未来解禁风险评分
        if unlock_amount > 10000:
            logger.debug(f"{ts_code} 未来解禁 {unlock_amount:.1f} 万股，扣6分")
            return -6
        elif unlock_amount > 5000:
            return -4
        elif unlock_amount > 2000:
            return -2
        return 0

    except Exception as e:
        logger.error(f"{ts_code} 解禁评分失败：{str(e)}")
        return 0


def initialize_holdernumber_data(ts_codes: List[str]):
    logger.info("🚨 初始化股东人数数据...")
    batch_size = 200
    today = datetime.today()
    one_year_ago = (today - timedelta(days=365)).strftime('%Y%m%d')

    uncached_codes = [code for code in ts_codes if code not in holdernumber_cache]

    for i in range(0, len(uncached_codes), batch_size):
        batch = uncached_codes[i:i + batch_size]
        ts_str = ",".join(batch)

        try:
            # 获取股东人数数据
            df = safe_api_call(pro.stk_holdernumber, ts_code=ts_str, start_date=one_year_ago, end_date=today.strftime('%Y%m%d'))

            if df.empty or 'ts_code' not in df.columns:
                logger.warning(f"⚠️ 批次 {i // batch_size + 1} 股东人数数据为空或字段缺失")
                continue

            # 计算股东人数变化
            for code in batch:
                sub_df = df[df['ts_code'] == code].sort_values('end_date', ascending=False)
                if len(sub_df) < 2:
                    holdernumber_cache[code] = 0
                else:
                    latest, prev = sub_df.iloc[0]['holder_num'], sub_df.iloc[1]['holder_num']
                    holdernumber_cache[code] = prev - latest

        except Exception as e:
            logger.error(f"股东人数数据处理失败（批次 {i // batch_size + 1}）: {str(e)}")
            for code in batch:
                holdernumber_cache[code] = 0

    logger.info(f"✅ 股东人数数据缓存完成：有效股票数 {len(holdernumber_cache)}")


def evaluate_holdernumber(ts_code: str) -> float:
    diff = holdernumber_cache.get(ts_code, 0)
    if diff > 500:
        logger.debug(f"{ts_code} 股东人数大幅减少 {diff}，加5分")
        return 5
    elif diff > 100:
        return 3
    return 0


def initialize_express_data(period: str, ts_codes: List[str]):
    logger.info("🚨 初始化业绩快报数据...")

    try:
        df = safe_api_call(pro.express_vip, period=period, fields='ts_code,ann_date,end_date,revenue,operate_profit,total_profit,n_income,total_assets')

        if df is None or df.empty:
            logger.warning(f"⚠️ {period} 的业绩快报数据为空")
            return

        for index, row in df.iterrows():
            ts_code = row['ts_code']
            profit_yoy = row.get('net_profit_yoy', 0) or row.get('yoy_net_profit', 0) or row.get('yoy_sales', 0)
            express_cache[ts_code] = profit_yoy
            logger.info(f"{ts_code} 业绩快报净利同比：{profit_yoy:.1f}%")

        logger.info(f"✅ 业绩快报数据缓存完成：{len(express_cache)} 支股票")

    except Exception as e:
        logger.error(f"获取业绩快报数据失败 for {period}: {e}")


def evaluate_express(ts_code: str) -> float:
    profit_yoy = express_cache.get(ts_code, 0)
    if profit_yoy > 50:
        logger.debug(f"{ts_code} 业绩快报净利同比 +{profit_yoy:.1f}%，加6分")
        return 6
    elif profit_yoy > 20:
        return 4
    elif profit_yoy > 0:
        return 2
    elif profit_yoy < -30:
        return -4
    return 0



def initialize_risk_data(ts_codes: List[str]):
    today = datetime.today()
    batch_size = 200
    three_years_ago = (today - timedelta(days=3*365)).strftime('%Y%m%d')
    one_year_ago = (today - timedelta(days=365)).strftime('%Y%m%d')

    # 统一为短代码格式（如600000）
    ts_codes = [c.split('.')[0] for c in list(set(ts_codes))]
    uncached_codes = [
        c for c in ts_codes
        if c not in pledge_stat_cache 
        or c not in pledge_detail_cache 
        or c not in holder_trade_cache
    ]

    if not uncached_codes:
        logger.info("✅ 风险数据已全缓存，跳过初始化")
        return

    for i in range(0, len(uncached_codes), batch_size):
        batch = uncached_codes[i:i + batch_size]
        # 补全交易所代码（600000 -> 600000.SH）
        ts_str = ",".join([
            f"{c}.SH" if c.startswith(('6', '9')) else f"{c}.SZ" 
            for c in batch
        ])

        # ================== 质押率统计 ==================
        try:
            # 从资产负债表获取最新质押率
            df_stat = safe_api_call(
                pro.balancesheet,
                ts_code=ts_str,
                period='20231231',  # 使用最新年报
                fields='ts_code,pledge_ratio'
            )
            if not df_stat.empty:
                for _, row in df_stat.iterrows():
                    code = row['ts_code'].split('.')[0]
                    pledge_stat_cache[code] = float(row['pledge_ratio'])
        except Exception as e:
            logger.error(f"质押率获取失败：{str(e)}")

        # ================== 质押次数统计 ==================
        try:
            df_pledge = pd.DataFrame()
            three_years_ago = (today - timedelta(days=3*365)).strftime('%Y%m%d')
            for offset in range(0, 5000, 1000):
                chunk = safe_api_call(
                    pro.pledge_detail,
                    ts_code=ts_str,
                    limit=1000,
                    offset=offset
                )
                df_pledge = pd.concat([df_pledge, chunk])
            
            if not df_pledge.empty:
                df_pledge['base_code'] = df_pledge['ts_code'].str.split('.').str[0]
                df_pledge = df_pledge[df_pledge['ann_date'] >= three_years_ago]
                df_pledge = df_pledge.drop_duplicates(['base_code', 'ann_date'])
                pledge_counts = df_pledge.groupby('base_code').size()
                
                for code in batch:
                    pledge_detail_cache[code] = pledge_counts.get(code, 0)
        except Exception as e:
            logger.error(f"质押次数获取失败：{str(e)}")

        # ================== 减持次数统计 ==================
        try:
            df_holders = safe_api_call(
                pro.top10_holders,
                ts_code=ts_str,
                start_date=one_year_ago,
                end_date=today.strftime('%Y%m%d'),
                fields='ts_code,hold_change'
            )
            if not df_holders.empty:
                df_holders['hold_change'] = pd.to_numeric(df_holders['hold_change'], errors='coerce')
                reduce_df = df_holders[df_holders['hold_change'] < 0]
                reduce_counts = reduce_df.groupby('ts_code').size()
                
                for code in batch:
                    short_code = code.split('.')[0] if '.' in code else code
                    holder_trade_cache[code] = reduce_counts.get(f"{short_code}.SH", reduce_counts.get(f"{short_code}.SZ", 0))
        except Exception as e:
            logger.error(f"减持次数获取失败：{str(e)}")

    logger.info(f"✅ 风险数据初始化完成 | 质押率:{len(pledge_stat_cache)} 质押次数:{sum(pledge_detail_cache.values())} 减持:{sum(holder_trade_cache.values())}")

def evaluate_risk_factors(ts_code: str) -> float:
    penalty = 0

    pledg_ratio = pledge_stat_cache.get(ts_code, 0)
    if pledg_ratio >= 60:
        penalty -= 12
    elif pledg_ratio >= 40:
        penalty -= 8
    elif pledg_ratio >= 30:
        penalty -= 5

    pledge_times = pledge_detail_cache.get(ts_code, 0)
    if pledge_times >= 5:
        penalty -= 4
    elif pledge_times >= 2:
        penalty -= 2

    reduce_times = holder_trade_cache.get(ts_code, 0)
    if reduce_times >= 3:
        penalty -= 6
    elif reduce_times >= 1:
        penalty -= 3

    if penalty != 0:
        logger.debug(f"{ts_code} 风险扣分：{penalty} (质押率: {pledg_ratio}%, 质押次数: {pledge_times}, 减持次数: {reduce_times})")
    return penalty


    
# ===== 批量缓存数据 =====
industry_cache = {}          # 行业信息缓存：ts_code -> 行业名称
concept_cache = {}           # 概念题材缓存：ts_code -> 概念列表
moneyflow_scores = {}        # 资金流向得分缓存：ts_code -> 得分
concept_list_cache = {}      # 概念名称与ID映射缓存：concept_name -> concept_id
concept_detail_cache = {}    # 概念涨幅缓存：concept_id -> 平均涨幅
pledge_stat_cache = {}       # 质押率缓存：ts_code -> 质押率(%)
pledge_detail_cache = {}     # 质押次数缓存：ts_code -> 次数
holder_trade_cache = {}      # 股东减持次数缓存：ts_code -> 次数
block_trade_cache = {}       # 大宗交易次数缓存：ts_code -> 次数
stk_limit_cache = {}         # 涨跌停价缓存：ts_code -> {'up_limit': x, 'down_limit': y}
share_float_cache = {}      # 限售解禁缓存
holdernumber_cache = {}     # 股东人数缓存
express_cache = {}          # 业绩快报缓存

# ===== 工具函数 =====
def get_strategy_type(strategy_name: str) -> str:
    for group_name, strategy_list in STRATEGY_GROUPS.items():
        if strategy_name in strategy_list:
            return group_name
    return "未知"

# ===== 获取行业与题材信息 =====
def get_industry(ts_code: str) -> str:
    return industry_cache.get(ts_code, "未知行业")

def get_concepts(ts_code: str) -> List[str]:
    return concept_cache.get(ts_code, [])

# ===== 分散度算法 =====
def diversify_recommendations(scored_stocks: List[Tuple], max_recommend=10, min_score_threshold=0) -> List[Tuple]:
    # ⭐ 得分从高到低排序
    scored_stocks = sorted(scored_stocks, key=lambda x: x[0], reverse=True)

    industry_count = {}
    concept_count = {}
    final_selection = []
    dynamic_industry_limit = 3
    dynamic_concept_limit = 2

    for stock in scored_stocks:
        score, ts_code, name, matched, pct_change, df, score_details = stock

        if score < min_score_threshold:
            continue

        # 前三名不受行业和题材限制，必须放在最前面
        if len(final_selection) < 3:
            final_selection.append(stock)
            industry = get_industry(ts_code)
            concepts = get_concepts(ts_code)
            industry_count[industry] = industry_count.get(industry, 0) + 1
            for c in concepts:
                concept_count[c] = concept_count.get(c, 0) + 1
            continue

        industry = get_industry(ts_code)
        concepts = get_concepts(ts_code)

        if len(final_selection) >= max_recommend:
            break

        remaining_slots = max_recommend - len(final_selection)
        if remaining_slots <= 3:
            dynamic_industry_limit = 4
            dynamic_concept_limit = 3

        # 行业限制
        if industry and industry_count.get(industry, 0) >= dynamic_industry_limit:
            continue

        # 题材限制
        if concepts and any(concept_count.get(c, 0) >= dynamic_concept_limit for c in concepts):
            continue

        final_selection.append(stock)
        industry_count[industry] = industry_count.get(industry, 0) + 1
        for c in concepts:
            concept_count[c] = concept_count.get(c, 0) + 1

    # ✅ 最后保险再排一次得分
    final_selection = sorted(final_selection, key=lambda x: x[0], reverse=True)

    return final_selection





# ===== 根据市场行情动态调整策略权重 =====

def adjust_strategy_weights_by_market(trade_date: str = None) -> Dict[str, float]:
    """根据市场行情动态调整策略权重（支持回测模式）"""
    if IS_BACKTEST and CURRENT_TRADE_DATE:
        trade_date = CURRENT_TRADE_DATE  # 强制使用回测日期
    try:
        # ===== 日期处理增强版 =====
        current_trade_date_str = get_valid_trade_date(
            api_func=pro.daily,  # 明确指定接口函数
            date_field='trade_date',  # 明确日期字段名称
            base_date=datetime.strptime(trade_date, "%Y%m%d") if trade_date else datetime.today(),
            max_back_days=5
        )
        
        if not current_trade_date_str:
            logger.error("❌ 无法获取有效交易日")
            return STRATEGY_TYPE_WEIGHTS.copy()
            
        # 将字符串转换为 datetime 对象
        current_trade_date = datetime.strptime(current_trade_date_str, "%Y%m%d")
        logger.info(f"📆 最终处理日期：{current_trade_date.strftime('%Y%m%d')}")

        # ===== 市场指标计算 =====
        market_data = _get_market_indicators(current_trade_date)
        if market_data is None:
            return STRATEGY_TYPE_WEIGHTS.copy()

        pct_change, volatility, market_momentum, market_status, ma20 = market_data
        
        # ===== 市场结果日志 =====
        logger.info(f"当前市场状态：{market_status} (涨跌幅: {pct_change:.2%}, 波动率: {volatility:.2%}, 动量: {market_momentum:.2%}, 20日均线: {ma20:.2f})")
        
        # ===== 动态权重调整核心逻辑 =====
        adjusted = STRATEGY_TYPE_WEIGHTS.copy()
        
        # 根据市场状态调整
        adjustment_rules = {
            "极端熊市": {
                "趋势型": 0.9,
                "反转型": 1.2,
                "市场中性型": 1.2,
                "风险型": 1.3
            },
            "熊市": {
                "趋势型": 0.95,
                "动量型": 0.9,
                "反转型": 1.1,
                "市场中性型": 1.1,
                "风险型": 1.1
            },
            "温和熊市": {
                "趋势型": 1.0,
                "反转型": 1.05,
                "市场中性型": 1.0,
                "风险型": 1.0
            },
            "震荡市": {
                "趋势型": 1.05,  
                "动量型": 1.05,  
                "市场中性型": 1.0,  
                "反转型": 1.05,  
                "风险型": 1.0
            },
            "温和牛市": {
                "趋势型": 1.1,
                "动量型": 1.15
            },
            "牛市": {
                "趋势型": 1.15,
                "动量型": 1.2
            },
            "极端牛市": {
                "趋势型": 1.2,
                "动量型": 1.25
            }
        }
        
        # 应用基础调整规则
        if market_status in adjustment_rules:
            for key, factor in adjustment_rules[market_status].items():
                adjusted[key] *= factor
                
        # 波动率动态调整（线性插值）
        volatility_factor = np.interp(volatility, [0.10, 0.40], [0.7, 1.3])
        adjusted.update({
            "市场中性型": min(adjusted.get("市场中性型", 1.0) * volatility_factor, 2.0),
            "风险型": np.clip(adjusted.get("风险型", -1.0) * (1.5 - 0.7 * volatility_factor), -2.0, 0.0)
        })
        
        # 动量补偿调整
        momentum_bonus = market_momentum * max(0.5 - 0.2 * abs(market_momentum), 0.3)
        adjusted["动量型"] = np.clip(adjusted["动量型"] + momentum_bonus, 0.6, 2.5)
        adjusted["趋势型"] = np.clip(adjusted["趋势型"] + momentum_bonus * 0.7, 0.5, 2.0)
        
        risk_weight = adjusted["风险型"]
        if market_status in ["牛市", "极端牛市"]:
            risk_weight = max(risk_weight, -1.5)  # 牛市中惩罚上限-1.5
        elif market_status == "极端熊市":
            risk_weight = max(risk_weight, -3.0)  # 极端熊市允许-3.0
        adjusted["风险型"] = risk_weight
        
        # 风险权重边界控制
        weight_limits = {
            "趋势型": (0.5, 1.5),
            "动量型": (0.6, 1.6),
            "反转型": (0.7, 1.8),
            "市场中性型": (0.8, 2.0),
            "风险型": (-3.0, 0.0)
        }
        
        # 应用边界限制
        for strategy_type in adjusted:
            if strategy_type in weight_limits:
                min_val, max_val = weight_limits[strategy_type]
                adjusted[strategy_type] = np.clip(adjusted[strategy_type], min_val, max_val)
        
        
        return adjusted

    except Exception as e:
        logger.error(f"❌ 权重调整过程异常：{str(e)}\n{traceback.format_exc()}")
        return STRATEGY_TYPE_WEIGHTS.copy()




def _get_market_indicators(trade_date: datetime) -> Optional[Tuple[float, float, float, str, float]]:
    """获取三大市场指标：涨跌幅、波动率、动量，并计算20日均线（MA20）及判断市场状态"""
    try:
        # 获取沪深300数据
        hs300 = safe_api_call(
            pro.index_daily,
            ts_code="000300.SH",
            start_date=(trade_date - timedelta(days=60)).strftime('%Y%m%d'),
            end_date=trade_date.strftime('%Y%m%d'),
            fields="trade_date,close"
        )
        
        if len(hs300) < 20:
            logger.warning("⚠️ 数据不足20个交易日")
            return None
        
        # 计算市场指标
        closes = hs300['close'].astype(float)
        returns = closes.pct_change().dropna()
        
        # 计算涨跌幅
        pct_change = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]
        
        # 计算波动率（年化）
        volatility = returns.std() * np.sqrt(252)
        
        # 计算动量
        momentum = closes[-20:].mean() / closes[:-20].mean() - 1
        
        # 计算20日均线（MA20）
        ma20 = closes.rolling(20).mean().iloc[-1]
        
        # 获取市场状态
        market_status = _determine_market_status(pct_change, volatility, ma20, closes.iloc[-1], momentum)
        
        return pct_change, volatility, momentum, market_status, ma20
        
    except Exception as e:
        logger.error(f"❌ 获取市场指标失败：{str(e)}")
        return None


def _determine_market_status(pct_change: float, volatility: float, ma20: float, last_close: float, momentum: float) -> str:
    """综合判断市场状态"""
    # 首先根据波动率判断是否为高波动震荡市
    if volatility > 0.25:
        return "高波动震荡市"
    
    # 根据涨跌幅和20日均线共同判断市场状态
    if pct_change < -0.15:
        return "极端熊市"
    elif pct_change < -0.05:
        return "熊市"
    elif pct_change > 0.15:
        # 动量正且较高，极端牛市
        if momentum > 0.1 and last_close > ma20 * 1.08:
            return "极端牛市"
        # 动量正但较低，牛市
        elif momentum > 0.05:
            return "牛市"
        # 涨幅大于15%，但收盘价不大于MA20的1.08倍，牛市
        else:
            return "牛市"
    elif pct_change > 0.05:
        # 动量正且较高，牛市
        if momentum > 0.05 and last_close > ma20 * 1.08:
            return "牛市"
        # 动量负且较低，熊市
        elif momentum < -0.05 or last_close < ma20 * 0.92:
            return "熊市"
    
    # 默认震荡市
    return "震荡市"





def get_valid_trade_date(
        api_func, 
        date_field: str, 
        base_date: Optional[datetime] = None, 
        max_back_days: int = 5, 
        **api_kwargs
    ) -> Optional[str]:
    """
    修复点1：确保返回统一格式的日期字符串
    新增有效性检查逻辑
    """
    if appy.IS_BACKTEST:
        return base_date.strftime('%Y%m%d') 
    base_date = base_date or datetime.today()
    for delta in range(max_back_days + 1):
        current_date = base_date - timedelta(days=delta)
        d = current_date.strftime('%Y%m%d')
        df = safe_api_call(api_func, **api_kwargs, **{date_field: d})
        
        # 新增数据有效性检查
        if not df.empty and 'ts_code' in df.columns and len(df['ts_code'].unique()) > 10:
            logger.info(f"✅ 验证有效交易日: {d} | 包含股票数: {len(df)}")
            return d
    logger.error(f"❌ 未找到有效交易日（最近 {max_back_days} 天）")
    return None


# ===== 全局定义：Query类接口名单 =====
QUERY_INTERFACE_NAMES = [
    # 基础行情类
    'daily', 'daily_basic', 'moneyflow', 
    # 财务指标类
    'fina_indicator_vip', 'express', 'forecast',
    # 市场参考类
    'stk_limit', 'limit_list_d', 'suspend_d', 'block_trade',
    # 股东股权类
    'stk_holdernumber', 'stk_holdertrade', 'pledge_stat', 'pledge_detail',
    # 基础信息类 
    'stock_basic', 'concept_detail', 'index_weight', 'index_member',
    # 指数数据类
    'index_daily',
    # 特色数据类
    'share_float', 'anns_d',
    # 同花顺概念板块数据类（新添加的接口）
    'ths_index'  # 同花顺概念板块接口
]

def safe_api_call(func, *args, retries=3, delay=2, **kwargs):
    """ 封装接口调用，自动选择限速器，处理异常，保证返回 DataFrame """
    
    # 如果 func 是 functools.partial 对象，获取其原始函数
    if isinstance(func, functools.partial):
        api_name = func.args[0] if func.args else kwargs.get('api_name')  # 获取 api_name
    else:
        api_name = func.__name__  # 获取函数名称作为 API 名
    
    # 确保 api_name 与 API_CONFIG 中的键一致
    if api_name not in API_CONFIG:
        api_name = kwargs.get('api_name', api_name)  # 如果 API_CONFIG 中没有该接口，使用 kwargs 中传递的 api_name
    
    # 详细日志记录
    logger.debug(f"准备调用接口: {api_name} 参数: {kwargs}")

    # 限速器
    if api_name in QUERY_INTERFACE_NAMES or 'query' in api_name.lower():
        query_rate_limiter.wait(interface_type="QUERY接口")
        logger.debug(f"✅ 已应用QUERY接口限速: {api_name}")
    else:
        normal_rate_limiter.wait(interface_type="普通接口")
        logger.debug(f"✅ 已应用普通接口限速: {api_name}")

    # 获取API配置信息
    api_config = API_CONFIG.get(api_name, {})  # 获取接口配置，默认空字典
    fields = api_config.get('fields', [])       # 获取字段列表，默认空列表
    
    # 检查是否需要为接口添加 trade_date 参数
    if 'trade_date' in fields:
        kwargs['trade_date'] = appy.CURRENT_TRADE_DATE
        logger.debug(f"✅ 为 {api_name} 添加 trade_date 参数: {appy.CURRENT_TRADE_DATE}")
    else:
        logger.debug(f"⏩ {api_name} 未配置 trade_date 字段")
    
    # 尝试调用接口，最多重试 retries 次
    for attempt in range(1, retries + 1):
        try:
            result = func(*args, **kwargs)
            if isinstance(result, pd.DataFrame):
                logger.debug(f"✅ 接口调用成功: {api_name} 返回 {len(result)} 行数据")
                return result
            else:
                # 返回类型不是 DataFrame 时处理
                logger.warning(f"⚠️ {api_name} 返回非 DataFrame 类型: {type(result)}")
                # 如果返回不是 DataFrame，可以返回空 DataFrame 或做其他处理
                return pd.DataFrame()  
        except Exception as e:
            error_msg = str(e)
            if "请指定正确的接口名" in error_msg or "parameter" in error_msg.lower():
                logger.error(f"❌ {api_name} 参数错误或接口名错误: {error_msg}")
                break
            logger.warning(f"⚠️ 调用 {api_name} 失败（第{attempt}次）: {error_msg}")
            time.sleep(delay * attempt)  # 指数退避

    logger.error(f"🚫 调用 {api_name} 完全失败，返回空 DataFrame")
    return pd.DataFrame()  # 如果调用失败，返回空 DataFrame









# ===== 配置日志系统 =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ===== 系统配置 =====
class Config:
    MAX_STOCKS_TO_ANALYZE = 5500         # 保持不变，最大分析股票数量
    MIN_DATA_DAYS = 30

    
    POINTS = 10000
    MAX_CALLS_PER_MIN = 1000             # 每分钟1000次
    MAX_WORKERS = 24                 


# ===== 动态限速控制器 =====
class RateLimiter:
    def __init__(self, max_calls: int, period: int):
        self.max_calls = max_calls
        self.period = period
        self.lock = threading.Lock()
        self.calls = []

    def wait(self, interface_type=None) -> None:
        # 智能判断接口类型
        if interface_type is None:
            if self.max_calls == 1000:
                interface_type = "QUERY接口"
            elif self.max_calls == 1000:
                interface_type = "普通接口"
            else:
                interface_type = "接口"

        with self.lock:
            now = time.time()
            self.calls = [call for call in self.calls if now - call < self.period]

            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                sleep_time = max(sleep_time, 0)
                if sleep_time > 0:
                    logger.info(f"⏳ [{interface_type}] 限速中：等待 {sleep_time:.1f}s  (限额 {self.max_calls}/{self.period}s)")
                    time.sleep(sleep_time)

            self.calls.append(time.time())


# 初始化两个限速器
normal_rate_limiter = RateLimiter(1000, 60)
query_rate_limiter = RateLimiter(1000, 60)





# ===== API 配置 =====


# 验证配置
AppConfig.validate()
AppConfig.create_dirs()

# 初始化API
ts.set_token(AppConfig.TUSHARE_TOKEN)
pro = ts.pro_api()
DEEPSEEK_API_KEY = AppConfig.DEEPSEEK_API_KEY
# ===== DeepSeek API 交互类 =====
class DeepSeekAPI:
    @staticmethod
    def call_deepseek(prompt: str) -> str:
        """调用DeepSeek API获取策略分析"""
        try:
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个专业的股票分析AI助手。请从用户输入中识别股票技术分析策略，"
                                  "只返回JSON格式的策略列表和解释。"
                    },
                    {
                        "role": "user",
                        "content": f"从以下文本中识别股票技术分析策略: {prompt}"
                    }
                ],
                "temperature": 0.3
            }
            
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"调用DeepSeek API失败: {str(e)}")
            return ""

    @staticmethod
    def parse_strategies(response: str) -> Tuple[List[str], str, Dict[str, int]]:
        """解析DeepSeek API返回的策略"""
        try:
            # 简单实现 - 实际应根据API返回格式调整
            if not response:
                return [], "无法解析策略", {}
                
            # 这里假设API返回的是策略列表
            # 实际实现需要根据API实际返回格式调整
            strategies = []
            explanation = "识别到的策略:\n"
            custom_weights = {}
            
            for strategy in STRATEGY_WEIGHTS.keys():
                if strategy in response:
                    strategies.append(strategy)
                    explanation += f"- {strategy}\n"
                    custom_weights[strategy] = STRATEGY_WEIGHTS[strategy]
            
            return strategies, explanation, custom_weights
        except Exception as e:
            logger.error(f"解析策略失败: {str(e)}")
            return [], "解析策略时出错", {}
# ===== 股票分析核心类 =====
class StockAnalyzer:
    pro = ts.pro_api(AppConfig.TUSHARE_TOKEN) 
    @staticmethod
    def get_valid_daily_data(
        api_func: Any,
        fields: str,
        max_days_back: int = 10,
        extra_kwargs: Optional[Dict[str, Any]] = None,
        base_date: Optional[datetime] = None   # ⭐ 新增参数
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        extra_kwargs = extra_kwargs or {}
        func_name = getattr(api_func, '__name__', None)
        if func_name is None and hasattr(api_func, 'func'):
            func_name = getattr(api_func.func, '__name__', str(api_func))
        if func_name is None:
            func_name = str(api_func)

        valid_date = get_valid_trade_date(
            api_func,
            date_field='trade_date',
            base_date=base_date or datetime.today(),
            max_back_days=max_days_back,
            fields=fields,
            **extra_kwargs
        )
        if valid_date is None:
            logger.error(f"❌ 连续 {max_days_back} 天无有效数据 ({func_name})")
            return pd.DataFrame(), None

        df = safe_api_call(
            api_func,
            trade_date=valid_date,
            fields=fields,
            **extra_kwargs
        )
        if df.empty:
            logger.error(f"❌ 找到交易日 {valid_date}，但拉取数据仍为空 ({func_name})")
            return pd.DataFrame(), None

        logger.info(f"✅ 成功获取 {valid_date} 的数据 ({func_name})")
        return df, valid_date

    @staticmethod
    def get_single_stock_info(ts_code: str) -> Optional[Dict]:
        try:
            if not ts_code.endswith(('.SH', '.SZ')):
                ts_code = f"{ts_code}.SZ" if ts_code.startswith(('00', '30')) else f"{ts_code}.SH"
            basic = safe_api_call(pro.stock_basic, ts_code=ts_code, fields='ts_code,name,industry,list_date,market')
            daily = safe_api_call(pro.daily, ts_code=ts_code,
                                  start_date=(datetime.now() - timedelta(days=30)).strftime('%Y%m%d'),
                                  end_date=datetime.now().strftime('%Y%m%d'))
            if basic.empty or daily.empty:
                return None
            daily = daily.sort_values('trade_date')
            daily.index = pd.to_datetime(daily['trade_date'])

            indicators, _ = StockAnalyzer.calculate_technical_indicators(daily)
            latest_signals = {strategy: bool(values.iloc[-1]) for strategy, values in indicators.items()}

            return {
                'basic_info': basic.iloc[0].to_dict(),
                'price_info': daily.iloc[-1].to_dict(),
                'technical_signals': latest_signals
            }
        except Exception as e:
            logger.error(f"查询股票 {ts_code} 失败: {str(e)}")
            return None

    @staticmethod
    def get_stock_list(
        selected_markets: Tuple[str, ...], 
        max_count: Optional[int] = None, 
        strategy_mode: str = "默认", 
        trade_date: Optional[str] = None
    ) -> Tuple[List[Tuple[str, str]], Dict[str, float]]:
        """优化点：
        1. 批量获取所有指数成分股（减少API调用次数）
        2. 合并行情数据查询（原2次→1次）
        3. 优化市场筛选逻辑（减少循环次数）
        4. 添加LRU缓存装饰器
        """
        from market_utils import get_market_status
        try:
            # 1. 获取全量股票数据（单次API调用）
            df = safe_api_call(
                pro.stock_basic,
                exchange='',
                list_status='L',
                fields='ts_code,name,market,list_date,industry'
            )
            if df.empty:
                logger.error("❌ 获取股票基础数据失败")
                return [], {}

            # 2. 获取合并后的行情数据（单次API调用）
            base_date = datetime.strptime(trade_date, '%Y%m%d') if trade_date else None
            valid_date = get_valid_trade_date(
                pro.daily,
                date_field='trade_date',
                base_date=base_date,
                max_back_days=5
            )
            if not valid_date:
                logger.error("❌ 无法获取有效交易日")
                return [], {}

            # 批量获取所有行情数据
            daily_data = safe_api_call(
                pro.daily,
                trade_date=valid_date,
                fields='ts_code,open,close,high,low,pct_chg,vol,amount'  # 确保包含open字段用于判断阳线
            )
            daily_basic = safe_api_call(
                pro.daily_basic,
                trade_date=valid_date,
                fields='ts_code,total_mv,turnover_rate'
            )

            # 3. 合并数据
            df = df.merge(daily_basic, on='ts_code', how='left').merge(
                daily_data, on='ts_code', how='left'
            )
            total_before_filter = len(df)

            # 获取当天停牌股票列表
            suspend_df = pro.suspend_d(suspend_type='S', trade_date=valid_date)
            suspend_stocks = set(suspend_df['ts_code'])  # 停牌股票的 TS 代码集合

            # 确保停牌股票信息正确
            logger.info(f"停牌股票数：{len(suspend_stocks)}")  # 输出停牌股票的数量，便于调试
            logger.info(f"停牌股票示例：{list(suspend_stocks)[:5]}")  # 输出一些停牌股票示例，查看数据是否正确

            # 4. 筛除停牌股票
            df = df[~df['ts_code'].isin(suspend_stocks)]
            logger.info(f"🚫 筛除停牌股：{total_before_filter} ➜ {len(df)}")

            # 5. 处理市场/板块选择
            index_map = {
                "中证500": "000905.SH", "沪深300": "000300.SH", "上证50": "000016.SH",
                "中证白酒": "399997.SZ", "中证消费": "000932.SH", "科创50": "000688.SH",
                "深证100": "399330.SZ", "北证50": "899050.BJ", "国证ETF": "399380.SZ"
            }

            # 批量获取所有需要的指数成分股
            needed_indexes = [index_map[m] for m in selected_markets if m in index_map]
            index_members = {}  # 如果不需要 `_get_index_members`，直接使用这个空字典

            # 构建筛选条件
            market_filters = []
            hs300_set = set()

            for m in selected_markets:
                if m in ["主板", "创业板", "科创板"]:
                    market_filters.append(df['market'] == m)
                elif m in index_map:
                    codes = index_members.get(index_map[m], set())
                    if m == "沪深300":
                        hs300_set = codes
                    market_filters.append(df['ts_code'].isin(codes))

            if market_filters:
                df = df[np.logical_or.reduce(market_filters)]
            else:
                logger.warning("⚠️ 未应用任何市场筛选条件")

            # 6. 数值转换
            for col in ['close', 'open', 'high', 'low', 'total_mv', 'turnover_rate', 'pct_chg']:
                if col not in df.columns:
                    logger.warning(f"⚠️ 列 {col} 不存在，跳过处理该列")
                    continue
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 7. 筛选条件配置
            mode_config = {
                "激进型": {"mv": (100_000, 3_000_000), "turnover": (1, 35), "pct": (-6, 12), "pe": (15, 45), "pb": (0.8, 8), "roe": 6},
                "稳健型": {"mv": (300_000, 6_000_000), "turnover": (0.5, 20), "pct": (-4, 8), "pe": (8, 35), "pb": (1.0, 5), "roe": 10},
                "穿线型": {"mv": (50_000, 10_000_000), "turnover": (0.2, 50), "pct": (-20, 30), "pe": (0, 200), "pb": (0.1, 20), "roe": 0},  # 新增穿线型策略参数
                "默认":   {"mv": (300_000, 3_000_000), "turnover": (1.5, 25), "pct": (-9, 28), "pe": (5, 50), "pb": (0.8, 8), "roe": 8}
            }
            cfg = mode_config.get(strategy_mode, mode_config["默认"])

            # 8. 新增获取基本面数据并进行筛选
            fundamental_basic = safe_api_call(
                pro.daily_basic,
                trade_date=valid_date,
                fields='ts_code,pe_ttm,pb'
            )

            # 获取 fina_indicator_vip 中的净资产收益率（roe）
            fundamental_fina = safe_api_call(
                pro.fina_indicator_vip,
                trade_date=valid_date,
                fields='ts_code,roe'
            )

            # 合并数据，使用 ts_code 作为连接键
            df = df.merge(fundamental_basic, on='ts_code', how='left')
            df = df.merge(fundamental_fina, on='ts_code', how='left')

            # 9. 添加基本面筛选条件：市盈率、净资产收益率、净资产收益率等
            filtered = df[
                (df['pe_ttm'].between(cfg['pe'][0], cfg['pe'][1])) &  # 动态市盈率范围
                (df['pb'].between(cfg['pb'][0], cfg['pb'][1])) &  # 动态市净率范围
                (df['roe'] >= cfg['roe'])  # 动态ROE条件
            ]
            logger.info(f"📊 基本面过滤：{len(filtered)}")

            # 10. 添加穿线型特殊筛选（更严格版本）
            if strategy_mode == "穿线型":
                # 计算需要的历史日期
                end_date = datetime.strptime(valid_date, '%Y%m%d')
                start_date = (end_date - timedelta(days=30)).strftime('%Y%m%d')  # 获取30天的数据

                # 初始化
                not_overbought_stocks = []

                # 分批处理股票，每批最多100支
                batch_size = 100
                stock_codes = filtered['ts_code'].unique()[:max_count if max_count else len(filtered)]  # 限制数量

                logger.info(f"📊 穿线型策略：开始筛选 {len(stock_codes)} 支股票")

                # 存储最终过滤结果
                filtered_stocks = []

                for i in range(0, len(stock_codes), batch_size):
                    batch_codes = stock_codes[i:i + batch_size]
                    batch_str = ','.join(batch_codes)  # 多个股票用逗号分隔

                    try:
                        # 使用 pro.daily 获取批量股票的历史数据
                        hist_daily = pro.daily(
                            ts_code=batch_str,
                            start_date=start_date,
                            end_date=valid_date
                        )

                        if hist_daily is not None and not hist_daily.empty:
                            # 按股票代码分组处理
                            for ts_code in batch_codes:
                                stock_data = hist_daily[hist_daily['ts_code'] == ts_code].sort_values('trade_date')

                                if len(stock_data) >= 5:
                                    try:
                                        # 计算累计涨幅
                                        pct_3d = stock_data['pct_chg'].tail(3).sum()
                                        pct_5d = stock_data['pct_chg'].tail(5).sum()
                                        pct_10d = stock_data['pct_chg'].tail(10).sum() if len(stock_data) >= 10 else pct_5d
                                        pct_20d = stock_data['pct_chg'].tail(20).sum() if len(stock_data) >= 20 else pct_10d
                                        pct_30d = stock_data['pct_chg'].sum()

                                        # 计算单日最大涨幅
                                        max_single_day = stock_data['pct_chg'].tail(10).max() if len(stock_data) >= 10 else stock_data['pct_chg'].max()

                                        # 🔍 成交额量能过滤 + 涨停日豁免
                                        avg_amount = stock_data['amount'].tail(5).mean()
                                        curr_amount = stock_data['amount'].iloc[-1]
                                        pct_chg_today = stock_data['pct_chg'].iloc[-1]
                                        vol_pass = (curr_amount > 1.5 * avg_amount) or (pct_chg_today >= 9.8)  # 涨停日豁免

                                        # 更严格的涨幅判断条件
                                        is_not_overbought = (
                                            pct_3d < 20 and
                                            pct_5d < 10 and
                                            pct_10d < 15 and
                                            pct_20d < 20 and
                                            pct_30d < 30 and
                                            max_single_day < 7
                                        )

                                        recent_pullback = stock_data['pct_chg'].tail(5).min() < -3

                                        if vol_pass and (is_not_overbought or (recent_pullback and pct_5d < 20)):
                                            not_overbought_stocks.append(ts_code)

                                        if len(not_overbought_stocks) <= 20:
                                            logger.debug(f"{ts_code}: 3d={pct_3d:.1f}%, 5d={pct_5d:.1f}%, "
                                                         f"10d={pct_10d:.1f}%, 20d={pct_20d:.1f}%, "
                                                         f"max_single={max_single_day:.1f}%, vol_pass={vol_pass}, "
                                                         f"pullback={recent_pullback}, selected={ts_code in not_overbought_stocks}")

                                    except Exception as e:
                                        logger.warning(f"处理{ts_code}时出错: {e}")
                        else:
                            logger.debug(f"批次 {i // batch_size + 1} 无数据返回")

                    except Exception as e:
                        logger.warning(f"获取历史数据失败（批次 {i // batch_size + 1}）: {e}")
                        for ts_code in batch_codes:
                            stock_row = filtered[filtered['ts_code'] == ts_code]
                            if not stock_row.empty:
                                if stock_row.iloc[0]['pct_chg'] <= 5:
                                    not_overbought_stocks.append(ts_code)

                # 过滤出涨幅适中的股票
                filtered = filtered.copy()
                filtered['is_not_overbought'] = filtered['ts_code'].isin(not_overbought_stocks)
                filtered_before = len(filtered)
                filtered = filtered[filtered['is_not_overbought']]

                logger.info(f"📈 涨幅筛选：{filtered_before} → {len(filtered)} 支（过滤{filtered_before - len(filtered)}支）")

                if len(filtered) < 20:
                    logger.warning("⚠️ 筛选后股票过少，使用当日涨幅作为筛选条件")
                    filtered = df[
                        (df['pe_ttm'].between(cfg['pe'][0], cfg['pe'][1])) &
                        (df['pb'].between(cfg['pb'][0], cfg['pb'][1])) &
                        (df['roe'] >= cfg['roe']) &
                        (df['pct_chg'] <= 6)
                    ]
                    logger.info(f"📈 放宽条件后：{len(filtered)} 支")

            # 11. 执行其他筛选和最终处理
            filtered = filtered[~filtered['name'].str.contains('ST|退', na=False)]
            logger.info(f"🚫 过滤ST及退市股： {len(filtered)}")

            filtered = filtered[
                ((filtered['market'] == '主板') & (filtered['close'] >= 1.5)) |
                ((filtered['market'].isin(['创业板', '科创板'])) & (filtered['close'] >= 2))
            ]
            logger.info(f"💰 收盘价过滤：{len(filtered)}")

            filtered = filtered[filtered['total_mv'].between(*cfg['mv'])]
            logger.info(f"🏦 市值区间过滤：{len(filtered)}")

            filtered = filtered[
                (filtered['turnover_rate'] >= cfg['turnover'][0]) &
                (filtered['turnover_rate'] <= cfg['turnover'][1])
            ]
            logger.info(f"🔄 换手率过滤：{len(filtered)}")

            filtered = filtered[filtered['pct_chg'].between(*cfg['pct'])]
            logger.info(f"📈 涨跌幅过滤：{len(filtered)}")

            # 12. 最终处理
            filtered = filtered.sort_values('list_date', ascending=False)
            if max_count:
                filtered = filtered.head(max_count)

            logger.info(f"✅ 最终筛选股票数: {len(filtered)}")
            stock_list = list(filtered[['ts_code', 'name']].itertuples(index=False, name=None))
            turnover_map = filtered.set_index('ts_code')['turnover_rate'].to_dict()

            return stock_list, turnover_map

        except Exception as e:
            logger.error(f"get_stock_list error: {e}")
            logger.error(traceback.format_exc())
            return [], {}

    @staticmethod
    def get_k_data(ts_code: str, days: int = 180, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        try:
            normal_rate_limiter.wait()
            today = end_date or datetime.today().strftime("%Y%m%d")
            past = (datetime.strptime(today, '%Y%m%d') - timedelta(days=days)).strftime('%Y%m%d')

            df = ts.pro_bar(
                ts_code=ts_code,
                start_date=past,
                end_date=today,
                freq='D',
                asset='E',
                adj='qfq',
                factors=['tor'],
                fields='ts_code,trade_date,open,high,low,close,vol'
            )

            if df is None or df.empty:
                return None

            # ✅ 主动检查 vol 是否存在
            if 'vol' not in df.columns:
                logger.warning(f"{ts_code} 缺失 vol 字段，跳过")
                return None

            df = df.sort_values("trade_date")
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df.set_index("trade_date", inplace=True)

            # ✅ 重命名 vol → volume，统一用法
            df.rename(columns={'vol': 'volume'}, inplace=True)

            if len(df) < Config.MIN_DATA_DAYS:
                return None

            return df
        except Exception as e:
            logger.error(f"获取 {ts_code} 数据失败: {str(e)}")
            return None

    @staticmethod
    def calculate_technical_indicators(df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], pd.DataFrame]:
        result = {}
        try:
            # === 基础指标 ===
            close = df["close"]
            high = df["high"]
            low = df["low"]
            open_price = df["open"]
            vol = df["volume"]

            # 动态均线系统
            windows = [5, 10, 20, 30, 60]
            for w in windows:
                df[f'ma{w}'] = close.rolling(w).mean()

            df['vol_ratio'] = vol / vol.rolling(20).mean()

            # === MACD指标计算 ===
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            dif = ema12 - ema26
            dea = dif.ewm(span=9, adjust=False).mean()
            macd = (dif - dea) * 2

            # === 布林带指标 ===
            df['boll_mid'] = close.rolling(20).mean()
            df['boll_std'] = close.rolling(20).std()
            df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
            df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']

            # === RSI指标 ===
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / (avg_loss + 1e-6)
            df['rsi'] = 100 - (100 / (1 + rs))

            # === 相对位置计算（短期为主）===
            high_20d = high.rolling(20).max()
            low_20d = low.rolling(20).min()
            position_20d = (close - low_20d) / (high_20d - low_20d + 0.001)

            # 超短线“底部”判断（更宽松）
            is_low_position = position_20d < 0.65
            oversold = df['rsi'] < 55
            below_boll_mid = close < df['boll_mid']
            at_bottom = is_low_position | oversold | below_boll_mid

            # === 一阳穿三线（短线） ===
            is_yang = close > open_price
            cross_today = (close > df['ma5']) & (close > df['ma10']) & (close > df['ma20'])
            prev_below_ma = (
                (close.shift(1) < df['ma5'].shift(1)) |
                (close.shift(1) < df['ma10'].shift(1)) |
                (close.shift(1) < df['ma20'].shift(1))
            )
            volume_increase = vol > vol.rolling(5).mean() * 1.05
            yang_cross_three_line = is_yang & cross_today & prev_below_ma & at_bottom & volume_increase

            # === OBV动量 ===
            df['obv'] = (np.sign(close.diff()) * vol).fillna(0).cumsum()
            df['obv_ma'] = df['obv'].rolling(20).mean()

            # === 信号生成 ===
            result = {
                # === 趋势型 ===
                "均线突破（5/20/30日）": (close > df[['ma5', 'ma20', 'ma30']].max(axis=1)),
                "均线多头排列": (df['ma5'] > df['ma20']) & (df['ma20'] > df['ma30']),
                "MACD零轴共振": (dif > 0) & (dea > 0) & (dif > dea),
                "趋势突破确认": (close > high.rolling(5).max()) & (vol > vol.rolling(5).mean() * 1.5),

                # === 动量型 ===
                "量价齐升": (df['vol_ratio'].between(1.5, 3)) & (close > close.shift(3) * 1.05),
                "主力资金共振": (macd > 0) & (df['vol_ratio'] > 1.8),
                "OBV动量引擎": (df['obv'] > df['obv_ma']) & (close > close.shift(5) * 1.03),

                # === 反转型 ===
                "超跌反弹（RSI+BOLL）": (close < df['boll_lower']) & (df['rsi'] < 30),
                "底部反转确认": (close < df['boll_lower'] * 0.98) & (vol > vol.rolling(5).mean() * 1.2) & (df['rsi'] < 35),

                # === 风险型 ===
                "趋势破位（MA60+MACD死叉）": (close < df['ma60']) & (dif < dea),
                "高位滞涨风险": (close > df['boll_upper']) & (df['rsi'] > 70) & (vol < vol.rolling(5).mean() * 0.8),

                # === 穿线型 ===
                "一阳穿三线": yang_cross_three_line
            }

            return result, df

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"指标计算错误: {str(e)}")
            return {}, df

    @staticmethod
    def check_risk(df: pd.DataFrame) -> bool:
        return True

class MarketNeutralAnalyzer:
    @staticmethod
    def calculate_relative_strength(ts_code, trade_date, lookback=20):
        """计算相对强弱得分(-50到50)"""
        try:
            # 获取股票和基准指数过去lookback天的收益率
            stock_ret = MarketNeutralAnalyzer._get_stock_returns(ts_code, trade_date, lookback)
            bench_ret = MarketNeutralAnalyzer._get_benchmark_returns(trade_date, lookback)
            
            # 计算相对强弱
            relative_strength = (stock_ret - bench_ret).mean()
            return min(max(relative_strength * 100, -50), 50)  # 限制在-50到50之间
        except Exception as e:
            logger.error(f"计算相对强弱失败 {ts_code}: {str(e)}")
            return 0

    @staticmethod
    def _get_stock_returns(ts_code, trade_date, lookback):
        """获取股票收益率"""
        end_date = datetime.strptime(trade_date, '%Y%m%d')
        start_date = end_date - timedelta(days=lookback)
        
        df = safe_api_call(
            pro.daily,
            ts_code=ts_code,
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d'),
            fields='trade_date,pct_chg'
        )
        return df['pct_chg'] / 100 if not df.empty else pd.Series([0]*lookback)

    @staticmethod
    def _get_benchmark_returns(trade_date, lookback, benchmark='000300.SH'):
        """获取基准指数收益率"""
        end_date = datetime.strptime(trade_date, '%Y%m%d')
        start_date = end_date - timedelta(days=lookback)
        
        df = safe_api_call(
            pro.index_daily,
            ts_code=benchmark,
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d'),
            fields='trade_date,pct_chg'
        )
        return df['pct_chg'] / 100 if not df.empty else pd.Series([0]*lookback)


class RecommendationTracker:
    def __init__(self):
        self.data_file = "recommendations.pkl"
        try:
            self.recommendations = pd.read_pickle(self.data_file)
            logger.info(f"✅ 成功加载推荐历史，共 {len(self.recommendations)} 条记录")
        except:
            logger.info("📂 无历史记录，创建新文件")
            self.recommendations = pd.DataFrame()

    def add_recommendation(self, stock_data: dict, recommend_date: Optional[str] = None):
        if self.stock_exists(stock_data['ts_code']):
            logger.info(f"⚠️ {stock_data['ts_code']} 已存在推荐记录")
            return False

        stock_data['recommend_date'] = recommend_date or datetime.today().strftime('%Y-%m-%d')

        new_record = pd.DataFrame([stock_data])
        self.recommendations = pd.concat([self.recommendations, new_record], ignore_index=True)
        self._save_data()
        logger.info(f"✅ 已添加推荐: {stock_data['ts_code']} ({stock_data['recommend_date']})")
        return True

    def export_to_watchlist(self):
        # 删除导出watchlist文件的功能
        logger.info("📝 导出watchlist功能已禁用")
        return

    def stock_exists(self, ts_code: str) -> bool:
        if self.recommendations.empty:
            return False
        return ts_code in self.recommendations['ts_code'].values

    def remove_stock(self, ts_code: str):
        if self.recommendations.empty:
            logger.info("⚠️ 当前无记录可删除")
            return
        before_count = len(self.recommendations)
        self.recommendations = self.recommendations[self.recommendations['ts_code'] != ts_code]
        self._save_data()
        after_count = len(self.recommendations)
        logger.info(f"🗑️ 删除完成，记录数: {before_count} -> {after_count}")

    def clear(self):
        self.recommendations = pd.DataFrame()

    def _save_data(self):
        self.recommendations.to_pickle(self.data_file)


# =====================
# 实例化 tracker
# =====================

tracker = RecommendationTracker()

def calculate_position(score: float, pct_change: float = 0.0, risk_warnings: List[str] = None, strategy_mode: str = "稳健型") -> str:
    if risk_warnings is None:
        risk_warnings = []

    # ❌ 强制过滤极端涨幅
    # 根据策略类型调整涨幅限制
    if strategy_mode == "稳健型":
        max_pct_change = 9.5  # 稳健型限制更严格
    elif strategy_mode == "穿线型":
        max_pct_change = 12.0  # 穿线型允许较高涨幅
    else:
        max_pct_change = 15.0  # 激进型允许更高的涨幅
        
    if pct_change >= max_pct_change:
        return "❌ 不建议买入"

    # 🚨 高波动警告减仓或剔除
    volatility_penalty = 0.5 if "高波动" in risk_warnings else 1.0
    if volatility_penalty < 1.0:
        return "❌ 不建议买入"  # 波动过大，直接剔除

    # 📈 根据策略调整仓位分配
    if strategy_mode == "稳健型":
        # 稳健型策略较低的仓位分配
        if score >= 150:
            base_position = 0.12
        elif score >= 145:
            base_position = 0.10
        elif score >= 140:
            base_position = 0.09
        elif score >= 135:
            base_position = 0.08
        elif score >= 130:
            base_position = 0.07
        elif score >= 125:
            base_position = 0.06
        elif score >= 120:
            base_position = 0.05
        elif score >= 115:
            base_position = 0.04
        elif score >= 110:
            base_position = 0.03
        else:
            return "❌ 不建议买入"
    
    elif strategy_mode == "穿线型":
        # 穿线型策略的仓位分配
        if score >= 180:
            base_position = 0.15
        elif score >= 160:
            base_position = 0.13
        elif score >= 150:
            base_position = 0.12
        elif score >= 140:
            base_position = 0.10
        elif score >= 130:
            base_position = 0.08
        elif score >= 120:
            base_position = 0.06
        elif score >= 110:
            base_position = 0.04
        elif score >= 100:
            base_position = 0.03
        else:
            return "❌ 不建议买入"

    elif strategy_mode == "激进型":
        # 激进型策略允许较高的仓位分配
        if score >= 200:
            base_position = 0.15
        elif score >= 180:
            base_position = 0.14
        elif score >= 170:
            base_position = 0.13
        elif score >= 160:
            base_position = 0.12
        elif score >= 150:
            base_position = 0.11
        elif score >= 140:
            base_position = 0.10
        elif score >= 130:
            base_position = 0.09
        elif score >= 125:
            base_position = 0.08
        elif score >= 120:
            base_position = 0.07
        elif score >= 115:
            base_position = 0.06
        elif score >= 110:
            base_position = 0.05
        elif score >= 105:
            base_position = 0.04
        elif score >= 100:
            base_position = 0.03
        else:
            return "❌ 不建议买入"

    final_position = base_position * volatility_penalty

    # 🧾 转换为文本标签，适应不同策略
    if final_position >= 0.15:
        if strategy_mode == "激进型":
            return "15%-20%"
        elif strategy_mode == "穿线型":
            return "15%-18%"
        else:
            return "12%-15%"
    elif final_position >= 0.10:
        if strategy_mode == "激进型":
            return "10%-15%" 
        elif strategy_mode == "穿线型":
            return "10%-13%"
        else:
            return "8%-12%"
    elif final_position >= 0.05:
        return "5%-8%"
    elif final_position >= 0.01:
        return "≤5%"
    else:
        return "⚠️ 仓位过小"







# ===== 界面相关函数 =====
def get_tracking_html():
    html = "<h3>📊 推荐历史记录</h3><ul>"
    if tracker.recommendations.empty:
        return "<h3>📭 暂无推荐记录</h3>"

    for _, row in tracker.recommendations.iterrows():
        html += f"<li>{row['recommend_date']} - {row['name']} ({row['ts_code']})</li>"
    html += "</ul>"
    return html



HIGH_VOLATILITY_INDUSTRIES = [
    "通信设备", "半导体", "新能源", "生物医药", "军工", "游戏", "创业板"
]

@lru_cache(maxsize=5000)
def is_high_volatility_industry(ts_code: str) -> bool:
    try:
        basic_info = safe_api_call(pro.stock_basic, ts_code=ts_code, fields='ts_code,industry')
        if basic_info.empty:
            return False
        industry = basic_info.iloc[0]['industry']
        return industry in HIGH_VOLATILITY_INDUSTRIES
    except Exception as e:
        logger.warning(f"行业查询失败 {ts_code}: {str(e)}")
        return False

@lru_cache(maxsize=1000)
def check_negative_announcements(ts_code: str) -> bool:
    # 暂无权限，直接跳过
    return False
    try:
        start_date = (datetime.today() - timedelta(days=7)).strftime('%Y%m%d')
        df = safe_api_call(pro.anns_d, ts_code=ts_code, start_date=start_date)
        if df.empty:
            return False
        negative_keywords = ['诉讼', '仲裁', '立案调查', '项目终止', '控制权变更', '重大风险']
        return df['title'].str.contains('|'.join(negative_keywords)).any()
    except Exception as e:
        logger.warning(f"{ts_code} 公告检查失败: {str(e)}")
        return False



@lru_cache(maxsize=1000)
def check_earnings_warning(ts_code: str) -> bool:
    try:
        current_year = datetime.today().year
        df = safe_api_call(pro.forecast_vip, ts_code=ts_code, start_date=f"{current_year}0101")
        if df.empty:
            return False
        return df['type'].str.contains("预减|预亏|续亏").any()
    except Exception as e:
        logger.warning(f"{ts_code} 预警检查失败: {str(e)}")
        return False








import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_type_weight_safe(strategy_type: str, type_weights: Dict[str, float]) -> float:
    """安全获取策略类型权重，三重兜底"""
    return type_weights.get(strategy_type) if type_weights.get(strategy_type) is not None else STRATEGY_TYPE_WEIGHTS.get(strategy_type, 1.0)


def evaluate_yang_cross_strength(df: pd.DataFrame) -> str:
    """
    评估一阳穿三线信号的质量，返回评级文本。
    """
    close = df['close'].iloc[-1]
    open_price = df['open'].iloc[-1]
    high = df['high'].iloc[-1]
    low = df['low'].iloc[-1]
    vol = df['volume'].iloc[-1]
    ma5 = df['ma5'].iloc[-1]
    ma10 = df['ma10'].iloc[-1]
    ma20 = df['ma20'].iloc[-1]

    # 实体强度：阳线长度
    body_pct = (close - open_price) / (open_price + 1e-6)

    # 均线发散结构
    trend_order = (ma5 > ma10) and (ma10 > ma20)

   
    recent_high = df['high'].rolling(10).max().iloc[-1]
    space_pct = (recent_high - close) / (close + 1e-6)

    # 成交量放大程度
    vol_mean = df['volume'].rolling(5).mean().iloc[-1]
    vol_ratio = vol / (vol_mean + 1e-6)

    score = 0
    if body_pct > 0.02: score += 1
    if trend_order: score += 1
    if space_pct > 0.05: score += 1
    if vol_ratio > 1.5: score += 1

    if score >= 3:
        return "🔥高质量穿线"
    elif score == 2:
        return "⚠️中等穿线"
    else:
        return "❌弱穿线"

def analyze_stocks(stock_list_with_turnover: Tuple[List[Tuple[str, str]], Dict[str, float]],
                   strategies: List[str],
                   custom_weights: Dict[str, int],
                   max_stocks: int,
                   strategy_mode: str,
                   trade_date: Optional[str] = None,
                   export_watchlist: bool = True,
                   top_n: int = 10) -> List[Tuple]:
    stock_list, turnover_map = stock_list_with_turnover

    if trade_date:
        trade_date = datetime.strptime(trade_date, '%Y%m%d')
    else:
        trade_date = datetime.today()

    actual_trade_date = get_valid_trade_date(api_func=pro.daily, date_field='trade_date', base_date=trade_date)
    if not actual_trade_date:
        logger.error("❌ 无法获取有效交易日，结束分析！")
        return []

    logger.info(f"🚀 分析启动：{len(stock_list)} 支股票 ｜ 模式：{strategy_mode} ｜ 数据日期：{actual_trade_date}")
    
    # ✅ 板块热度评分系统集成部分
    if isinstance(actual_trade_date, datetime):
        trade_date_str = actual_trade_date.strftime('%Y%m%d')
    else:
        trade_date_str = actual_trade_date  # 若已为字符串，直接使用
    
    # 修改：回测模式下跳过板块评分
    if IS_BACKTEST:
        logger.info(f"🔙 回测模式：跳过板块热度评分")
        sector_scores = {}
        concept_map = {}
        concept_name_to_ts_code = {}
    else:
        logger.info(f"📅 开始获取板块数据，日期: {trade_date_str}")
        
        sector_scores = get_sector_strength_scores(trade_date_str)
        logger.info(f"📊 板块评分结果数量: {len(sector_scores)}")
        if sector_scores:
            # 打印前5个板块的评分
            sample_scores = dict(list(sector_scores.items())[:5])
            logger.info(f"📊 板块评分样例: {sample_scores}")
        else:
            logger.warning("⚠️ 板块评分为空！可能接口调用失败")
        
        concept_map = load_concept_to_stock_map()
        logger.info(f"📚 加载概念股票映射: {len(concept_map)} 个概念")
        if concept_map:
            # 打印一个样例
            sample_concept = list(concept_map.keys())[0]
            sample_stocks = concept_map[sample_concept][:3]
            logger.info(f"📚 概念映射样例: {sample_concept} -> {sample_stocks}")
        
        concept_name_to_ts_code = load_concept_name_to_code()  # 从文件加载映射
        logger.info(f"🔗 加载概念代码映射: {len(concept_name_to_ts_code)} 个映射")
        if concept_name_to_ts_code:
            # 打印前3个映射
            sample_mappings = dict(list(concept_name_to_ts_code.items())[:3])
            logger.info(f"🔗 概念代码映射样例: {sample_mappings}")
        
        # 如果文件不存在，提示用户运行构建脚本
        if not concept_map or not concept_name_to_ts_code:
            logger.warning("⚠️ 概念映射文件不存在，请先运行 build_concept_mapping.py 构建映射")
            concept_name_to_ts_code = {}

    if not IS_BACKTEST:
        concept_data = get_concept_trends(trade_date)
        concept_trend_score = calculate_concept_trend_score(concept_data, trade_date)
    else:
        concept_trend_score = 0  # 回测模式下，跳过概念趋势得分的计算，设置为0
        
    scored_stocks = []
    ts_codes = [ts_code for ts_code, _ in stock_list]

    # Initialize various data
    initialize_industry_and_concept(ts_codes)
    initialize_moneyflow_scores(ts_codes)
    initialize_risk_data(ts_codes)
    initialize_stk_limit(ts_codes)
    initialize_block_trade(ts_codes)
    initialize_top_inst()
    initialize_share_float_data(ts_codes)
    initialize_holdernumber_data(ts_codes)
   
    current_suspend = set(pro.suspend_d(trade_date=actual_trade_date)['ts_code'].str.upper())
    ts_codes = [code for code in ts_codes if code not in current_suspend]

    # 获取市场状态调整后的权重
    type_weights = STRATEGY_TYPE_WEIGHTS.copy()

    if strategy_mode == "稳健型":
        type_weights["趋势型"] *= 1.2
        type_weights["动量型"] *= 0.8
        type_weights["反转型"] *= 0.9
        type_weights["市场中性型"] *= 1.2
        type_weights["风险型"] = max(-2.0, type_weights.get("风险型", -1.2) * 1.4)
    elif strategy_mode == "激进型":
        type_weights["趋势型"] *= 0.9
        type_weights["动量型"] *= 1.15
        type_weights["反转型"] *= 1.1
        type_weights["市场中性型"] *= 0.75
        type_weights["风险型"] = max(-3.0, type_weights.get("风险型", -1.0) * 0.8)
    elif strategy_mode == "穿线型":
        # 穿线型策略特殊调整
        type_weights["穿线型"] *= 1.5  
        type_weights["趋势型"] *= 0.8
        type_weights["动量型"] *= 0.8
        type_weights["风险型"] = max(-2.0, type_weights.get("风险型", -1.0) * 0.6)

    # Step 2: 再根据市场状态动态调整权重
    if IS_BACKTEST and custom_weights:
        logger.info("🎯 使用回测传入的市场扰动权重")
        for key, market_weight in custom_weights.items():
            type_weights[key] = type_weights.get(key, 1.0) * float(market_weight)
    else:
        market_weights = adjust_strategy_weights_by_market(trade_date=actual_trade_date)
        logger.info("📡 使用实时市场扰动权重")
        for key, market_weight in market_weights.items():
            type_weights[key] = type_weights.get(key, 1.0) * float(market_weight)

    # Step 3: 最终合并权重
    merged_weights = type_weights
    logger.info(f"⚖️ 最终策略权重 (策略模式: {strategy_mode}):\n" +
                "\n".join([f"- {k}: {v:.2f}" for k, v in merged_weights.items()]))

    daily_info_df = safe_api_call(pro.daily, trade_date=actual_trade_date, fields='ts_code,pct_chg')
    
    # 检查 'pct_chg' 列是否存在
    if 'pct_chg' in daily_info_df.columns:
        pct_chg_map = daily_info_df.set_index('ts_code')['pct_chg'].to_dict()
    else:
        logger.warning("❌ 'pct_chg' 列在数据框中未找到")
        pct_chg_map = {}  # 如果没有找到，返回空字典

    removal_stats = defaultdict(int)

    # 记录已处理的股票，避免重复
    seen_stocks = set()

    def process_stock(ts_code_name):
        ts_code, name = ts_code_name
        try:
            # 检查是否已经处理过该股票，避免重复
            if ts_code in seen_stocks:
                return None
            seen_stocks.add(ts_code)  # 标记该股票已处理

            if check_earnings_warning(ts_code):
                removal_stats["风险预警"] += 1
                logger.info(f"🛑 {ts_code} 被筛除，原因：风险预警")
                return None

            df = StockAnalyzer.get_k_data(ts_code, days=60, end_date=actual_trade_date)
            if df is None or len(df) < Config.MIN_DATA_DAYS:
                removal_stats["数据不足"] += 1
                logger.info(f"🛑 {ts_code} 被筛除，原因：数据不足")
                return None

            indicators, df = StockAnalyzer.calculate_technical_indicators(df)

            financial_score = evaluate_financials(ts_code)
            score = 0
            score_details = {}
            score += financial_score * 1
            score_details['基本面得分'] = financial_score * 1

            # 策略匹配过滤
            raw_matched = list(set(
                [s for s in strategies if s in indicators and indicators[s].iloc[-1]] +
                [s for s in POTENTIAL_SIGNALS if s in indicators and indicators[s].iloc[-1]]
            ))

            # 定义有效策略白名单
            valid_strategies = list(STRATEGY_WEIGHTS.keys()) + POTENTIAL_SIGNALS
            matched = [s for s in raw_matched if s in valid_strategies]

            if strategy_mode == "穿线型":
                # 检查是否满足"一阳穿三线"指标
                if "一阳穿三线" in indicators and indicators["一阳穿三线"].iloc[-1]:
                    # ✅ 新增：穿线质量判断，提前筛除弱信号
                    quality = evaluate_yang_cross_strength(df)
                    if quality == "❌弱穿线":
                        removal_stats["穿线信号弱"] += 1
                        logger.info(f"🛑 {ts_code} 被筛除，原因：穿线信号强度不足（弱穿线）")
                        return None

                    # 如果满足，强制添加到匹配策略中
                    if "一阳穿三线" not in matched:
                        matched.append("一阳穿三线")
                    # 给予额外加分
                    score += 30
                    score_details['穿线型加分'] = 30

                    # 写入评分等级
                    score_details['穿线评分'] = quality
                    if quality == "🔥高质量穿线":
                        score += 15
                    elif quality == "⚠️中等穿线":
                        score += 0  # 保留中等，不额外加分

                elif not matched:
                    # 如果是穿线型策略模式但不满足一阳穿三线且没有其他匹配策略，则跳过
                    removal_stats["不满足穿线条件"] += 1
                    logger.info(f"🛑 {ts_code} 被筛除，原因：穿线型策略模式下不满足穿线条件")
                    return None

            if not matched:
                removal_stats["无匹配策略"] += 1
                logger.info(f"🛑 {ts_code} 被筛除，原因：无匹配策略")
                return None

            # 权重计算及策略得分
            weights = {s: STRATEGY_WEIGHTS[s] * merged_weights.get(get_strategy_type(s), 1.0) for s in STRATEGY_WEIGHTS}
            weights.update(custom_weights)

            matched = sorted(matched, key=lambda s: weights.get(s, 0), reverse=True)[:3]
            matched = [s for s in matched if get_strategy_type(s) != "风险型"]

            tech_score = 0
            score_details['技术面得分细节'] = []

            for s in matched:
                # 获取策略的类型并根据动态权重进行加权
                strategy_type = get_strategy_type(s)
                strategy_score = weights.get(s, 10) * merged_weights.get(strategy_type, 1.0)

                tech_score += strategy_score
                score_details['技术面得分细节'].append(f"{s}: {strategy_score:.1f}")

            score += tech_score
            score_details['技术面得分'] = tech_score

            market_neutral_weight = merged_weights.get("市场中性型", 1.0)
            rs_score = MarketNeutralAnalyzer.calculate_relative_strength(ts_code, actual_trade_date)
            neutral_bonus = rs_score * 10
            neutral_bonus_weighted = neutral_bonus * market_neutral_weight * 0.5
            score += neutral_bonus_weighted
            score_details['市场中性得分'] = neutral_bonus_weighted
            
            # 其他评分逻辑
            
            old_score = score
            # 修改：回测模式下跳过板块热度加分
            if IS_BACKTEST:
                logger.debug(f"🔙 回测模式：{ts_code} 跳过板块热度加分")
                score_details['板块热度加分'] = 0
            else:
                score = inject_sector_score(score, ts_code, concept_name_to_ts_code, sector_scores, concept_map, weight=0.3)
                score_details['板块热度加分'] = round(score - old_score, 2)

            risk_penalty = evaluate_risk_factors(ts_code)  # 风险因子
            score -= risk_penalty
            logger.debug(f"🔴 {ts_code} 风险扣分: {risk_penalty}")

            share_float_penalty = evaluate_share_float(ts_code)  # 限售
            score += share_float_penalty
            logger.debug(f"💸 {ts_code} 未来限售解禁得分: {share_float_penalty}")

            holdernumber_score = evaluate_holdernumber(ts_code)  # 股东人数变化
            score += holdernumber_score
            logger.debug(f"👥 {ts_code} 股东人数变化得分: {holdernumber_score}")

            express_score = evaluate_express(ts_code)  # 快速财报评分
            score += express_score
            logger.debug(f"📈 {ts_code} 快速财报得分: {express_score}")

            top_inst_score = check_top_inst(ts_code)  # 主力资金评分
            score += top_inst_score
            logger.debug(f"🏦 {ts_code} 主力资金得分: {top_inst_score}")
            
            score += concept_trend_score
            score_details['概念趋势得分'] = concept_trend_score

            # 其他扣分逻辑
            current_price = df['close'].iloc[-1]
            limit_info = stk_limit_cache.get(ts_code)
            if limit_info and current_price <= limit_info['down_limit'] * 1.005:
                removal_stats["接近跌停"] += 1
                logger.info(f"🛑 {ts_code} 被筛除，原因：接近跌停")
                return None

            turnover = turnover_map.get(ts_code)
            if turnover is None:
                removal_stats["换手率过低"] += 1
                logger.info(f"🛑 {ts_code} 被筛除，原因：换手率过低")
                return None
            
            day_volatility = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-2]
            if day_volatility > 0.15:  # 单日波动超15%
                removal_stats["异常波动"] += 1
                return None
            
            # 根据策略模式设置不同的换手率评分
            if strategy_mode == "稳健型":
                turnover_score = 6 if 1.5 <= turnover <= 12 else 4
            elif strategy_mode == "穿线型":
                # 穿线型策略更看重换手率
                turnover_score = 10 if 4 <= turnover <= 20 else 6
            else:
                turnover_score = 8 if 4 <= turnover <= 15 else 4
                
            score += turnover_score
            score_details['换手率加分'] = turnover_score

            # 最后返回得分
            return (score, ts_code, name, matched, df['close'].pct_change(5).iloc[-1] * 100, df, score_details)

        except Exception as e:
            logger.error(f"{ts_code} 分析异常: {e}")
            return None

    # 处理股票数据
    with concurrent.futures.ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        futures = [executor.submit(process_stock, stock) for stock in stock_list[:max_stocks]]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                scored_stocks.append(result)

    # 结果统计和去重
    logger.info(f"✅ 分析完成：总{len(stock_list)}支，候选{len(scored_stocks)}支")
    for reason, count in removal_stats.items():
        logger.info(f"⚠️ 被筛除的原因统计：{reason}：{count}支")

    # 根据策略模式调整推荐机制
    if strategy_mode == "穿线型":
        # 提取满足"一阳穿三线"条件的股票
        cross_line_stocks = [stock for stock in scored_stocks if "一阳穿三线" in stock[3]]
        
        if cross_line_stocks:
            logger.info(f"🎯 找到满足一阳穿三线的股票：{len(cross_line_stocks)}支")
            # 修改：显示所有命中的股票，不限制数量，但仍按照得分排序
            final_stocks = sorted(cross_line_stocks, key=lambda x: x[0], reverse=True)
        else:
            logger.warning("⚠️ 未找到满足一阳穿三线条件的股票，将使用普通排序")
            final_stocks = sorted(scored_stocks, key=lambda x: x[0], reverse=True)[:top_n]
    elif strategy_mode == "稳健型":
        final_stocks = diversify_recommendations(scored_stocks, max_recommend=top_n)
    else:
        final_stocks = sorted(scored_stocks, key=lambda x: x[0], reverse=True)[:top_n]

    # 去重操作：确保每只股票只出现一次
    final_stocks = list({stock[1]: stock for stock in final_stocks}.values())

    for score, ts_code, name, matched, pct_change, df, score_details in final_stocks:
        logger.info(f"🍇 {ts_code} {name} 总分: {score:.1f} | 组成: {score_details}")

    if export_watchlist:
        tracker.clear()
        # 导出股票列表时，如果是穿线型策略且数量过多，只导出前top_n只
        export_stocks = final_stocks
        if strategy_mode == "穿线型" and len(final_stocks) > top_n:
            logger.info(f"🔄 穿线型策略检测到{len(final_stocks)}支股票，但只导出得分最高的{top_n}支")
            export_stocks = final_stocks[:top_n]
            
        for score, ts_code, name, matched, pct_change, df, score_details in export_stocks:
            tracker.add_recommendation({
                'ts_code': ts_code,
                'name': name,
                'strategies': matched,
                'score': score,
                'price': df['close'].iloc[-1],
                'position': calculate_position(score, pct_change, [], strategy_mode),  # 添加strategy_mode参数
                'is_top': True
            })
        #tracker.export_to_watchlist()

    logger.info(f"📢 最终推荐：{[ts_code for _, ts_code, *_ in final_stocks]}")
    
    return final_stocks




def chat_interface(user_input: str, market_type: List[str], max_stocks: int, strategy_mode: str, history: List) -> Tuple[List, List]:
    default_trigger_phrases = ["推荐股票", "帮我推荐", "选股", "给我推荐", "找股票"]
    markets_str = ", ".join(market_type)

    # 启用全策略（主动剔除风险型策略）
    strategy_items = [s for s in STRATEGY_WEIGHTS.keys() if get_strategy_type(s) != "风险型"]
    custom_weights = {s: w for s, w in STRATEGY_WEIGHTS.items() if get_strategy_type(s) != "风险型"}

    if strategy_mode == "稳健型":
        explanation = "📘 【稳健型】：趋势型为主，适度保留动量与反弹，严格规避风险。"
       
    elif strategy_mode == "激进型":
        explanation = "🚀 【激进型】：突出短线动量与量能机会，趋势适当降低，风险策略已隔离。"
        
    elif strategy_mode == "穿线型":
        explanation = "🌟 【穿线型】：专注于捕捉均线穿越信号，主要寻找一阳穿三线形态，注重突破确认。"
        # 穿线型模式下，强制添加一阳穿三线策略
        if "一阳穿三线" not in strategy_items:
            strategy_items.append("一阳穿三线")
            custom_weights["一阳穿三线"] = STRATEGY_WEIGHTS.get("一阳穿三线", 45)
        
    elif any(phrase in user_input for phrase in default_trigger_phrases):
        explanation = "🤖 泛化请求：均衡启用策略，已自动剔除风险策略，综合评估机会。"
    else:
        response = DeepSeekAPI.call_deepseek(user_input)
        strategy_items, explanation, custom_weights = DeepSeekAPI.parse_strategies(response)
        # 再次过滤风险型策略
        strategy_items = [s for s in strategy_items if get_strategy_type(s) != "风险型"]
        custom_weights = {s: w for s, w in custom_weights.items() if get_strategy_type(s) != "风险型"}

        if not strategy_items:
            error_msg = f"⚠️ 未识别到有效策略\n{explanation}"
            history.append((user_input, error_msg))
            return history, history

    try:
        logger.info(f"📋 启用策略数：{len(strategy_items)}，市场：{markets_str}")
        history.append((user_input, f"📋 策略明细：\n{explanation}\n\n🔍 正在扫描 {markets_str} (最多分析 {max_stocks} 支)..."))

        # 获取股票列表
        stock_list = StockAnalyzer.get_stock_list(
            tuple(market_type),
            max_count=max_stocks,
            strategy_mode=strategy_mode
        )
        if not stock_list:
            history.append(("", "⚠️ 获取股票列表失败"))
            return history, history

        scored_stocks = analyze_stocks(stock_list, strategy_items, custom_weights, max_stocks, strategy_mode)

        if not scored_stocks:
            history.append(("", "❌ 没有找到符合条件的股票"))
        else:
            result_msg = "✅ 推荐股票 (按分数排序):\n"
            
            # 添加针对穿线型策略的特殊提示
            if strategy_mode == "穿线型" and len(scored_stocks) > 10:
                result_msg += f"🔍 找到 {len(scored_stocks)} 支满足一阳穿三线条件的股票\n"
                
            result_msg += "排名 | 代码 | 名称 | 得分 | 5日涨幅 | 仓位 | 匹配策略 | 风险提示\n"
            result_msg += "-" * 100 + "\n"
            
            # 如果是穿线型策略且结果过多，考虑分页显示或限制结果行数避免UI显示问题
            max_display = len(scored_stocks)
            if strategy_mode == "穿线型" and max_display > 50:
                max_display = 50  # UI显示限制，最多显示50行
            
            # ========== 关键修改开始：UI显示过滤 ==========
            valid_strategies = list(STRATEGY_WEIGHTS.keys()) + POTENTIAL_SIGNALS  # 有效策略列表
            for i, (score, ts_code, name, matched, pct_change, _, score_details) in enumerate(scored_stocks[:max_display], 1):
                # 清洗策略列表
                clean_matched = [s for s in matched if s in valid_strategies]

                # 如果包含“一阳穿三线”，在其后添加评分标签
                if "一阳穿三线" in clean_matched and isinstance(score_details, dict):
                    quality = score_details.get("穿线评分", "")
                    if quality:
                        index = clean_matched.index("一阳穿三线")
                        clean_matched[index] = f"一阳穿三线（{quality}）"

                # 提取买点提示
                buy_signals = [s for s in clean_matched if s in KEY_BUY_SIGNALS]
                buy_info = "🔥买点:" + ",".join(buy_signals) if buy_signals else ""

                # 提取风险提示
                risk_warnings = []
                if isinstance(score_details, dict) and 'risk_warnings' in score_details:
                    risk_warnings = score_details['risk_warnings']
                risk_info = " | ".join(risk_warnings) if risk_warnings else "无"

                position = calculate_position(score, pct_change, risk_warnings, strategy_mode)

                # 输出最终结果行
                result_msg += (
                    f"{i:2d}. {ts_code.split('.')[0]} {name[:10]} | "
                    f"📊{int(score)} | "
                    f"📈{pct_change:.1f}% | "
                    f"⚖️{position} | "
                    f"{buy_info} {', '.join(clean_matched[:3])} | "
                    f"{risk_info}\n"
                )

            # ========== 关键修改结束 ==========
            
            # 如果有更多结果未显示，添加提示
            if len(scored_stocks) > max_display:
                result_msg += f"\n... 还有 {len(scored_stocks) - max_display} 支满足条件的股票未显示 (总共 {len(scored_stocks)} 支) ..."
                
            history.append(("", result_msg))

    except Exception as e:
        logger.error(f"界面交互错误: {str(e)}", exc_info=True)
        history.append(("", f"⚠️ 系统错误: {str(e)}"))

    return history, history



def query_stock(ts_code: str) -> str:
    stock_info = StockAnalyzer.get_single_stock_info(ts_code)
    if not stock_info:
        return "❌ 未找到该股票或数据获取失败"
    
    basic = stock_info['basic_info']
    price = stock_info['price_info']
    signals = stock_info['technical_signals']
    signal_msgs = [f"🔹 {s}: {'✅' if v else '❌'}" for s, v in signals.items() if v]
    
    return f"""
📈 股票信息 [{basic['ts_code']}]
----------------------------
名称：{basic['name']}
行业：{basic.get('industry', 'N/A')}
上市日期：{basic['list_date']}
市场：{basic['market']}

💵 最新行情（{price['trade_date']}）
----------------------------
收盘价：{price['close']}
涨跌幅：{price['pct_chg']}%
成交量：{price['vol']/10000:.2f}万手
成交额：{price['amount']/10000:.2f}万元

📊 技术信号
----------------------------
{'\n'.join(signal_msgs) if signal_msgs else '⚠️ 未触发任何技术信号'}
"""

# ===== 默认值配置 =====
DEFAULT_TURNOVER = 8000   # 今日成交额默认值（亿元）
DEFAULT_AVG_TURNOVER = 9000  # 近30日平均成交额默认值（亿元）
# ===== 涨停数据文件缓存 =====
LIMIT_UP_CACHE_FILE = 'limit_up_cache.json'

# 启动时加载缓存
if os.path.exists(LIMIT_UP_CACHE_FILE):
    with open(LIMIT_UP_CACHE_FILE, 'r', encoding='utf-8') as f:
        _limit_up_cache = json.load(f)
else:
    _limit_up_cache = {}

# ===== 缓存今日行情数据，避免重复调用 =====
_daily_data_cache = None
def get_last_trade_date() -> str:
    """获取最近一个交易日"""
    today = datetime.today()
    while today.weekday() >= 5:  # 周六日
        today -= timedelta(days=1)
    return today.strftime('%Y%m%d')

def get_daily_data(trade_date: str):
    global _daily_data_cache
    if _daily_data_cache is None or _daily_data_cache.get('date') != trade_date:
        df = safe_api_call(pro.daily, trade_date=trade_date, fields='ts_code,pct_chg')
        _daily_data_cache = {'date': trade_date, 'data': df}
    return _daily_data_cache['data'] if _daily_data_cache else pd.DataFrame()

def get_up_stock_count(trade_date: str) -> int:
    try:
        df = get_daily_data(trade_date)
        return (df['pct_chg'] > 0).sum() if not df.empty else 0
    except:
        return 0

def get_down_stock_count(trade_date: str) -> int:
    try:
        df = get_daily_data(trade_date)
        return (df['pct_chg'] < 0).sum() if not df.empty else 0
    except:
        return 0

def get_today_turnover(trade_date: str) -> float:
    """获取上证指数的成交额（亿元）"""
    try:
        df = safe_api_call(pro.index_daily, ts_code='000001.SH', trade_date=trade_date, fields='trade_date,vol,amount')
        if not df.empty and 'amount' in df.columns:
            return df['amount'].iloc[0] / 10000  # 换算为亿元
        return DEFAULT_TURNOVER
    except:
        return DEFAULT_TURNOVER

def get_avg_turnover_30d(trade_date: str) -> float:
    """获取上证指数近30日平均成交额（亿元）"""
    try:
        end = datetime.strptime(trade_date, '%Y%m%d')
        start = end - timedelta(days=40)
        df = safe_api_call(pro.index_daily, ts_code='000001.SH', start_date=start.strftime('%Y%m%d'), end_date=end.strftime('%Y%m%d'), fields='trade_date,amount')
        if not df.empty:
            return df['amount'].mean() / 10000
        return DEFAULT_AVG_TURNOVER
    except:
        return DEFAULT_AVG_TURNOVER


def get_limit_up_count(trade_date: str) -> int:
    """获取当日涨停股数量，带文件缓存"""
    if trade_date in _limit_up_cache:
        logger.info(f"[涨停统计] 使用缓存数据：{trade_date} 涨停股数量：{_limit_up_cache[trade_date]}")
        return _limit_up_cache[trade_date]

    try:
        df = safe_api_call(
            pro.limit_list_d,
            trade_date=trade_date,
            limit_type='U',
            fields='ts_code'
        )
        if df is None or df.empty:
            logger.warning(f"[涨停统计] {trade_date} 无涨停数据返回，缓存为0")
            _limit_up_cache[trade_date] = 0
            save_limit_up_cache()
            return 0

        count = len(df)
        _limit_up_cache[trade_date] = count
        save_limit_up_cache()
        logger.info(f"[涨停统计] {trade_date} 涨停股数量：{count}")
        return count

    except Exception as e:
        logger.error(f"[涨停统计] 获取涨停数据失败：{e}，缓存为0")
        _limit_up_cache[trade_date] = 0
        save_limit_up_cache()
        return 0


def save_limit_up_cache():
    with open(LIMIT_UP_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(_limit_up_cache, f, ensure_ascii=False, indent=2)

# ===== 新增：强制刷新市场情绪 =====
def refresh_market_sentiment() -> str:
    global _last_sentiment_date, _cached_sentiment, _limit_up_cache
    # 1) 清掉今日情绪缓存
    _last_sentiment_date = None
    _cached_sentiment = None

    # 2) 清掉当日涨停股文件缓存，让 get_limit_up_count 重新拉取
    today = datetime.today().strftime('%Y%m%d')
    trade_date = get_last_trade_date() if datetime.today().weekday() >= 5 else today
    if trade_date in _limit_up_cache:
        del _limit_up_cache[trade_date]
        save_limit_up_cache()

    # 3) 再次返回最新的市场情绪
    return get_market_sentiment_ui()

# ===== 市场情绪缓存机制 =====
_last_sentiment_date = None
_cached_sentiment = None

def get_market_sentiment_ui():
    global _last_sentiment_date, _cached_sentiment
    today = datetime.today().strftime('%Y%m%d')

    # 判断是否休市
    is_weekend = datetime.today().weekday() >= 5
    trade_date = get_last_trade_date() if is_weekend else today

    if _last_sentiment_date == today and _cached_sentiment:
        sentiment, description = _cached_sentiment
    else:
        sentiment, description = calculate_market_sentiment()
        _cached_sentiment = (sentiment, description)
        _last_sentiment_date = today

    # 格式化日期显示
    display_date = datetime.strptime(trade_date, '%Y%m%d').strftime('%Y-%m-%d')
    notice = f"📅 数据日期：{display_date}"
    if is_weekend:
        notice += " （今日休市，展示最近交易日数据）"

    return f"{notice}\n\n📊 当前市场情绪：**{sentiment}**\n\n{description}"


def calculate_market_sentiment() -> Tuple[str, str]:
    # 判断是否休市
    is_weekend = datetime.today().weekday() >= 5
    trade_date = get_last_trade_date() if is_weekend else datetime.today().strftime('%Y%m%d')

    if is_weekend:
        logger.info(f"[市场情绪] 今日休市，使用最近交易日数据：{trade_date}")
    else:
        logger.info(f"[市场情绪] 使用今日数据：{trade_date}")

    # 获取数据（这里你需要改造各个方法支持传入 trade_date）
    up_count = get_up_stock_count(trade_date)
    down_count = get_down_stock_count(trade_date)
    turnover = get_today_turnover(trade_date)
    avg_turnover = get_avg_turnover_30d(trade_date)
    limit_up_count = get_limit_up_count(trade_date)

    logger.info(f"[市场情绪] 指标 => 上涨: {up_count} | 下跌: {down_count} | 成交额: {turnover:.2f} 亿 | 30日均量: {avg_turnover:.2f} 亿 | 涨停数: {limit_up_count}")

    score = 0
    up_down_score = (up_count / (down_count + 1)) * 20
    turnover_score = (turnover / avg_turnover) * 30
    limit_up_score = min(limit_up_count, 50) * 1.5

    score = up_down_score + turnover_score + limit_up_score

    logger.info(f"[市场情绪] 评分 => 涨跌比: {up_down_score:.2f} | 成交额: {turnover_score:.2f} | 涨停: {limit_up_score:.2f} | 总分: {score:.2f}")

    if score > 100:
        sentiment = "乐观"
        description = "😄 当前市场活跃，题材轮动加快，适合短线操作"
    elif score < 60:
        sentiment = "悲观"
        description = "😟 市场低迷，注意控制风险，防守为主"
    else:
        sentiment = "震荡"
        description = "😐 市场观望情绪浓厚，精选优质标的"

    logger.info(f"[市场情绪] 最终判断：{sentiment}（{score:.2f} 分）")

    return sentiment, description

# ===== 创建Gradio界面 =====
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📈 AI量化选股系统 (V25.5.15)
    **功能**:
    - 使用Tushare获取当日数据（晚上8点左右更新完毕）            
    - 支持自然语言策略输入
    - 技术指标分析
    - 推荐历史记录管理
    - 个股详情查询
    """)

    # ===== 市场情绪展示 =====
    with gr.Row():
        sentiment_box = gr.Markdown(get_market_sentiment_ui())
        refresh_sentiment_btn = gr.Button("🔄 刷新市场情绪")
        refresh_sentiment_btn.click(fn=refresh_market_sentiment, outputs=sentiment_box)

    with gr.Row():
        with gr.Column(scale=3):
            # 选股分析结果窗口（保留）
            chatbot = gr.Chatbot(height=500, label="选股分析")
            
            # 输入框（移除或隐藏）
            # txt = gr.Textbox(label="输入选股策略", 
            #                  placeholder="例如: 找出MACD金叉且成交量放大的股票，如果你想直接获取智能推荐，可以输入"推荐股票"",
            #                  visible=False)  # 或者直接注释掉
            
        with gr.Column(scale=1):
            market_type = gr.CheckboxGroup(
                choices=list(MARKET_SECTORS.keys()),
                value=["主板"],
                label="选择市场/板块"
            )
            with gr.Accordion("高级选项", open=True):
                max_stocks = gr.Slider(
                    minimum=100,
                    maximum=5500,
                    step=100,
                    value=Config.MAX_STOCKS_TO_ANALYZE,
                    label="最大分析股票数量"
                )
                strategy_mode = gr.Radio(
                    choices=["稳健型", "激进型", "穿线型"], 
                    value="稳健型",
                    label="策略模式选择"
                )
            clear_btn = gr.Button("清除记录", variant="secondary")
            analyze_btn = gr.Button("开始分析", variant="primary")

    # ===== 推荐历史 Tab =====
    with gr.Tab("📊 推荐历史"):
        tracking_html = gr.HTML(value=get_tracking_html())
        refresh_btn = gr.Button("🔄 刷新推荐历史", variant="primary")
        refresh_btn.click(fn=get_tracking_html, outputs=tracking_html)

        with gr.Row():
            delete_code = gr.Textbox(label="输入要删除的股票代码", placeholder="例如: 000001 或 000001.SZ", scale=4)
            delete_btn = gr.Button("🗑️ 删除", variant="stop", scale=1)

        with gr.Row():
            clear_all_btn = gr.Button("💣 清空所有记录", variant="stop")

        confirm_clear = gr.Checkbox(label="确认清空所有记录", visible=False)
        status_msg = gr.Textbox(visible=False)
        delete_msg = gr.Textbox(visible=False)

        delete_btn.click(
            fn=lambda code: tracker.remove_stock(code),
            inputs=delete_code,
            outputs=delete_msg
        ).then(fn=lambda: "", outputs=delete_code).then(fn=get_tracking_html, outputs=tracking_html)

        def toggle_confirm_clear():
            return {"visible": True}, "请勾选确认框后再次点击清空按钮"

        clear_all_btn.click(fn=toggle_confirm_clear, outputs=[confirm_clear, status_msg])

        def handle_clear_confirmation(confirmed):
            if confirmed:
                tracker.clear_all_recommendations(confirmation=True)
                return {"visible": False}, "✅ 已清空所有记录", get_tracking_html()
            else:
                return {"visible": False}, "❌ 清空操作已取消", get_tracking_html()

        confirm_clear.change(
            fn=handle_clear_confirmation,
            inputs=confirm_clear,
            outputs=[confirm_clear, status_msg, tracking_html]
        )

    # ===== 个股查询 Tab =====
    with gr.Tab("🔍 个股查询"):
        stock_query = gr.Textbox(label="输入股票代码（如：000006或000006.SZ）")
        query_btn = gr.Button("查询")
        stock_output = gr.Textbox(label="查询结果", interactive=False)
        query_btn.click(fn=query_stock, inputs=stock_query, outputs=stock_output)

    history = gr.State([])

    # 修改分析按钮的处理函数，直接调用推荐股票
    def analyze_without_input(market_type, max_stocks, strategy_mode, history):
        return chat_interface("推荐股票", market_type, max_stocks, strategy_mode, history)

    # 绑定事件处理
    analyze_btn.click(
        analyze_without_input, 
        [market_type, max_stocks, strategy_mode, history], 
        [chatbot, history]
    )
    
    clear_btn.click(lambda: ([], []), None, [chatbot, history])



# ===== 启动服务 =====
if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
    
