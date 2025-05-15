import pandas as pd  
from config import Config  
import matplotlib.pyplot as plt
import gradio as gr
import os
import logging
import time
from pandas.tseries.holiday import get_calendar
from datetime import datetime
from datetime import datetime, timedelta
import numpy as np
from matplotlib import rcParams
import threading
import functools
from typing import List, Dict
import appy
import tushare as ts
from typing import Union
# 新增带时效性的缓存装饰器
from functools import lru_cache
from datetime import timedelta
import pandas_market_calendars as mcal
rcParams['font.family'] = 'SimHei'  # 设置中文字体为 SimHei（黑体）
rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

from appy import StockAnalyzer, analyze_stocks, STRATEGY_WEIGHTS
from appy import adjust_strategy_weights_by_market

BENCHMARK_INDEX = '000300.SH'

# 动态调整的基础风险敞口
BASE_PORTFOLIO_RISK = 0.3  # 基础风险敞口
VOLATILITY_LOOKBACK = 20  # 波动率计算回溯期

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 数据缓存
price_cache = {}
market_status_cache = {}

def get_dynamic_risk_exposure(market_status: str, trend_day_count: int = 0, momentum: float = 0.0) -> float:
    """根据市场状态动态计算风险敞口"""
    # 基础风险敞口映射
    risk_map = {
        "极端牛市": 0.95,  # 95%最大仓位
        "牛市": 0.85,      # 85%
        "温和牛市": 0.75,  # 75%
        "震荡市": 0.5,     # 50%
        "温和熊市": 0.4,   # 40%
        "熊市": 0.3,       # 30%
        "极端熊市": 0.2,   # 20%
        "高波动市": 0.35,  # 35%（基础值，将根据动量调整）
    }
    
    base_risk = risk_map.get(market_status, 0.5)
    
    # 特殊处理高波动市：根据动量（涨跌幅）动态调整
    if market_status == "高波动市":
        if momentum > 0.05:  # 动量大于5%，说明是上涨的高波动
            # 高波动+上涨时，提高风险敞口
            base_risk = 0.65  # 提高到65%
            if momentum > 0.1:  # 动量大于10%
                base_risk = 0.8  # 进一步提高到80%
        elif momentum < -0.05:  # 动量小于-5%，说明是下跌的高波动
            # 高波动+下跌时，降低风险敞口
            base_risk = 0.25  # 降到25%
    
    # 根据趋势持续天数调整（趋势越稳定，敞口可以越大）
    if market_status in ["极端牛市", "牛市", "温和牛市"]:
        # 牛市趋势持续越久，可以加大仓位
        trend_bonus = min(0.1, trend_day_count * 0.01)  # 每天增加1%，最多增加10%
        base_risk = min(0.95, base_risk + trend_bonus)
    elif market_status in ["熊市", "极端熊市"]:
        # 熊市趋势持续越久，继续降低仓位
        trend_penalty = min(0.1, trend_day_count * 0.005)  # 每天减少0.5%，最多减少10%
        base_risk = max(0.1, base_risk - trend_penalty)
    elif market_status == "高波动市" and momentum > 0:
        # 高波动市但上涨趋势，也可以根据持续天数适当加仓
        trend_bonus = min(0.05, trend_day_count * 0.005)  # 每天增加0.5%，最多增加5%
        base_risk = min(0.85, base_risk + trend_bonus)
    
    logger.info(f"📊 动态风险敞口: {market_status} -> {base_risk:.2%} (趋势{trend_day_count}天, 动量{momentum:.2%})")
    return base_risk

def get_dynamic_parameters(market_status: str, base_params: Dict, momentum: float = 0.0) -> Dict:
    """根据市场状态动态调整交易参数"""
    # 基础参数
    expected_return = base_params.get('expected_return', 5.0)
    stop_loss = base_params.get('stop_loss', -3.0)
    holding_days = base_params.get('holding_days', 10)
    
    # 动态调整规则
    if market_status in ["极端牛市", "牛市"]:
        # 牛市放宽参数
        expected_return *= 1.5  # 提高止盈点
        stop_loss *= 1.5        # 放宽止损
        holding_days *= 1.2     # 延长持仓时间
    elif market_status in ["温和牛市"]:
        # 温和牛市适度放宽
        expected_return *= 1.2
        stop_loss *= 1.2
        holding_days *= 1.1
    elif market_status in ["熊市", "极端熊市"]:
        # 熊市收紧参数
        expected_return *= 0.7  # 降低止盈点，快速获利了结
        stop_loss *= 0.8        # 收紧止损
        holding_days *= 0.8     # 缩短持仓时间
    elif market_status == "高波动市":
        # 高波动市根据动量方向调整
        if momentum > 0.05:  # 上涨的高波动
            # 参数向牛市靠拢
            expected_return *= 1.3
            stop_loss *= 1.3
            holding_days *= 1.1
        elif momentum < -0.05:  # 下跌的高波动
            # 参数向熊市靠拢
            expected_return *= 0.8
            stop_loss *= 0.7
            holding_days *= 0.9
        else:  # 震荡的高波动
            # 适度收紧止损，保持止盈
            stop_loss *= 0.8
            holding_days *= 0.95
    
    return {
        'expected_return': expected_return,
        'stop_loss': stop_loss,
        'holding_days': int(holding_days)
    }

def ttl_cache(maxsize=128, ttl=300):
    def decorator(func):
        @lru_cache(maxsize=maxsize)
        def wrapper(*args, **kwargs):
            value = func(*args, **kwargs)
            expiration = datetime.now() + timedelta(seconds=ttl)
            return (value, expiration)
            
        def wrapped(*args, **kwargs):
            value, expiration = wrapper(*args, **kwargs)
            if datetime.now() > expiration:
                wrapper.cache_clear()
                value, expiration = wrapper(*args, **kwargs)
            return value
        return wrapped
    return decorator

def format_ts_code(ts_code: str) -> str:
    """格式化股票代码，确保后缀正确"""
    if not ts_code.endswith(('.SH', '.SZ')):
        # 主板/科创板以6/9开头用.SH，创业板/中小板以0/3开头用.SZ
        suffix = '.SH' if ts_code.startswith(('6', '9')) else '.SZ'
        return f"{ts_code}{suffix}"
    return ts_code

def precheck_stock(ts_code: str, trade_date: str) -> bool:
    """修复点2：增强实时停牌检查"""
    # 统一代码格式为大写
    ts_code = ts_code.upper()
    pro = ts.pro_api(Config.TUSHARE_TOKEN)
    # 实时获取最新停牌数据（关键修复）
    current_suspend = pro.suspend_d(
        suspend_date=trade_date,
        fields="ts_code"
    )['ts_code'].str.upper().tolist()
    
    if ts_code in current_suspend:
        logger.warning(f"⏸️ 实时验证 {ts_code} 在 {trade_date} 停牌状态")
        return False
    
    # 原有上市日期检查
    list_info = StockAnalyzer.pro.stock_basic(ts_code=ts_code, fields='list_date')
    if not list_info.empty:
        list_date = list_info.iloc[0]['list_date']
        if datetime.strptime(list_date, '%Y%m%d') > datetime.strptime(trade_date, '%Y%m%d'):
            logger.warning(f"🆕 {ts_code} 在 {trade_date} 尚未上市")
            return False
    
    return True

def is_selection_day(current_date, frequency_mode, custom_days):
    """ 判断是否选股日 """
    if frequency_mode == "每日选股":
        return True
    elif frequency_mode == "每周两次":
        return current_date.weekday() in [0, 3]  # 周一和周四
    elif frequency_mode == "每隔N天":
        start_anchor = datetime.strptime("20240101", "%Y%m%d")
        return (current_date - start_anchor).days % custom_days == 0
    return False

def get_recent_or_market_avg_price(ts_code: str, trade_date: str) -> float:
    """尝试获取前一日价格或市场均值替代停牌股票的价格"""
    # 尝试获取前一日的收盘价
    recent_price = get_stock_price_single(ts_code, trade_date)
    if recent_price:
        return recent_price

    # 如果前一日的价格不可用，尝试获取市场均值
    market_avg_price = get_market_average_price(trade_date)
    if market_avg_price:
        logger.info(f"⚡ 使用市场均值 {market_avg_price} 代替停牌价格")
        return market_avg_price
    
    return None

def get_market_average_price(trade_date: str) -> float:
    """计算市场的均值价格"""
    try:
        # 获取所有股票的收盘价
        df = StockAnalyzer.pro.daily(
            ts_code='all', 
            trade_date=trade_date,
            fields='ts_code,close'
        )
        
        if df is not None and not df.empty:
            # 计算所有股票的均值（去除停牌或无效数据）
            valid_data = df[df['close'] > 0]
            if not valid_data.empty:
                market_avg_price = valid_data['close'].mean()
                return market_avg_price
            else:
                logger.warning(f"⚠️ {trade_date} 市场无有效股票数据，无法计算均值价格")
    except Exception as e:
        logger.error(f"❌ 获取市场均值价格出错: {str(e)}")
    
    return None

def get_market_status(trade_date):
    """获取市场状态（支持更多细分状态），并缓存波动率与动量"""
    if trade_date in market_status_cache:
        return market_status_cache[trade_date]

    trade_date_dt = datetime.strptime(trade_date, '%Y%m%d')
    end_date = trade_date_dt
    start_date = end_date - timedelta(days=60)

    try:
        df = StockAnalyzer.pro.index_daily(
            ts_code="000300.SH",
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d'),
            fields="trade_date,close"
        )

        if df is not None and not df.empty:
            df = df.sort_values('trade_date')
            closes = df['close'].astype(float)
            returns = closes.pct_change().dropna()

            pct_change = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]
            volatility = returns.std() * np.sqrt(252)
            momentum = closes[-20:].mean() / closes[:-20].mean() - 1 if len(closes) > 20 else 0.0
            ma20 = closes.rolling(20).mean().iloc[-1] if len(closes) >= 20 else closes.mean()

            # 状态判断逻辑（更细化）
            if volatility > 0.25:
                status = "高波动市"
            elif pct_change < -0.15:
                status = "极端熊市"
            elif pct_change < -0.08:
                status = "熊市"
            elif pct_change < -0.03:
                status = "温和熊市"
            elif pct_change > 0.15 and momentum > 0.1 and closes.iloc[-1] > ma20 * 1.08:
                status = "极端牛市"
            elif pct_change > 0.08 and momentum > 0.05:
                status = "牛市"
            elif pct_change > 0.03:
                status = "温和牛市"
            else:
                status = "震荡市"

            market_status_cache[trade_date] = status
            get_market_status.market_indicators_cache[trade_date] = {
                "volatility": volatility,
                "momentum": momentum
            }

            logger.info(f"📊 {trade_date} | 市场状态：{status} | 涨跌：{pct_change:.2%} | "
                        f"动量：{momentum:.2%} | 波动率：{volatility:.2%}")
            return status

    except Exception as e:
        logger.error(f"❌ {trade_date} 获取市场状态出错: {e}")

    market_status_cache[trade_date] = "震荡市"
    get_market_status.market_indicators_cache[trade_date] = {
        "volatility": 0.2,
        "momentum": 0.0
    }
    return "震荡市"

def get_stock_prices_batch(ts_codes: List[str], trade_date: str) -> Dict[str, float]:
    """优化后的批量获取逻辑"""
    valid_codes = [format_ts_code(code) for code in ts_codes if precheck_stock(code, trade_date)]
    if not valid_codes:
        return {}
    
    try:
        df = StockAnalyzer.pro.daily(
            ts_code=",".join(valid_codes),
            trade_date=trade_date,
            fields="ts_code,close"
        )
        if df is not None and not df.empty:
            # 过滤无效数据（如零成交量）
            df = df[df['close'] > 0]
            return df.set_index('ts_code')['close'].to_dict()
    except Exception as e:
        logger.error(f"批量获取失败，回退单股模式: {str(e)}")
    
    # 批量失败时逐个获取
    return {code: get_stock_price_single(code, trade_date) for code in valid_codes}

@ttl_cache(maxsize=128, ttl=3600) 
def get_stock_price_single(ts_code: str, trade_date: str) -> float:
    """单个股票价格获取（备用方案）"""
    key = (ts_code, trade_date)
    
    if key in price_cache:
        return price_cache[key]

    max_lookback, retries = 5, 3
    trade_date_obj = datetime.strptime(trade_date, '%Y%m%d')

    # 尝试获取停牌的股票的最近交易日价格
    for attempt in range(retries):
        for i in range(max_lookback):
            check_date_str = (trade_date_obj - timedelta(days=i)).strftime('%Y%m%d')
            try:
                df = StockAnalyzer.pro.daily(
                    ts_code=ts_code, 
                    start_date=check_date_str, 
                    end_date=check_date_str, 
                    fields="ts_code,trade_date,close"
                )
                if df is not None and not df.empty:
                    price = df.loc[df['ts_code'] == ts_code, 'close'].iloc[-1]
                    price_cache[key] = price
                    if i > 0:
                        logger.warning(f"⚠️ 使用 {check_date_str} 的收盘价作为 {ts_code} 在 {trade_date} 的近似价格：{price}")
                    return price
            except Exception as e:
                logger.error(f"获取 {ts_code} 在 {check_date_str} 的数据出错: {str(e)}")
        logger.warning(f"❌ 获取 {ts_code} 在 {trade_date} 的收盘价失败，重试 {attempt + 1}/{retries}")
        time.sleep(2)
    
    logger.error(f"❌ 无法获取 {ts_code} 在 {trade_date} 的收盘价，跳过！")
    return None

def get_market_trend_days(market_status: str, trade_date: Union[str, datetime], lookback_days: int = 60) -> int:
    """获取某个市场状态的连续维持天数"""
    if isinstance(trade_date, str):
        trade_date = datetime.strptime(trade_date, '%Y%m%d')

    trend_days = 0
    for i in range(lookback_days):
        date = trade_date - timedelta(days=i)
        date_str = date.strftime('%Y%m%d')

        status = market_status_cache.get(date_str)
        if status is None:
            status = get_market_status(date_str)

        if status == market_status:
            trend_days += 1
        else:
            break

    return trend_days

def buy_stock(account, stock, score, pct_change, risk_warnings, trade_date, batch_prices):
    market_status = get_market_status(trade_date)
    trend_day_count = get_market_trend_days(market_status, trade_date)
    
    # 获取市场状态指标
    vol_mom = get_market_status.market_indicators_cache.get(trade_date, {})
    momentum = vol_mom.get("momentum", 0.0)
    
    # 获取动态风险敞口
    dynamic_risk_cap = get_dynamic_risk_exposure(market_status, trend_day_count, momentum)
    
    ma5, atr = get_ma5_and_atr(stock[1], trade_date)
    buy_price = batch_prices.get(stock[1])
    if not buy_price and precheck_stock(stock[1], trade_date):
        buy_price = get_stock_price_single(stock[1], trade_date)
    if not buy_price:
        logger.warning(f"❌ 无法获取 {stock[1]} 的价格，跳过买入")
        return

    logger.debug(f"📈 当前市场趋势天数: {trend_day_count}")

    total_position_value = sum(
        pos['quantity'] * batch_prices.get(pos['ts_code'], 0)
        for pos in account['positions'] if not pos.get('is_short')
    )
    current_total_position_pct = total_position_value / account['equity']

    position_pct = calculate_position(
        stock_code=stock[1],
        score=score,
        pct_change=pct_change,
        risk_warnings=risk_warnings,
        market_status=market_status,
        strategy_mode=account.get("strategy_mode", "稳健型"),
        trend_day_count=trend_day_count,
        portfolio_risk_cap=dynamic_risk_cap  # 使用动态风险敞口
    )
    if position_pct == 0:
        return

    remaining_risk_pct = dynamic_risk_cap - current_total_position_pct
    if remaining_risk_pct <= 0:
        logger.info(f"🛑 组合已满仓（{current_total_position_pct:.2%}），跳过 {stock[1]} 并缓存到待买列表")
        # 方法2：缓存因满仓而被跳过的优质股票
        account.setdefault('pending_buys', []).append({
            'stock': stock,
            'score': score,
            'pct_change': pct_change,
            'risk_warnings': risk_warnings,
            'trade_date': trade_date
        })
        logger.info(f"📌 缓存 {stock[1]} 到 pending_buys")
        return

    if position_pct > remaining_risk_pct:
        logger.info(f"⚠️ 本票仓位超出组合限制，缩减为 {remaining_risk_pct:.2%}")
        position_pct = remaining_risk_pct

    allocation_amount = account['equity'] * position_pct
    if score > 150 and trend_day_count > 5:
        buy_amount = allocation_amount
    else:
        buy_amount = allocation_amount / 3

    stock_quantity = int(buy_amount // buy_price)
    if stock_quantity > 0:
        cost = stock_quantity * buy_price
        account['equity'] -= cost
        account['positions'].append({
            'ts_code': stock[1], 'buy_price': buy_price, 'quantity': stock_quantity,
            'buy_date': stock[0], 'days_held': 0, 'max_profit_pct': 0, 'buy_count': 1,
            'target_price': buy_price * (1 + account['expected_return'] / 100),
            'ma5': ma5, 'atr': atr
        })
        account['buy_records'].append({
            'ts_code': stock[1], 'buy_price': buy_price, 'quantity': stock_quantity,
            'buy_date': stock[0], 'position_pct': position_pct, 'allocated_amount': cost,
            'score': score, 'market_status': market_status, 'ma5': ma5,
            'atr': atr, 'trend_day_count': trend_day_count,
            'dynamic_risk_cap': dynamic_risk_cap  # 记录当时的风险敞口
        })
        if cost > 0.1 and position_pct < 0.03:
            logger.info(f"🛑 小仓位高成本跳过  仓位:{position_pct:.2%}")
            return
        logger.info(f"✅ 买入 {stock[1]} | 数量：{stock_quantity} | 仓位：{position_pct*100:.2f}% | 当前净值：{account['equity']:.2f} | 动态风险上限：{dynamic_risk_cap:.2%}")

def calculate_position(stock_code: str, score: float, pct_change: float = 0.0,
                       risk_warnings: List[str] = None, market_status: str = "震荡市",
                       strategy_mode: str = "稳健型", trend_day_count: int = 0,
                       portfolio_risk_cap: float = 1.0) -> float:
    """根据评分 × 动态 multiplier × 市场状态 × 策略模式 × 风险上限计算仓位"""

    if risk_warnings is None:
        risk_warnings = []

    # 获取动态风险敞口
    dynamic_risk_cap = get_dynamic_risk_exposure(market_status, trend_day_count)
    portfolio_risk_cap = min(portfolio_risk_cap, dynamic_risk_cap)

    # 1. 波动因子
    volatility_factor_map = {
        "极端牛市": 1.5, "牛市": 1.3, "温和牛市": 1.2,
        "震荡市": 0.9, "温和熊市": 0.8, "熊市": 0.7,
        "极端熊市": 0.5, "高波动市": 0.7,
    }
    volatility_factor = volatility_factor_map.get(market_status, 1.0)

    # 2. 评分修正
    score_base_map = {
        "极端牛市": 10, "牛市": 5, "温和牛市": 2,
        "温和熊市": -2, "熊市": -4, "极端熊市": -10,
        "高波动市": -3, "震荡市": 0
    }
    adjusted_score = score + score_base_map.get(market_status, 0)

    # 3. multiplier
    base_multiplier = {
        "极端牛市": 1.8, "牛市": 1.4, "温和牛市": 1.2,
        "震荡市": 1.0, "温和熊市": 0.8, "熊市": 0.6,
        "极端熊市": 0.4, "高波动市": 0.9
    }
    multiplier_growth = {
        "极端牛市": 0.05, "牛市": 0.03,
        "温和牛市": 0.02, "震荡市": 0.01
    }
    multiplier_cap = {
        "极端牛市": 2.5, "牛市": 1.8, "温和牛市": 1.5,
        "震荡市": 1.3, "温和熊市": 1.1, "熊市": 1.0,
        "极端熊市": 0.8, "高波动市": 1.0
    }

    base = base_multiplier.get(market_status, 1.0)
    growth = multiplier_growth.get(market_status, 0.0)
    cap = multiplier_cap.get(market_status, 1.0)
    multiplier = min(base + growth * trend_day_count, cap)

    if abs(pct_change) > 8:
        penalty = np.interp(abs(pct_change), [8, 12], [0.9, 0.7])
        multiplier *= penalty

    lower_bound, upper_bound = 100, 240
    norm_score = np.clip((adjusted_score - lower_bound) / (upper_bound - lower_bound), 0, 1)

    # 添加穿线型策略的支持
    if strategy_mode == "稳健型":
        base_pos = 0.06 + 0.25 * (norm_score ** 1.3)
        strategy_cap = 0.35
    elif strategy_mode == "激进型":
        base_pos = 0.08 + 0.35 * (norm_score ** 1.4)
        strategy_cap = 0.45
    elif strategy_mode == "穿线型":  # 新增穿线型策略
        base_pos = 0.08 + 0.32 * (norm_score ** 1.2)
        strategy_cap = 0.40
    else:
        logger.warning(f"⚠️ 未知策略模式 {strategy_mode}，返回 0 仓位")
        return 0.0

    # 将 cap 乘以组合风险限制（如 MAX_PORTFOLIO_RISK）
    max_cap = strategy_cap * portfolio_risk_cap
    position = min(base_pos * multiplier, max_cap)

    # 穿线型策略特殊的涨幅限制
    if strategy_mode == "穿线型":
        if abs(pct_change) > 12.0:
            logger.info(f"❌ 超过穿线型波动限制，跳过建仓：{pct_change:.2f}%")
            return 0.0
    elif strategy_mode == "稳健型":
        if abs(pct_change) > 12.8:
            logger.info(f"❌ 超过稳健型波动限制，跳过建仓：{pct_change:.2f}%")
            return 0.0
    elif strategy_mode == "激进型":
        if abs(pct_change) > 14.9:
            logger.info(f"❌ 超过激进型波动限制，跳过建仓：{pct_change:.2f}%")
            return 0.0

    final_position = position * volatility_factor

    logger.info(
        f"📌 仓位计算 | {stock_code} | 策略:{strategy_mode} | 状态:{market_status} | trend_day:{trend_day_count}\n"
        f"▫️ 原始评分: {score:.1f} → 调整后: {adjusted_score:.1f} | 标准化分数: {norm_score:.2f}\n"
        f"▫️ multiplier: {multiplier:.2f} | base_pos: {base_pos:.2%} | cap: {strategy_cap:.2f} × 风险限:{portfolio_risk_cap:.2f} = {max_cap:.2%}\n"
        f"▫️ volatility_factor: {volatility_factor:.2f} | 最终仓位: {final_position:.2%}"
    )

    return final_position

def generate_strategy_weights_by_market(market_status: str, volatility: float = 0.2, momentum: float = 0.0) -> Dict[str, float]:
    from appy import STRATEGY_TYPE_WEIGHTS
    base = STRATEGY_TYPE_WEIGHTS.copy()

    rules = {
        "极端熊市": {"趋势型": 0.9, "反转型": 1.2, "市场中性型": 1.2, "风险型": 1.3},
        "熊市": {"趋势型": 0.95, "动量型": 0.9, "反转型": 1.1, "市场中性型": 1.1, "风险型": 1.1},
        "温和熊市": {"趋势型": 1.0, "反转型": 1.05, "市场中性型": 1.0, "风险型": 1.0},
        "震荡市": {"趋势型": 1.05, "动量型": 1.05, "反转型": 1.05},
        "温和牛市": {"趋势型": 1.1, "动量型": 1.15},  
        "牛市": {"趋势型": 1.15, "动量型": 1.2}, 
        "极端牛市": {"趋势型": 1.2, "动量型": 1.25},
    }
    if market_status in rules:
        for k, v in rules[market_status].items():
            base[k] = base.get(k, 1.0) * v
    if market_status == "极端牛市":
        base.update({"趋势型":2.0, "动量型":2.5, "反转型":0.5}) 
    elif market_status == "熊市":
        base.update({"反转型":1.8, "市场中性型":1.5, "趋势型":0.6})

    # 波动率调节：中性型 ↑，风险型 ↓
    v_factor = np.interp(volatility, [0.1, 0.4], [0.9, 1.0])
    base["市场中性型"] = np.clip(base.get("市场中性型", 1.0) * v_factor, 0.9, 1.5)
    base["风险型"] = np.clip(base.get("风险型", -1.0) * (1.5 - 0.6 * v_factor), -2.0, 0.0)

    # 动量加权
    bonus = momentum * max(0.4 - 0.1 * abs(momentum), 0.2)
    base["动量型"] = np.clip(base.get("动量型", 1.0) + bonus, 0.6, 2.0)
    base["趋势型"] = np.clip(base.get("趋势型", 1.0) + bonus * 0.6, 0.5, 1.8)

    # 最终裁剪
    limits = {
        "趋势型": (0.7, 1.3),
        "动量型": (0.7, 1.3),
        "反转型": (0.7, 1.3),
        "市场中性型": (0.8, 1.3),
        "风险型": (-3.0, 0.0),
        "穿线型": (0.8, 1.3)  # 添加穿线型的限制
    }
    for key, (low, high) in limits.items():
        base[key] = np.clip(base.get(key, 1.0), low, high)

    return base

def short_stock(account, stock, score, pct_change, risk_warnings, trade_date, batch_prices): 
    market_status = get_market_status(trade_date)
    trend_day_count = get_market_trend_days(market_status, trade_date)
    
    # 获取动态风险敞口
    dynamic_risk_cap = get_dynamic_risk_exposure(market_status, trend_day_count)
    
    ma5, atr = get_ma5_and_atr(stock[1], trade_date)

    position_pct = calculate_position(
        stock[1], score, pct_change, risk_warnings, market_status,
        strategy_mode="激进型" if account.get('allow_short') else "稳健型",
        trend_day_count=trend_day_count,
        portfolio_risk_cap=dynamic_risk_cap
    ) * 0.8
    if pct_change < -5 and market_status in ["熊市", "高波动市"]: 
        position_pct *= 1.5
    if position_pct <= 0:
        return

    current_risk = (
        sum(pos['quantity'] * batch_prices.get(pos['ts_code'], 0) for pos in account['positions'] if not pos.get('is_short')) 
        + sum(pos['quantity'] * pos['short_price'] * 1.2 for pos in account['positions'] if pos.get('is_short'))
    ) / account['equity']
    if current_risk + (position_pct * 0.8) > dynamic_risk_cap:
        logger.info("空头开仓风险超限，跳过")
        return

    if not account.get('allow_short', True):
        return

    short_price = batch_prices.get(stock[1])
    if not short_price:
        logger.warning(f"❌ 无法获取 {stock[1]} 的价格，跳过做空")
        return

    MARGIN_RATIO = 1.1
    short_amount = account['equity'] * position_pct
    short_quantity = int((short_amount * MARGIN_RATIO) // short_price)
    if short_quantity > 0:
        account['equity'] -= short_quantity * short_price * MARGIN_RATIO
        account['positions'].append({
            'ts_code': stock[1], 'short_price': short_price, 'quantity': short_quantity,
            'short_date': trade_date, 'days_held': 0, 'max_profit_pct': 0,
            'is_short': True, 'atr': atr,
            'initial_margin': short_quantity * short_price * MARGIN_RATIO
        })
        account['short_records'].append({
            'ts_code': stock[1], 'short_price': short_price, 'quantity': short_quantity,
            'short_date': trade_date, 'position_pct': position_pct,
            'score': score, 'market_status': market_status
        })
        logger.info(f"🟦 做空 {stock[1]} | 数量：{short_quantity} | 仓位：{position_pct * 100:.2f}%")

def get_ma5_and_atr(ts_code: str, trade_date: str, atr_period=14):
    """获取MA5和ATR"""
    try:
        # 获取过去20天的历史数据来计算MA5和ATR
        df = StockAnalyzer.pro.daily(
            ts_code=ts_code,
            start_date=(datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=20)).strftime('%Y%m%d'),
            end_date=trade_date,
            fields="ts_code,trade_date,close,high,low"
        )
        if df.empty:
            logger.warning(f"⚠️ {ts_code} 无法获取足够的历史数据来计算MA5和ATR")
            return None, None
        
        # 计算MA5（5日均线）
        df['ma5'] = df['close'].rolling(window=5).mean()
        ma5 = df['ma5'].iloc[-1]  # 获取最后一个MA5值

        # 计算ATR（14日周期）
        df['tr'] = df['high'] - df['low']
        df['tr'] = df[['tr', 'high']].max(axis=1) - df['low']
        df['atr'] = df['tr'].rolling(window=atr_period).mean()
        atr = df['atr'].iloc[-1]  # 获取最后一个ATR值
        
        return ma5, atr

    except Exception as e:
        logger.error(f"❌ 获取 {ts_code} 的 MA5 和 ATR 数据出错: {str(e)}")
        return None, None

def calculate_risk_metrics(equity_curve, initial_equity, dates):
    """计算风险指标"""
    if not equity_curve or len(equity_curve) <= 1:
        return 0, 0, 0, 0

    equity_series = pd.Series(equity_curve)

    # 计算累计收益率
    total_return = (equity_series.iloc[-1] - initial_equity) / initial_equity

    # 计算年化收益率
    # 计算实际交易日天数（过滤掉周末）
    trading_days = len([d for d in dates if d.weekday() < 5])  # 过滤掉周末的交易日
    if trading_days > 0:
        years = trading_days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    else:
        annualized_return = 0

    # 计算最大回撤
    peak = equity_series.cummax()
    drawdown = (peak - equity_series) / peak
    max_drawdown = drawdown.max()

    # 计算夏普比率 (使用无风险利率2%)
    daily_returns = equity_series.pct_change().dropna()
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() - 0.02 / 252) / daily_returns.std() * np.sqrt(252)
    else:
        sharpe_ratio = 0

    return total_return * 100, annualized_return * 100, max_drawdown * 100, sharpe_ratio

def should_sell(position, current_price, account):
    """优化版卖出逻辑：支持多空头寸"""
    TRAILING_STOP_PCT = 4.0  # 固定回撤阈值

    if position.get('is_short'):
        # 空头平仓逻辑
        profit_pct = (position['short_price'] - current_price) / position['short_price'] * 100
        position['days_held'] += 1
        position['max_profit_pct'] = max(position.get('max_profit_pct', profit_pct), profit_pct)
        
        # 空头止损止盈（比多头更严格）
        if profit_pct <= -(account['stop_loss'] + position['atr']/position['short_price']) * 100:
            logger.info(f"🛑 空头止损 {position['ts_code']} | 盈亏: {profit_pct:.2f}%")
            return "空头止损"
        elif profit_pct >= account['expected_return'] * 0.8:
            logger.info(f"🟩 空头止盈 {position['ts_code']} | 盈亏: {profit_pct:.2f}%")
            return "空头止盈"
        elif position['days_held'] >= account['holding_days']:
            logger.info(f"⏳ 空头持仓到期 {position['ts_code']} | 持有天数: {position['days_held']}")
            return "空头持仓到期"
  
    else:
        # 多头平仓逻辑
        buy_price = position.get('buy_price', 0)
        if buy_price == 0:
            logger.error(f"⚠️ 检测到异常买入价格 {position['ts_code']}")
            return "异常价格平仓"
            
        profit_pct = (current_price - buy_price) / buy_price * 100
        position['days_held'] += 1
        position['max_profit_pct'] = max(position.get('max_profit_pct', profit_pct), profit_pct)
        
        # ============== 关键优化点 ==============
        # 1. 放宽保护期止损阈值 (通过参数调整)
        if position['days_held'] < 3:
            if profit_pct <= account['stop_loss'] * 1.8:
                logger.info(f"⚠️ 保护期内强制止损 {position['ts_code']} | 亏损:{profit_pct:.1f}%")
                return "保护期止损"
            logger.debug(f"🔒 {position['ts_code']} 持仓保护期 | 持有{position['days_held']}天")
            return None

        # 2. 动态参数计算优化 (通过expected_return参数调整)
        dynamic_return = account['expected_return'] * (1 + (position['days_held']**0.5)/25)  # 分母从30→25加速增长
        
        # 3. 波动跟踪止损优化 (保持代码不变，通过参数影响)
        atr_factor = position.get('atr', 0) / buy_price * 100
        trailing_stop = max(TRAILING_STOP_PCT, atr_factor * 1.5)

        # 4. 趋势增强逻辑 (新增MA20校验)
        if current_price > position.get('ma20', 0) and position.get('ma20', 0) > 0:
            dynamic_return *= 1.2
            trailing_stop *= 1.2
        else:
            dynamic_return *= 0.9  # 趋势不符合时降低预期

        # ============== 卖出条件优先级优化 ==============
        # 强制止损条件（阈值通过stop_loss参数控制）
        if profit_pct <= account['stop_loss'] * 1.5:  # stop_loss=-5时实际触发线为-7.5%
            logger.info(f"📉 强制止损 {position['ts_code']} | 亏损:{profit_pct:.1f}%")
            return "强制止损"
            
        # 加速止盈条件（保持代码不变，依赖3d_gain数据）
        if position.get('3d_gain', 0) > 20 and profit_pct > 15:
            logger.info(f"🚀 加速止盈 {position['ts_code']} | 三日涨幅:{position['3d_gain']}%")
            return "加速止盈"
            
        # 动态止盈条件（通过expected_return参数调整触发率）
        if profit_pct >= dynamic_return * 1.5:
            logger.info(f"🎯🔥 超额止盈 {position['ts_code']} | 收益:{profit_pct:.1f}%")
            return "超额止盈"
            
        if profit_pct >= dynamic_return:
            logger.info(f"🎯 基础止盈 {position['ts_code']} | 收益:{profit_pct:.1f}%")
            return "基础止盈"
        
        # 波动止损条件（参数影响ATR计算）
        current_drawdown = position['max_profit_pct'] - profit_pct
        if current_drawdown >= trailing_stop:
            logger.info(f"📉 波动止损 {position['ts_code']} | 回撤:{current_drawdown:.1f}%≥{trailing_stop:.1f}%")
            return "波动止损"
            
        # 时间止损条件（参数调整阈值）
        if position['days_held'] >= int(account['holding_days']*0.8):
            if profit_pct < max(2.0, account['expected_return']*0.3):
                logger.info(f"⌛ 时间止损 {position['ts_code']} | 持有:{position['days_held']}天 收益:{profit_pct:.1f}%")
                return "时间止损"
                
        # 保本条件（增加波动过滤）
        if profit_pct <= 0 and position['days_held'] > 5 and position['atr']/buy_price < 0.03:
            logger.info(f"🛡️ 保本退出 {position['ts_code']} | 持有{position['days_held']}天")
            return "保本退出"
    
    return None

market_status_cache = {}
get_market_status.market_indicators_cache = {}

def run_backtest(start_date, end_date, holding_days, strategy_modes, markets,  
                 expected_return, stop_loss, cost, benchmark=None, max_stock_num=200, 
                 frequency_mode="每周两次", custom_days=5, initial_equity=100000.0):
    """ 支持多空对冲的回测主逻辑（修正净值计算版本） """
    from pandas.tseries.holiday import get_calendar
    from appy import StockAnalyzer, analyze_stocks, STRATEGY_WEIGHTS

    used_benchmark = benchmark if benchmark else BENCHMARK_INDEX
    logger.info(f"🚀 回测开始：{start_date} ~ {end_date} | 策略：{strategy_modes} | 基准：{used_benchmark}")
   
    # === 获取基准指数数据 ===
    benchmark_data = StockAnalyzer.pro.index_daily(
        ts_code=used_benchmark, 
        start_date=start_date, 
        end_date=end_date, 
        fields="ts_code,trade_date,close"
    )
    if benchmark_data is None or benchmark_data.empty:
        logger.error(f"❌ 无法获取基准指数({used_benchmark})历史数据，回测中止")
        raise ValueError(f"无法获取基准指数({used_benchmark})数据")
    nyse = mcal.get_calendar('XSHG')
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    valid_dates = schedule.index.to_pydatetime().tolist()
    benchmark_data = benchmark_data.sort_values('trade_date')
    benchmark_data['trade_date'] = pd.to_datetime(benchmark_data['trade_date'])
    benchmark_data = benchmark_data.set_index('trade_date')
    dates = valid_dates  # 使用实际交易日历

    # === 初始化多空账户（添加穿线型策略账户） ===
    accounts = {
        mode: {
            'equity': initial_equity,
            'equity_curve': [initial_equity],
            'positions': [],
            'buy_records': [],
            'short_records': [],
            'pending_buys': [],               # 方法2：待买列表
            'expected_return': expected_return,
            'stop_loss': stop_loss,
            'holding_days': holding_days,
            'allow_short': True if mode == "激进型" else False,
            'short_position_ratio': 0.3,
            'strategy_mode': mode,  # 添加策略模式标识
            'used_margin': 0,
            'MARGIN_RATIO': 1.2,
            'base_params': {  # 保存基础参数
                'expected_return': expected_return,
                'stop_loss': stop_loss,
                'holding_days': holding_days
            }
        }
        for mode in strategy_modes
    }

    if isinstance(markets, dict) and "weights" in markets:
        mode_weights_map = markets["weights"]
    else:
        mode_weights_map = {}

    current_idx = 0
    while current_idx < len(dates):
        current_date = dates[current_idx]
        trade_date_str = current_date.strftime('%Y%m%d')

        appy.IS_BACKTEST = True
        appy.CURRENT_TRADE_DATE = trade_date_str

        selection_today = is_selection_day(
            current_date=current_date,
            frequency_mode=frequency_mode,
            custom_days=custom_days,
        )

        for mode in strategy_modes:
            account = accounts[mode]

            # === 获取策略权重（每日动态） === 
            market_status = get_market_status(trade_date_str)
            trend_day_count = get_market_trend_days(market_status, trade_date_str)
            
          
            
            vol_mom = get_market_status.market_indicators_cache.get(trade_date_str, {})
            volatility = vol_mom.get("volatility", 0.2)
            momentum = vol_mom.get("momentum", 0.0)
            

              # 动态调整账户参数
            dynamic_params = get_dynamic_parameters(market_status, account['base_params'], momentum)
            account.update(dynamic_params)
            type_weights = generate_strategy_weights_by_market(
                market_status, volatility=volatility, momentum=momentum
            )
            
            logger.info(f"📡 动态策略权重 [{mode}]：{type_weights}")
            logger.info(f"📊 动态参数调整 [{mode}]：止盈{account['expected_return']:.1f}% | 止损{account['stop_loss']:.1f}% | 持仓天数{account['holding_days']}")

            # === 持仓管理 ===
            positions_to_remove = []
            current_prices = get_stock_prices_batch(
                [pos['ts_code'] for pos in account['positions']],
                trade_date_str
            ) if account['positions'] else {}

            for pos in account['positions']:
                pos['days_held'] += 1
                current_price = current_prices.get(pos['ts_code'])
                if not current_price:
                    current_price = get_recent_or_market_avg_price(pos['ts_code'], trade_date_str)
                    if not current_price:
                        current_price = pos.get('buy_price') or pos.get('short_price') or 0
                        logger.warning(f"⚡ 使用持仓价格替代 {pos['ts_code']}: {current_price}")

                sell_reason = should_sell(pos, current_price, account)
                if sell_reason:
                    if pos.get('is_short'):
                        profit = (pos['short_price'] - current_price) * pos['quantity']
                        account['equity'] += pos.get('initial_margin', 0)
                        account['equity'] += profit * (1 - cost / 100)
                        logger.info(f"💸 平空 {pos['ts_code']} | 数量:{pos['quantity']} 盈亏:{profit:.2f}")
                    else:
                        sell_amount = current_price * pos['quantity'] * (1 - cost / 100)
                        account['equity'] += sell_amount
                        logger.info(f"💸 卖出 {pos['ts_code']} | 数量:{pos['quantity']} 金额:{sell_amount:.2f}")
                    positions_to_remove.append(pos)

            for pos in positions_to_remove:
                account['positions'].remove(pos)

            # 方法2：卖出后尝试 pending_buys 补建仓
            if account['pending_buys']:
                codes = [pb['stock'][1] for pb in account['pending_buys']]
                pending_prices = get_stock_prices_batch(codes, trade_date_str) or {}
                for pb in account['pending_buys'][:]:
                    buy_stock(
                        account,
                        pb['stock'], pb['score'], pb['pct_change'],
                        pb['risk_warnings'],
                        trade_date_str,    # 当前交易日
                        pending_prices
                    )
                    account['pending_buys'].remove(pb)

            # === 选股日执行买入 ===
            if selection_today:
                # 方法1：选股日前主动清理即将到期的持仓
                early_threshold = int(account['holding_days'] * 0.8)
                for pos in account['positions'][:]:
                    if not pos.get('is_short') and pos['days_held'] >= early_threshold:
                        current_price = current_prices.get(pos['ts_code'], pos.get('buy_price'))
                        sell_amount = current_price * pos['quantity'] * (1 - cost / 100)
                        account['equity'] += sell_amount
                        logger.info(
                            f"🔖 提前卖出到期持仓 {pos['ts_code']} | 持仓天数:{pos['days_held']} | 金额:{sell_amount:.2f}"
                        )
                        account['positions'].remove(pos)

                held_ts_codes = {pos['ts_code'] for pos in account['positions']}
                stock_list = StockAnalyzer.get_stock_list(
                    tuple(markets),
                    max_count=max_stock_num,
                    strategy_mode=mode,
                    trade_date=trade_date_str
                )

                if stock_list:
                    recs = analyze_stocks(
                        stock_list_with_turnover=stock_list,
                        strategies=list(STRATEGY_WEIGHTS.keys()),
                        custom_weights=type_weights,
                        max_stocks=max_stock_num,
                        strategy_mode=mode,
                        trade_date=trade_date_str,
                        export_watchlist=False
                    )

                    recommend_codes = [s[1] for s in recs if s[1] not in held_ts_codes]
                    batch_rec_prices = get_stock_prices_batch(recommend_codes, trade_date_str) or {}

                    # 多头开仓
                    if mode == "穿线型":
                        # 穿线型策略限制买入数量为10支
                        long_candidates = sorted(recs, key=lambda x: x[0], reverse=True)[:10]
                    else:
                        long_candidates = sorted(recs, key=lambda x: x[0], reverse=True)[:max_stock_num // 2]
                    
                    for stock in long_candidates:
                        if stock[1] not in held_ts_codes:
                            buy_stock(account, stock, stock[0], 0, [], trade_date_str, batch_rec_prices)

                    # 空头开仓 (仅激进型策略)
                    if account['allow_short']:
                        short_candidates = [s for s in recs if s[0] < 115]
                        if short_candidates:
                            short_candidates = sorted(short_candidates, key=lambda x: x[0])[:max_stock_num // 4]
                        for stock in short_candidates:
                            if stock[1] not in held_ts_codes:
                                short_stock(account, stock, stock[0], 0, [], trade_date_str, batch_rec_prices)

            # === 净值更新 ===
            total_long = sum(
                pos['quantity'] * current_prices.get(pos['ts_code'], pos.get('buy_price', 0))
                for pos in account['positions'] if not pos.get('is_short')
            )
            total_short_liability = sum(
                pos['quantity'] * (current_prices.get(pos['ts_code'], pos.get('short_price', 0)) - pos['short_price'])
                for pos in account['positions'] if pos.get('is_short')
            )
            net_value = account['equity'] + total_long - total_short_liability
            account['equity_curve'].append(net_value)

        current_idx += 1
        
    calendar = mcal.get_calendar('XSHG')
    last_date = dates[-1]
    extended_dates = calendar.valid_days(
        start_date=last_date + pd.Timedelta(days=1),
        end_date=last_date + pd.Timedelta(days=holding_days*2)
    ).tolist()

    for mode in strategy_modes:
        account = accounts[mode]
        if not account['positions']:
            continue

        current_extended_idx = 0
        while current_extended_idx < len(extended_dates) and account['positions']:
            current_date = extended_dates[current_extended_idx]
            trade_date_str = current_date.strftime('%Y%m%d')
            
            positions_to_remove = []
            batch_prices = get_stock_prices_batch(
                [pos['ts_code'] for pos in account['positions']],
                trade_date_str
            )
            
            for pos in account['positions']:
                pos['days_held'] += 1
                current_price = batch_prices.get(pos['ts_code'])
                if not current_price:
                    current_price = get_recent_or_market_avg_price(pos['ts_code'], trade_date_str)
                    if not current_price:
                        current_price = pos.get('buy_price') or pos.get('short_price') or 0

                force_close = pos['days_held'] > account['holding_days'] * 2
                force_close |= current_extended_idx == len(extended_dates)-1
                sell_reason = should_sell(pos, current_price, account) or ("强制平仓" if force_close else None)
                
                if sell_reason:
                    if pos.get('is_short'):
                        profit = (pos['short_price'] - current_price) * pos['quantity']
                        account['equity'] += pos['initial_margin']
                        account['equity'] += profit * (1 - cost/100)
                        logger.warning(f"💥 强制平空 {pos['ts_code']} | 数量:{pos['quantity']} 盈亏:{profit:.2f}")
                    else:
                        sell_amount = current_price * pos['quantity'] * (1 - cost/100)
                        account['equity'] += sell_amount
                        logger.warning(f"💥 强制卖出 {pos['ts_code']} | 数量:{pos['quantity']} 金额:{sell_amount:.2f}")
                    positions_to_remove.append(pos)

            for pos in positions_to_remove:
                account['positions'].remove(pos)

            # 方法2：延长期卖出后尝试 pending_buys 补建仓
            if account['pending_buys']:
                codes = [pb['stock'][1] for pb in account['pending_buys']]
                pending_prices = get_stock_prices_batch(codes, trade_date_str) or {}
                for pb in account['pending_buys'][:]:
                    buy_stock(
                        account,
                        pb['stock'], pb['score'], pb['pct_change'],
                        pb['risk_warnings'],
                        trade_date_str,    # 当前交易日
                        pending_prices
                    )
                    account['pending_buys'].remove(pb)

            total_long = sum(
                p['quantity'] * batch_prices.get(p['ts_code'], 0)
                for p in account['positions'] if not p.get('is_short')
            )
            total_short_liability = sum(
                p['quantity'] * (batch_prices.get(p['ts_code'], 0) - p['short_price'])
                for p in account['positions'] if p.get('is_short')
            )
            net_value = account['equity'] + total_long - total_short_liability
            account['equity_curve'].append(net_value)
            
            current_extended_idx += 1

    for mode in strategy_modes:
        account = accounts[mode]
        if account['positions']:
            logger.error(f"⚠️ 仍有{len(account['positions'])}只持仓未能平仓，强制清空")
            for pos in account['positions']:
                current_price = get_recent_or_market_avg_price(pos['ts_code'], extended_dates[-1].strftime('%Y%m%d')) or 0
                if pos.get('is_short'):
                    profit = (pos['short_price'] - current_price) * pos['quantity']
                    account['equity'] += pos['initial_margin'] + profit * (1 - cost/100)
                else:
                    sell_amount = current_price * pos['quantity'] * (1 - cost/100)
                    account['equity'] += sell_amount
            account['positions'].clear()

    bench_start = benchmark_data['close'].iloc[0]
    benchmark_curve = [initial_equity * (benchmark_data.loc[d, 'close'] / bench_start) for d in dates]

    fig, ax = plt.subplots(figsize=(12, 6))
    for mode in strategy_modes:
        account = accounts[mode]
        equity_curve = account['equity_curve'][:len(dates)]
        ax.plot(dates, equity_curve, label=f'{mode}策略', linewidth=1.5)
        ax.scatter(dates[-1], equity_curve[-1], s=50, label=f'{mode}终值({equity_curve[-1]:.2f})')

    ax.plot(dates, benchmark_curve, label=f'基准({used_benchmark})', linestyle=':', color='gray')
    ax.set_title(f"净值曲线对比 {start_date}~{end_date}")
    ax.legend(loc='upper left')
    plt.xticks(rotation=35, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    stats = f"📊 回测统计 {start_date}~{end_date}\n初始本金: {initial_equity:,.2f}\n基准指数: {used_benchmark}\n"
    for mode in strategy_modes:
        account = accounts[mode]
        final_equity = account['equity_curve'][-1]
        logger.info(
            f"\n🔔 策略[{mode}]最终净值报告\n"
            f"▫️ 初始本金: {initial_equity:,.2f}\n"
            f"▫️ 最终净值: {final_equity:,.2f}\n"
            f"▫️ 绝对收益: {final_equity - initial_equity:+,.2f}\n"
            f"▫️ 剩余持仓: {len(account['positions'])}只"
        )
    for mode in strategy_modes:
        account = accounts[mode]
        total_ret, ann_ret, mdd, sharpe = calculate_risk_metrics(account['equity_curve'], initial_equity, dates)
        stats += f"""
【{mode}】\n最终净值: {account['equity_curve'][-1]:,.2f}
累计收益: {total_ret:.2f}%\n年化收益: {ann_ret:.2f}%\n最大回撤: {mdd:.2f}%\n夏普比率: {sharpe:.2f}\n多头交易: {len(account['buy_records'])}\n空头交易: {len(account['short_records'])}\n
"""

    save_backtest_results(dates, accounts, benchmark_curve, fig)
    return stats, fig

def save_backtest_results(dates, accounts, benchmark_curve, fig):
    """保存回测结果"""
    if not os.path.exists("results"):
        os.makedirs("results")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存净值曲线
    df_data = {'Date': [d.strftime('%Y%m%d') for d in dates]}
    df_data['Benchmark'] = benchmark_curve
    for mode in accounts:
        if len(accounts[mode]['equity_curve']) == len(dates):
            df_data[f'{mode}_Strategy'] = accounts[mode]['equity_curve']
    pd.DataFrame(df_data).to_csv(f'results/backtest_{timestamp}.csv', index=False)
    
    # 保存图表
    fig.savefig(f'results/backtest_{timestamp}.png')
    
    # 保存交易记录
    for mode in accounts:
        if accounts[mode]['buy_records']:
            pd.DataFrame(accounts[mode]['buy_records']).to_csv(f'results/{mode}_long_records_{timestamp}.csv', index=False)
        if accounts[mode].get('short_records'):
            pd.DataFrame(accounts[mode]['short_records']).to_csv(f'results/{mode}_short_records_{timestamp}.csv', index=False)
    
    print(f"✅ 回测结果已保存至 results/backtest_{timestamp}.*")

# Gradio UI 部分
with gr.Blocks() as demo:
    gr.Markdown("# 📊 股票策略回测工具 (V1.1)")

    with gr.Row():
        start_date = gr.Textbox(label="回测开始日期", value=(datetime.today() - timedelta(days=180)).strftime('%Y%m%d'))
        end_date = gr.Textbox(label="回测结束日期", value=datetime.today().strftime('%Y%m%d'))
        initial_equity = gr.Number(value=100000.0, label="初始净值（本金）", precision=2)
        benchmark = gr.Textbox(label="基准指数代码", value=BENCHMARK_INDEX)  # 新增的基准指数输入

    with gr.Row():
        strategy_modes = gr.CheckboxGroup(["稳健型", "激进型", "穿线型"], value=["稳健型"], label="策略模式")  # 添加穿线型选项
        markets = gr.CheckboxGroup(["主板", "创业板", "科创板"], value=["主板"], label="市场选择")

    with gr.Row():
        holding_days = gr.Slider(1, 120, value=10, step=1, label="最大持仓天数")
        expected_return = gr.Slider(1.0, 15.0, value=5.0, step=0.5, label="止盈点 (%)")
        stop_loss = gr.Slider(-15.0, -0.5, value=-3.0, step=0.5, label="固定止损点 (%)")

    with gr.Row():
        cost = gr.Slider(0.0, 0.5, value=0.1, step=0.01, label="交易成本 (%)")
        max_stock_num = gr.Slider(50, 5000, value=200, step=50, label="每次选股最大股票数")

    with gr.Row():
        frequency_mode = gr.Radio(["每日选股", "每周两次", "每隔N天"], value="每周两次", label="选股频率")
        custom_days = gr.Slider(2, 30, value=5, step=1, label="若选择每隔N天，请设置天数")

    run_btn = gr.Button("🚀 开始回测", variant="primary")
    
    with gr.Row():
        output_stats = gr.Textbox(label="回测统计结果")
        output_plot = gr.Plot(label="收益曲线对比")

    # 确保inputs包含所有参数
    run_btn.click(
        run_backtest,
        inputs=[
            start_date, end_date, holding_days, strategy_modes, markets,
            expected_return, stop_loss, cost, benchmark, max_stock_num,
            frequency_mode, custom_days, initial_equity
        ],
        outputs=[output_stats, output_plot]
    )

if __name__ == "__main__":
    demo.launch(server_port=7861)