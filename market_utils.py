# market_utils.py
from datetime import datetime, timedelta
import tushare as ts
import logging
import numpy as np
from config import Config 

logger = logging.getLogger(__name__)

BENCHMARK_INDEX = '000300.SH'
market_status_cache = {}

def get_market_status(trade_date: str) -> str:
    """获取市场状态（牛市/熊市/震荡市/高波动市）"""
    if trade_date in market_status_cache:
        return market_status_cache[trade_date]
    
    trade_date_dt = datetime.strptime(trade_date, '%Y%m%d')
    end_date = trade_date_dt
    start_date = end_date - timedelta(days=90)  # 取90天缓冲区

    pro = ts.pro_api(Config.TUSHARE_TOKEN)

    try:
        df = pro.index_daily(
            ts_code=BENCHMARK_INDEX,
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d'),
            fields="trade_date,close"
        )
        if df is not None and not df.empty:
            df = df.sort_values('trade_date')
            if len(df) < 20:
                logger.warning(f"⚠️ {trade_date} | 获取市场状态数据不足20条，默认震荡市")
                market_status_cache[trade_date] = "震荡市"
                return "震荡市"

            df['ma20'] = df['close'].rolling(20).mean()
            last_close = df['close'].iloc[-1]
            last_ma20 = df['ma20'].iloc[-1]
            
            volatility = df['close'].pct_change().std() * np.sqrt(len(df)) 

            if volatility > 0.2:
                status = "高波动市"
            elif last_close > last_ma20 * 1.08:
                status = "牛市"
            elif last_close < last_ma20 * 0.92:
                status = "熊市"
            else:
                status = "震荡市"
            
            market_status_cache[trade_date] = status
            logger.info(f"📊 {trade_date} | 市场状态：{status} | 收盘：{last_close:.2f} | MA20：{last_ma20:.2f} | 波动率：{volatility:.2%}")
            return status
        else:
            logger.warning(f"⚠️ {trade_date} | 获取市场状态失败，无数据")
    except Exception as e:
        logger.error(f"❌ {trade_date} | 获取市场状态出错: {str(e)}")

    market_status_cache[trade_date] = "震荡市"
    return "震荡市"