# config.py
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

class Config:
    """系统配置管理"""
    # API密钥配置
    TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN', '')
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
    
    # 验证必要的配置
    @classmethod
    def validate(cls):
        """验证配置是否完整"""
        errors = []
        
        if not cls.TUSHARE_TOKEN:
            errors.append("TUSHARE_TOKEN 未设置")
            
        # DeepSeek API是可选的，所以只给出警告
        if not cls.DEEPSEEK_API_KEY:
            print("⚠️ 警告：DEEPSEEK_API_KEY 未设置，将无法使用AI策略分析功能")
            
        if errors:
            raise ValueError(f"配置错误：{', '.join(errors)}")
            
    # 回测配置
    BACKTEST_MAX_STOCKS = 200
    BACKTEST_INITIAL_EQUITY = 100000.0
    BACKTEST_BENCHMARK = '000300.SH'
    
    # 系统配置
    MAX_STOCKS_TO_ANALYZE = 5500
    MIN_DATA_DAYS = 30
    POINTS = 10000
    MAX_CALLS_PER_MIN = 1000
    MAX_WORKERS = 24
    
    # 市场板块
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
    
    # 文件路径配置
    DATA_DIR = "history_data"
    RESULTS_DIR = "results"
    CACHE_DIR = "cache"
    
    @classmethod
    def create_dirs(cls):
        """创建必要的目录"""
        import os
        for dir_path in [cls.DATA_DIR, cls.RESULTS_DIR, cls.CACHE_DIR]:
            os.makedirs(dir_path, exist_ok=True)