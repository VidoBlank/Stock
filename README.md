# 量化选股工具

这是一个基于Tushare数据的量化选股工具，支持评分推荐、策略回测和可视化界面。

## 功能特点

- 📈 多策略选股：支持稳健型、激进型、穿线型等多种策略模式
- 🔍 智能分析：集成多个技术指标（MACD、均线、布林带、RSI等）
- 🔄 回测系统：支持历史数据回测，评估策略性能
- 🎯 市场情绪：监测市场情绪，动态调整策略权重
- 💹 风险控制：内置止盈止损和仓位管理
- 🌐 可视化界面：基于Gradio的友好操作界面

## 系统要求

- Python 3.13.2
- Tushare Pro账户（需要积分获取数据权限）
- DeepSeek API密钥（可选，用于智能策略分析，目前功能未完善）

## 快速开始

### 1. 下载项目
```bash
下载项目到指定文件夹，文件夹中的bat启动文件可以自己编辑，根据你的解压位置来改动。
安装python
注册并获取Tushare token
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置环境

#### 方法一：使用 .env 文件（推荐）
在根目录创建一个.env文件并填入你的API密钥：
```bash
TUSHARE_TOKEN=（必须填写）
DEEPSEEK_API_KEY=（目前功能未完善，可以不填）
```

然后编辑 `.env` 文件：
```
TUSHARE_TOKEN=你的tushare_token
DEEPSEEK_API_KEY=你的deepseek_api_key
```

#### 方法二：使用环境变量
```bash
export TUSHARE_TOKEN=你的tushare_token
export DEEPSEEK_API_KEY=你的deepseek_api_key
```

### 4. 运行系统

#### 主程序（选股系统）
```bash
python appy.py
```
访问 http://localhost:7860

#### 回测系统
```bash
python backtest_tool.py
```
访问 http://localhost:7861

## 项目结构

```
quant-stock-selection/
├── appy.py                 # 主程序：选股系统
├── backtest_tool.py        # 回测工具
├── market_utils.py         # 市场工具函数
├── fetch_a_shares_data.py  # 获取本地数据（暂无用）
├── build_concept_mapping.py  # 获取行业概念板块数据
├── config.py              # 配置管理
├── requirements.txt       # 依赖包列表
├── .env.example          # 环境变量示例
├── .gitignore           # Git忽略文件
├── README.md            # 项目说明文档
├── history_data/        # 历史数据目录（自动创建）
└── results/            # 回测结果目录（自动创建）
```

## 核心功能说明

### 1. 策略模式

- **稳健型**：趋势型为主，适度保留动量与反弹，规避风险
- **激进型**：突出短线动量与量能机会，趋势适当降低
- **穿线型**：专注于捕捉均线穿越信号，寻找一阳穿三线形态

### 2. 市场选择

- 主板、创业板、科创板
- 中证500、沪深300、上证50等指数成分股


### 3. 风险控制

- 动态止盈止损
- 波动率限制
- 仓位管理
- 资金流向监控

## 注意事项

1. Tushare数据有调用频率限制，请合理使用，某些数据接口需要足够的Tushare积分
2. 不同的策略有不同的评分逻辑，相同股票在不同策略下的评分不同是正常现象
3. 在config中可以调整接口的速率与线程，请根据具体的tushare积分权限调整防止接口超限
4.股票投资有风险，本工具仅供参考，不构成投资建议



## 贡献指南

欢迎提交Issue和Pull Request！

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过997818815@qq.com联系