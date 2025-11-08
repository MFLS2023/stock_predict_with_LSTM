# LSTM 股票预测与回测桌面工具

本项目基于 LSTM 做短期价格预测，并提供一个 PyQt5 桌面 GUI，目标是把数据预处理、模型训练、预测、可视化与回测串成一个易用的桌面工具，便于做策略验证和结果展示。

主要功能概览：

- 支持 PyTorch / Keras / TensorFlow 的 LSTM 模型训练与预测（通过 `main.py` 配置选择）
- 提供基于 PyQt5 的桌面界面 `gui.py`，可选择数据文件、一键训练/预测、查看日志
- 可视化：标准 K 线（candlestick）图、成交量子图、常见技术指标（如 KDJ/RSI/MACD）
- 回测：策略回测与买入并持有（Buy & Hold）对比，输出收益曲线与关键绩效指标（年化收益、最大回撤、夏普比率等）
- 支持导出回测报告（CSV/图片）以便后续分析

本仓库包含基础的模型实现和一个入门级的 GUI，后续可以扩展为更完整的策略套件（多策略、参数搜索、蒙特卡洛测试等）。

## 快速开始

### 环境要求

- Python 3.8+
- 建议使用虚拟环境（venv / conda）

### 安装依赖

项目根目录下运行：

```pwsh
pip install -r requirements.txt
```

（可选）若需要绘制更美观的 K 线图，建议安装 `mplfinance`：

```pwsh
pip install mplfinance
```

> 小贴士：如果暂时只使用 PyTorch，可以跳过安装 Keras/TensorFlow 相关依赖。命令行与 GUI 会自动检测框架可用性，缺失时会给出类似“请执行 `pip install tensorflow`”的提示并跳过对应框架。

### 启动 GUI

```pwsh
python gui.py
```

GUI 功能简要说明：

- "数据文件"：选择单只股票的 CSV 文件（示例：`data/sh000001.csv`）
- "框架"：选择训练/预测所用的深度学习框架（PyTorch/Keras/TensorFlow）
- 超参数：时间步（time_step）、预测天数、batch、epoch、学习率等
- 点击 "开始运行" 后，程序会在后台线程执行训练/预测并在界面实时输出日志；若选择预测，会在 GUI 中弹出绘图界面显示 K 线与预测结果。

## 数据格式

建议数据 CSV 的表头（大小写不敏感）示例：

```
Date,Open,High,Low,Close,Vol,Amount,Change,Change_Pct,Amplitude
1990/12/19,96.05,99.98,95.79,99.98,1260,494000,0,0,4.19
```

本项目的数据加载器会做常见列名映射（如 `Vol` → `volume`、`Money` → `amount`、`Change_Pct`/`pct_chg` → `change`），并自动删除文件尾部的全空行。训练的默认特征列为：`open, close, low, high, volume, amount, change`。

## 项目目录（简化）

```
.
├── data/                      # 原始与示例数据（如 sh000001.csv）
├── figure/                    # 输出的图像
├── model/                     # 模型实现（pytorch/keras/tensorflow）
├── gui.py                     # PyQt5 桌面界面（入口）
├── main.py                    # 程序主流程：数据-训练-预测-绘图
├── requirements.txt           # 依赖列表
└── Readme.md                  # 本文档
```

## 回测输出指标（示例）

- 总收益率（Total Return）
- 年化收益（Annualized Return）
- 最大回撤（Max Drawdown）
- 夏普比率（Sharpe Ratio）
- 胜率、盈亏比

## 如何扩展

- 增加更多策略实现并在 GUI 中注册
- 集成参数搜索（Grid Search / Random Search / Bayesian）
- 集成交易成本与滑点模型以提高回测真实性

## 开发与贡献

欢迎提交 issue 或 PR。若要贡献代码，请遵循项目的编码规范并附带相应的测试用例/回测结果。

## Predict stock with LSTM

This project includes training and predicting processes with LSTM for stock data. The characteristics is as fellow: 

- Concise and modular
- Support three mainstream deep learning frameworks of pytorch, keras and tensorflow
- Parameters, models and frameworks can be highly customized and modified
- Supports incremental training
- Support predicting multiple indicators at the same time
- Support predicting any number of days
- Support train visualization and log record


Chinese introduction can refer to : <https://blog.csdn.net/songyunli1111/article/details/78513811>


The simultaneous predict results for stock high and low price with pytorch show as follow:

![predict_high_with_pytorch](https://github.com/hichenway/stock_predict_with_LSTM/blob/master/figure/predict_high_with_pytorch.png)

![predict_low_with_pytorch](https://github.com/hichenway/stock_predict_with_LSTM/blob/master/figure/predict_low_with_pytorch.png)

### 数据格式约定

- 每个股票对应一个 CSV 文件，建议命名为 `data/<股票代码>.csv`，示例：`data/sh000001.csv`。
- 表头支持新的标准列名：`Date, Open, High, Low, Close, Vol, Amount, Change, Change_Pct, Amplitude`。
- 代码会自动转换常见的别名，例如 `Vol`→`volume`、`Money`→`amount`，也兼容旧版 `stock_data.csv` 中的字段。
- 训练特征默认使用 `open, close, low, high, volume, amount, change`，需要和表头一致（忽略大小写即可）。
- 如果文件末尾存在空行或全为空的记录，程序会自动过滤。
- 若历史数据较短，程序会在运行时自动缩短 `time_step` 并发出日志警告，以便仍能训练模型；建议尽可能提供更多样本或手动调整 `time_step` 以获得更稳定的结果。

### 桌面图形界面（PyQt5）

除了命令行脚本外，现在可以通过 `gui.py` 启动一个基于 PyQt5 的可视化界面，方便选择股票数据文件、切换框架（PyTorch/Keras/TensorFlow）、调整训练参数并运行训练或预测流程。

```bash
python gui.py
```

运行后可以在界面中：

- 浏览选择任意 CSV 数据文件。
- 勾选需要执行的任务（训练 / 预测）。
- 调整时间步长、学习率、Epoch 等常用超参数。
- 实时查看日志输出，失败时会弹出错误提示。

请确保已安装 `PyQt5`（在 `requirements.txt` 中已列出）。

## 未来功能展望：AI 驱动的动态交易策略

我们正在规划一个重大功能升级，旨在利用 AI/深度学习技术，为网格交易策略引入动态的买卖（BS）止盈止损决策，以期在回测中超越传统的定期定额投资（DCA）策略和市场基准。

**核心目标：**
训练一个 AI 模型，使其能够根据市场状态（如 KDJ 指标、涨跌幅、RSI、MACD 等）动态地决定买入、卖出或持有，从而形成一个智能化的“动态网格”交易策略。

**主要实现步骤：**

1.  **AI 策略建模与训练**：
    *   **决策模型**：AI 模型将学习在每个交易日根据市场特征做出“买入”、“卖出”或“持有”的决策。
    *   **特征工程**：利用 KDJ、涨跌幅、RSI、MACD 等多种技术指标作为模型的输入特征。
    *   **强化学习**：采用强化学习范式训练 AI 模型，通过模拟交易的奖励与惩罚机制，优化其决策能力。
    *   **UI 扩展**：在回测界面增加训练集和测试集的日期范围选择功能。

2.  **策略回测、对比与可视化**：
    *   **定投策略基准**：实现一个可配置的定期定额投资（DCA）回测功能，与 AI 策略进行公平对比。
    *   **多策略回测**：在用户指定的测试集日期范围内，同时运行 AI 策略、DCA 策略以及沪深300 ETF 的基准走势。
    *   **结果展示**：生成包含三条收益曲线（AI 策略、DCA 策略、基准 ETF）的回测图表，并展示关键绩效指标（总收益率、年化收益、最大回撤、夏普比率等）。

这将使平台从简单的预测和固定策略回测，升级为具备智能决策和策略优化能力的量化交易研究工具。
