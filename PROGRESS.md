# AI动态网格交易优化方案 v2.0 (PPO-Based)

**最后更新**: 2025年11月8日 (已成功完成首次PPO模型训练)

---

### **核心目标 (Core Objective)**

训练一个强化学习（RL）智能体，使其能够根据实时的市场状态，动态地调整网格交易策略的关键参数，以实现在多种市场环境下稳定超越“定投”基准的收益表现。

### **核心思想 (Core Idea)**

我们将AI的角色从一个直接的“交易员”（执行买卖）转变为一个更高级的“策略分析师”（设定交易参数）。AI不进行微观的买卖决策，而是专注于在每个决策周期开始时，为底层的网格交易系统输出一套最优的“作战计划”（即网格参数）。

### **系统架构 (System Architecture) - v2.0 标准**

1.  **智能体 (The Agent - “大脑”)**
    *   **技术选型**: **PPO（Proximal Policy Optimization）** 算法。
    *   **实现**: 通过集成 `stable-baselines3` 这个专业的强化学习库来完成。

2.  **状态空间 (State Space - “眼睛”)**
    *   **定义**: AI进行决策所依赖的全部市场信息。
    *   **构成**: 基于知行均线的特征体系，并包含投资组合的动态状态。

3.  **动作空间 (Action Space - “双手”)**
    *   **定义**: AI可以执行的操作，即输出一套网格参数。
    *   **构成**: 一个包含多个**连续值**的向量: `[grid_interval, stop_loss_rate, position_size]`。

4.  **环境 (Environment - “训练场”)**
    *   **定义**: 模拟真实交易，并为AI提供反馈的虚拟世界。
    *   **构成**: `ai_strategy.py` 中符合 `gymnasium` 标准的自定义环境 `DynamicGridEnv`。

5.  **奖励函数 (Reward Function - “计分板”)**
    *   **定义**: 评估AI动作好坏的唯一标准。
    *   **构成**: 以最大化“超额收益”为核心，同时对“最大回撤”和“交易成本”进行惩罚。
    ```python
    def calculate_reward(portfolio_return, dca_return, max_drawdown, transaction_cost):
        # 核心目标：超越定投基准
        excess_return = portfolio_return - dca_return
        base_reward = excess_return * 5  # 放大信号

        # 风险控制
        drawdown_penalty = -0.3 * max(0, max_drawdown + 0.05)  # 回撤超过5%开始惩罚
        cost_penalty = -0.0002 * transaction_cost

        return base_reward + drawdown_penalty + cost_penalty
    ```

---

### **实施路线图 (Implementation Roadmap)**

**第一阶段：基础建设与特征工程**
*   **状态**: `已完成`

**第二阶段：搭建RL环境**
*   **状态**: `已完成`

**第三阶段：AI智能体设计与训练**
*   **状态**: `已完成`
*   **任务**:
    *   [X] **3.1 & 3.2**: 已有的特征工程和环境类为重构提供了良好基础。
    *   [X] **3.3. 拨乱反正**: 成功决策并回归 v2.0 (PPO) 方案。
    *   [X] **3.4. 重构环境与训练脚本**: 已根据 v2.0 方案，重构 `ai_strategy.py` 和 `train_agent.py`。
    *   [X] **3.5. 启动并完成一次有意义的训练**: 已成功完成首次PPO模型训练并保存了模型。

**第四阶段：评估、对比与分析**
*   **状态**: `进行中`
*   **任务**:
    *   [ ] **4.1. 评估与分析 (当前步骤)**: 分析首次训练的结果，并根据“训练时间过长”的反馈优化训练流程。
    *   [ ] **4.2. 实现评估体系**: 计算超额收益、最大回撤、夏普比率等关键指标。
    *   [ ] **4.3. 结果可视化**: 绘制AI动态网格、静态网格、定投策略的收益对比曲线。

---