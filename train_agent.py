import os
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from ai_strategy import DynamicGridEnv

class LoggingCallback(BaseCallback):
    """A simple callback to print training progress."""
    def __init__(self, verbose=0, log_callback=None):
        super().__init__(verbose)
        self.log_callback = log_callback

    def _on_step(self) -> bool:
        # Log every 1000 steps
        if self.num_timesteps % 1000 == 0:
            # You can access logger data and log it
            reward = self.logger.get_latest_key_value('rollout/ep_rew_mean')
            if self.log_callback and reward is not None:
                self.log_callback(f"Timestep: {self.num_timesteps}, Mean Reward: {reward:.2f}")
        return True

def train_ppo_model(
    df_train: pd.DataFrame,
    total_timesteps: int,
    output_dir: Path,
    model_filename: str,
    device: str = 'auto',
    log_callback=print
):
    """
    Trains a PPO model using the DynamicGridEnv.

    :param df_train: DataFrame for training.
    :param total_timesteps: Total number of training steps.
    :param output_dir: Directory to save the model and logs.
    :param model_filename: Filename for the saved model.
    :param device: PyTorch device ('auto', 'cpu', 'cuda').
    :param log_callback: Function to handle log messages.
    :return: Path to the saved model.
    """
    log_callback("开始 PPO 模型训练...")
    
    # 1. 创建环境
    env = DynamicGridEnv(df=df_train, initial_cash=100000)
    log_callback(f"已创建 DynamicGridEnv 环境，观测空间维度: {env.observation_space.shape}, 动作空间维度: {env.action_space.shape}")

    # 2. 配置日志
    log_path = output_dir / "sb3_logs"
    new_logger = configure(str(log_path), ["stdout", "csv", "tensorboard"])

    # 3. 创建 PPO 模型
    # 超参数可以根据需要进行调整
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=0, # 设置为0，通过回调函数控制输出
        device=device,
        tensorboard_log=str(log_path)
    )
    model.set_logger(new_logger)
    log_callback(f"PPO 模型已创建，将在 {device} 设备上进行训练。")

    # 4. 训练模型
    log_callback(f"开始训练，总步数: {total_timesteps}...")
    callback = LoggingCallback(log_callback=log_callback)
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False) # GUI模式下禁用tqdm进度条
    log_callback("模型训练完成。")

    # 5. 保存模型
    saved_model_path = output_dir / model_filename
    model.save(saved_model_path)
    log_callback(f"模型已保存至: {saved_model_path}")
    
    return saved_model_path

if __name__ == '__main__':
    # 这是一个用于直接运行此脚本进行测试的示例
    print("这是一个用于训练AI智能体的模块，请通过其他脚本调用。")
    # 例如:
    # data = pd.read_csv("path/to/your/data.csv", parse_dates=['Date'])
    # train_ppo_model(
    #     df_train=data,
    #     total_timesteps=50000,
    #     output_dir=Path("models"),
    #     model_filename="ppo_test.zip"
    # )
