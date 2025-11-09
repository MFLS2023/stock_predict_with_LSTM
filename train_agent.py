import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

try:  # SB3 >= 2.0 first-party LSTM support (when available)
    from stable_baselines3.ppo.policies import MlpLstmPolicy as PPO_LSTM_POLICY_CLASS
    PPO_CLASS = SB3PPO
except ImportError:  # pragma: no cover - fallback for older/newer layouts
    try:
        from stable_baselines3.common.policies import ActorCriticLstmPolicy as PPO_LSTM_POLICY_CLASS  # type: ignore
        PPO_CLASS = SB3PPO
    except ImportError:  # pragma: no cover - try sb3-contrib recurrent PPO
        try:
            from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy as PPO_LSTM_POLICY_CLASS  # type: ignore
            from sb3_contrib import RecurrentPPO as SB3RecurrentPPO  # type: ignore

            PPO_CLASS = SB3RecurrentPPO
        except ImportError:  # pragma: no cover - LSTM policy unavailable
            PPO_LSTM_POLICY_CLASS = None
            PPO_CLASS = SB3PPO

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
            # 直接从日志记录器的内部字典中获取值
            reward = self.logger.name_to_value.get('rollout/ep_rew_mean')
            if self.log_callback and reward is not None:
                self.log_callback(f"Timestep: {self.num_timesteps}, Mean Reward: {reward:.2f}")
        return True

def train_ppo_model(
    df_train: pd.DataFrame,
    total_timesteps: int,
    output_dir: Path,
    model_filename: str,
    device: str = 'auto',
    log_callback=print,
    ppo_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Trains a PPO model using the DynamicGridEnv.

    :param df_train: DataFrame for training.
    :param total_timesteps: Total number of training steps.
    :param output_dir: Directory to save the model and logs.
    :param model_filename: Filename for the saved model.
    :param device: PyTorch device ('auto', 'cpu', 'cuda').
    :param log_callback: Function to handle log messages.
    :param ppo_kwargs: Optional dictionary to override PPO keyword arguments.
    :return: Path to the saved model.
    """
    log_callback("开始 PPO 模型训练...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 创建环境
    env = DynamicGridEnv(df=df_train, initial_cash=100000)
    log_callback(f"已创建 DynamicGridEnv 环境，观测空间维度: {env.observation_space.shape}, 动作空间维度: {env.action_space.shape}")

    # 2. 配置日志
    log_path = output_dir / "sb3_logs"
    new_logger = configure(str(log_path), ["stdout", "csv", "tensorboard"])

    # 3. 创建 PPO 模型
    # 超参数可以根据需要进行调整
    if PPO_LSTM_POLICY_CLASS is not None:
        default_policy: Any = PPO_LSTM_POLICY_CLASS
        base_policy_kwargs: Dict[str, Any] = dict(
            lstm_hidden_size=128,
            n_lstm_layers=1,
            net_arch=dict(pi=[128], vf=[128]),
        )
        if log_callback:
            algo_name = getattr(PPO_CLASS, "__name__", "PPO")
            log_callback(f"检测到稳定基线支持 LSTM Policy，使用 {algo_name}+MlpLstmPolicy 进行训练。")
    else:
        default_policy = "MlpPolicy"
        base_policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
        if log_callback:
            log_callback("当前 stable-baselines3 版本缺少 MlpLstmPolicy，自动回退到 MlpPolicy。")

    base_params: Dict[str, Any] = dict(
        policy=default_policy,
        env=env,
        policy_kwargs=base_policy_kwargs,
        learning_rate=1e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=0,  # 设置为0，通过回调函数控制输出
        device=device,
        tensorboard_log=str(log_path),
    )

    if ppo_kwargs:
        custom_policy_kwargs = ppo_kwargs.get("policy_kwargs") if isinstance(ppo_kwargs, dict) else None
        merged = {k: v for k, v in ppo_kwargs.items() if k not in {"env", "tensorboard_log", "policy_kwargs"}}
        base_params.update(merged)
        if custom_policy_kwargs:
            policy_kwargs = base_params["policy_kwargs"].copy()
            policy_kwargs.update(custom_policy_kwargs)
            base_params["policy_kwargs"] = policy_kwargs

        if PPO_LSTM_POLICY_CLASS is None and isinstance(base_params["policy"], str) and base_params["policy"] == "MlpPolicy":
            for key in list(base_params["policy_kwargs"].keys()):
                if key in {"lstm_hidden_size", "n_lstm_layers"}:
                    base_params["policy_kwargs"].pop(key, None)

    base_params["env"] = env
    base_params["tensorboard_log"] = str(log_path)

    model = PPO_CLASS(**base_params)
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

    scaler_path: Optional[Path] = None
    feature_scaler = getattr(env, "feature_scaler", None)
    if feature_scaler is not None:
        scaler_path = output_dir / "feature_scaler.pkl"
        try:
            with scaler_path.open("wb") as fh:
                pickle.dump(feature_scaler, fh)
            log_callback(f"特征标准化器已保存至: {scaler_path}")
        except Exception as exc:
            log_callback(f"警告: 保存特征标准化器失败 - {exc}")
            scaler_path = None

    env.close()
    
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
