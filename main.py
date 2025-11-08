# -*- coding: UTF-8 -*-
"""
@author: hichenway
@知乎: 海晨威
@contact: lyshello123@163.com
@time: 2020/5/9 17:00
@license: Apache
主程序：包括配置，数据读取，日志记录，绘图，模型训练和预测
"""

import logging
import os
import sys
import time
from datetime import datetime
from importlib import import_module
from logging.handlers import RotatingFileHandler
from typing import Iterable, Optional, Sequence

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from plot_utils import ensure_chinese_fonts, get_chinese_font_prop


ensure_chinese_fonts()
_GLOBAL_FONT_PROP = get_chinese_font_prop()


FRAME_MODULES = {
    "pytorch": "model.model_pytorch",
    "keras": "model.model_keras",
    "tensorflow": "model.model_tensorflow",
}

INSTALL_HINTS = {
    "pytorch": "pip install torch",
    "keras": "pip install tensorflow",
    "tensorflow": "pip install tensorflow",
}


class BackendUnavailableError(RuntimeError):
    """表示指定深度学习框架不可用的异常。"""

    def __init__(self, frame_key: str, reason: str, hint: Optional[str] = None) -> None:
        pretty_name = frame_key.title()
        base_message = f"{pretty_name} 框架不可用：{reason}"
        if hint:
            base_message += f"。建议执行：{hint}"
        super().__init__(base_message)
        self.frame_key = frame_key
        self.reason = reason
        self.hint = hint


def import_backend(frame_name: str):
    frame_key = frame_name.lower()
    if frame_key not in FRAME_MODULES:
        raise ValueError(
            f"Unsupported frame '{frame_name}'. Available options: {', '.join(FRAME_MODULES.keys())}"
        )

    try:
        module = import_module(FRAME_MODULES[frame_key])
    except ImportError as exc:
        hint = INSTALL_HINTS.get(frame_key)
        raise BackendUnavailableError(frame_key, str(exc), hint) from exc
    except Exception as exc:  # noqa: BLE001
        raise BackendUnavailableError(frame_key, str(exc)) from exc

    if frame_key in {"keras", "tensorflow"}:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # 静默 TensorFlow 日志
    return module.train, module.predict

class Config:
    """项目配置"""

    model_postfix = {"pytorch": ".pth", "keras": ".h5", "tensorflow": ".ckpt"}

    def __init__(self, used_frame: str = "pytorch"):
        # 数据参数
        self.feature_columns = [
            "open",
            "close",
            "low",
            "high",
            "volume",
            "amount",
            "change",
        ]  # 归一化使用的特征列，使用标准化列名
        self.label_columns = ["low", "high"]  # 需要预测的列
        self.predict_day = 1  # 预测未来几天

        # 网络参数
        self.hidden_size = 128
        self.lstm_layers = 2
        self.dropout_rate = 0.2
        self.time_step = 20  # 使用前多少天的数据来预测

        # 训练参数
        self.do_train = True
        self.do_predict = True
        self.add_train = False
        self.shuffle_train_data = True
        self.device_preference = "auto"  # 设备偏好: 'auto', 'cpu', 'cuda:0', etc.

        # 默认训练集比例和验证集比例，确保和小于1
        self.train_data_rate = 0.80
        self.valid_data_rate = 0.15

        self.batch_size = 64
        self.learning_rate = 0.001
        self.epoch = 20
        self.patience = 5
        self.random_seed = 42

        self.do_continue_train = False
        self.continue_flag = ""
        if self.do_continue_train:
            self.shuffle_train_data = False
            self.batch_size = 1
            self.continue_flag = "continue_"

        # 调试模式
        self.debug_mode = False
        self.debug_num = 500

        # 路径/日志参数
        self.train_data_path = "./data/sh000001.csv"
        self.figure_save_path = os.path.join(".", "figure") + os.sep
        self.model_save_root = os.path.join(".", "checkpoint")
        self.log_save_root = os.path.join(".", "log")
        self.do_log_print_to_screen = True
        self.do_log_save_to_file = True
        self.do_figure_save = False
        self.do_train_visualized = False
        self.show_plots = True  # 是否显示图表/Whether to display plots
        self.last_prediction_result = None

        self.used_frame = used_frame.lower()

        self.refresh_dependent_attributes()
        self.prepare_runtime_dirs()

    def refresh_dependent_attributes(self):
        # 确保 label_columns 都在 feature_columns 中
        missing = [c for c in self.label_columns if c not in self.feature_columns]
        if missing:
            raise ValueError(f"label_columns 中的字段未包含于 feature_columns：{missing}")
        self.label_in_feature_index = [self.feature_columns.index(i) for i in self.label_columns]
        self.input_size = len(self.feature_columns)
        self.output_size = len(self.label_columns)
        postfix = self.model_postfix[self.used_frame]
        self.model_name = f"model_{self.continue_flag}{self.used_frame}{postfix}"

    def prepare_runtime_dirs(self):
        os.makedirs(self.figure_save_path, exist_ok=True)
        os.makedirs(self.model_save_root, exist_ok=True)

        model_dir = os.path.join(self.model_save_root, self.used_frame)
        os.makedirs(model_dir, exist_ok=True)
        self.model_save_path = model_dir + os.sep

        os.makedirs(self.log_save_root, exist_ok=True)
        if self.do_train and (self.do_log_save_to_file or self.do_train_visualized):
            cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            log_dir = os.path.join(self.log_save_root, f"{cur_time}_{self.used_frame}")
        else:
            log_dir = self.log_save_root
        os.makedirs(log_dir, exist_ok=True)
        self.log_save_path = log_dir + os.sep

    def set_used_frame(self, used_frame: str):
        used_frame = used_frame.lower()
        if used_frame not in self.model_postfix:
            raise ValueError(
                f"Unsupported frame '{used_frame}'. Must be one of: {', '.join(self.model_postfix.keys())}"
            )
        self.used_frame = used_frame
        self.refresh_dependent_attributes()
        self.prepare_runtime_dirs()


class Data:
    def __init__(self, config):
        self.config = config
        self.dates: Optional[pd.Series] = None
        self.data, self.data_column_name = self.read_data()

        self.data_num = self.data.shape[0]
        self.train_num = int(self.data_num * self.config.train_data_rate)

        self._ensure_time_step_within_bounds()

        self.mean = np.mean(self.data, axis=0)              # 数据的均值和方差
        self.std = np.std(self.data, axis=0)
        self.std[self.std == 0] = 1
        self.norm_data = (self.data - self.mean)/self.std   # 归一化，去量纲

        self.start_num_in_test = 0      # 测试集中前几天的数据会被删掉，因为它不够一个time_step

    def _ensure_time_step_within_bounds(self) -> None:
        """保证 time_step 不会超过训练样本的数量，否则自动收缩并记录日志。"""
        if self.train_num <= 1:
            raise ValueError(
                "Not enough rows to form even a single training sample. Please provide more data."
            )

        if self.config.time_step < 1:
            logging.getLogger(__name__).warning(
                "time_step (%s) was < 1. Resetting to 1.", self.config.time_step
            )
            self.config.time_step = 1

        if self.train_num > self.config.time_step:
            return

        adjusted_time_step = max(self.train_num - 1, 1)
        if adjusted_time_step == self.config.time_step:
            return

        logging.getLogger(__name__).warning(
            "time_step (%s) exceeds available training rows (%s). Auto-adjusting to %s.",
            self.config.time_step,
            self.train_num,
            adjusted_time_step,
        )
        self.config.time_step = adjusted_time_step

    def read_data(self):                # 读取初始数据
        read_kwargs = {
            "skip_blank_lines": True,
            "engine": "python",
        }
        if self.config.debug_mode:
            read_kwargs["nrows"] = self.config.debug_num

        init_data = self._load_dataframe(self.config.train_data_path, **read_kwargs)
        init_data = self._standardize_dataframe(init_data)
        init_data = self._ensure_feature_columns(init_data)

        missing_columns = [col for col in self.config.feature_columns if col not in init_data.columns]
        if missing_columns:
            raise ValueError(
                f"Columns {missing_columns} required by feature_columns were not found in data file "
                f"{self.config.train_data_path}. Available columns: {list(init_data.columns)}"
            )

        feature_frame = init_data[self.config.feature_columns].copy()
        date_series = None
        if "date" in init_data.columns:
            date_series = init_data["date"].copy()
        for col in self.config.feature_columns:
            feature_frame.loc[:, col] = pd.to_numeric(feature_frame[col], errors="coerce")

        # 核心行情四要素必须有效，其他列缺失则用 0 填充避免整行丢弃
        essential_cols = [c for c in ("open", "high", "low", "close") if c in feature_frame.columns]
        feature_frame = feature_frame.dropna(subset=essential_cols, how="any")
        optional_cols = [c for c in feature_frame.columns if c not in essential_cols]
        if optional_cols:
            feature_frame[optional_cols] = feature_frame[optional_cols].fillna(0.0)

        feature_frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        feature_frame.fillna(value=0.0, inplace=True)

        if date_series is not None:
            date_series = date_series.loc[feature_frame.index]
            date_series = pd.to_datetime(date_series, errors="coerce")
            date_series = date_series.reset_index(drop=True)
        feature_frame = feature_frame.reset_index(drop=True)
        if date_series is not None and not date_series.isna().all():
            self.dates = date_series
        else:
            self.dates = None

        if feature_frame.empty:
            raise ValueError(
                "No usable rows found after cleaning. Please verify the CSV columns/encoding or provide more data."
            )

        return feature_frame.values, feature_frame.columns.tolist()     # .columns.tolist() 是获取列名

    def _load_dataframe(self, path: str, **read_kwargs) -> pd.DataFrame:
        encodings: Sequence[str] = ("utf-8-sig", "utf-8", "gbk", "gb2312")
        errors = []
        for encoding in encodings:
            try:
                df = pd.read_csv(path, encoding=encoding, **read_kwargs)
                if df.empty:
                    continue
                return df
            except Exception as exc:  # noqa: BLE001
                errors.append((encoding, exc))
        error_messages = "; ".join(f"{enc}: {err}" for enc, err in errors)
        raise ValueError(f"Failed to read CSV {path}. Tried encodings {encodings}. Errors: {error_messages}")

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # 删除全为空的行/列（例如 CSV 末尾的大量空行）
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

        column_mapping = {
            "vol": "volume",
            "volume": "volume",
            "成交量": "volume",
            "volume(手)": "volume",
            "money": "amount",
            "amount": "amount",
            "成交额": "amount",
            "成交金额": "amount",
            "turnover": "amount",
            # change 别名
            "pct_chg": "change",
            "change_pct": "change",
            "涨跌幅": "change",
            "涨跌幅(%)": "change",
            # 价格列
            "open": "open",
            "close": "close",
            "low": "low",
            "high": "high",
            "开盘价": "open",
            "收盘价": "close",
            "最低价": "low",
            "最高价": "high",
            "昨收": "pre_close",
        }

        normalized_columns = []
        for col in df.columns:
            canonical_col = str(col).strip().lower()
            canonical_col = column_mapping.get(canonical_col, canonical_col)
            normalized_columns.append(canonical_col)

        df.columns = normalized_columns
        df = df.loc[:, ~df.columns.duplicated()]

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df = df.sort_values("date").reset_index(drop=True)

        # 如果不存在 change 列，尝试用收盘价计算
        if "change" not in df.columns and "close" in df.columns:
            df_prev = df["close"].shift(1)
            df["change"] = (df["close"] - df_prev) / df_prev

        # 移除 index_code 等无关列
        drop_candidates = [col for col in ("index_code", "code", "symbol") if col in df.columns]
        if drop_candidates:
            df = df.drop(columns=drop_candidates)

        df = df.drop_duplicates(subset="date" if "date" in df.columns else None, keep="last")

        return df

    def _ensure_feature_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "amount" not in df.columns and {"volume", "close"}.issubset(df.columns):
            df["amount"] = df["volume"] * df["close"]

        if "volume" not in df.columns and {"amount", "close"}.issubset(df.columns):
            close = df["close"].replace(0, np.nan)
            df["volume"] = df["amount"] / close

        if "change" not in df.columns and "close" in df.columns:
            df_prev = df["close"].shift(1)
            df["change"] = (df["close"] - df_prev) / df_prev

        return df

    def get_train_and_valid_data(self):
        feature_data = self.norm_data[:self.train_num]
        label_data = self.norm_data[self.config.predict_day : self.config.predict_day + self.train_num,
                                    self.config.label_in_feature_index]    # 将延后几天的数据作为label

        if self.train_num <= self.config.time_step:
            raise ValueError(
                "Training samples are insufficient after cleaning. "
                f"Need more than time_step ({self.config.time_step}) rows but got {self.train_num}. "
                "Consider providing more history or reducing time_step."
            )

        if not self.config.do_continue_train:
            # 在非连续训练模式下，每time_step行数据会作为一个样本，两个样本错开一行，比如：1-20行，2-21行。。。。
            train_x = [feature_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
            train_y = [label_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
        else:
            # 在连续训练模式下，每time_step行数据会作为一个样本，两个样本错开time_step行，
            # 比如：1-20行，21-40行。。。到数据末尾，然后又是 2-21行，22-41行。。。到数据末尾，……
            # 这样才可以把上一个样本的final_state作为下一个样本的init_state，而且不能shuffle
            # 目前本项目中仅能在pytorch的RNN系列模型中用
            train_x = [feature_data[start_index + i*self.config.time_step : start_index + (i+1)*self.config.time_step]
                       for start_index in range(self.config.time_step)
                       for i in range((self.train_num - start_index) // self.config.time_step)]
            train_y = [label_data[start_index + i*self.config.time_step : start_index + (i+1)*self.config.time_step]
                       for start_index in range(self.config.time_step)
                       for i in range((self.train_num - start_index) // self.config.time_step)]

        train_x, train_y = np.array(train_x), np.array(train_y)

        sample_count = len(train_x)
        if sample_count < 2 or self.config.valid_data_rate <= 0:
            logging.getLogger(__name__).warning(
                "Not enough samples (%s) to create a separate validation split. Using training data for validation.",
                sample_count,
            )
            train_x_arr = np.array(train_x)
            train_y_arr = np.array(train_y)
            return train_x_arr, train_x_arr.copy(), train_y_arr, train_y_arr.copy()

        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=self.config.valid_data_rate,
                                                              random_state=self.config.random_seed,
                                                              shuffle=self.config.shuffle_train_data)   # 划分训练和验证集，并打乱
        return train_x, valid_x, train_y, valid_y

    def get_test_data(self, return_label_data=False):
        feature_data = self.norm_data[self.train_num:]
        if feature_data.shape[0] == 0:
            raise ValueError(
                "No test samples available for prediction. 请降低 train_data_rate 或提供更多历史数据后再进行预测。"
            )
        sample_interval = min(feature_data.shape[0], self.config.time_step)     # 防止time_step大于测试集数量
        self.start_num_in_test = feature_data.shape[0] % sample_interval  # 这些天的数据不够一个sample_interval
        time_step_size = feature_data.shape[0] // sample_interval

        # 在测试数据中，每time_step行数据会作为一个样本，两个样本错开time_step行
        # 比如：1-20行，21-40行。。。到数据末尾。
        test_x = [feature_data[self.start_num_in_test+i*sample_interval : self.start_num_in_test+(i+1)*sample_interval]
                   for i in range(time_step_size)]
        if return_label_data:       # 实际应用中的测试集是没有label数据的
            label_data = self.norm_data[self.train_num + self.start_num_in_test:, self.config.label_in_feature_index]
            return np.array(test_x), label_data
        return np.array(test_x)

def load_logger(
    config: Config, extra_handlers: Optional[Iterable[logging.Handler]] = None
) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    if config.do_log_print_to_screen:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(datefmt="%Y/%m/%d %H:%M:%S", fmt="[ %(asctime)s ] %(message)s")
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if config.do_log_save_to_file:
        file_handler = RotatingFileHandler(
            os.path.join(config.log_save_path, "out.log"), maxBytes=1024000, backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        config_dict = {}
        for key in dir(config):
            if key.startswith("_"):
                continue
            value = getattr(config, key)
            if callable(value):
                continue
            config_dict[key] = value
        config_str = str(config_dict)
        config_list = config_str[1:-1].split(", '") if len(config_str) > 2 else []
        config_save_str = "\nConfig:\n" + "\n'".join(config_list)
        logger.info(config_save_str)

    if extra_handlers:
        for handler in extra_handlers:
            logger.addHandler(handler)

    return logger

def draw(config: Config, origin_data: Data, logger, predict_norm_data: np.ndarray):
    label_data = origin_data.data[origin_data.train_num + origin_data.start_num_in_test : ,
                                            config.label_in_feature_index]
    predict_data = predict_norm_data * origin_data.std[config.label_in_feature_index] + \
                   origin_data.mean[config.label_in_feature_index]   # 通过保存的均值和方差还原数据
    
    logger.info(f"=== 绘图数据检查 ===")
    logger.info(f"label_data 形状: {label_data.shape}")
    logger.info(f"predict_data 形状: {predict_data.shape}")
    logger.info(f"预测天数 (predict_day): {config.predict_day}")
    
    assert label_data.shape[0]==predict_data.shape[0], "The element number in origin and predicted data is different"

    label_name = [origin_data.data_column_name[i] for i in config.label_in_feature_index]
    label_column_num = len(config.label_columns)
    
    logger.info(f"绘制 {label_column_num} 个标签的预测图表: {label_name}")

    # label 和 predict 是错开config.predict_day天的数据的
    # 下面是两种norm后的loss的计算方式，结果是一样的，可以简单手推一下
    # label_norm_data = origin_data.norm_data[origin_data.train_num + origin_data.start_num_in_test:,
    #              config.label_in_feature_index]
    # loss_norm = np.mean((label_norm_data[config.predict_day:] - predict_norm_data[:-config.predict_day]) ** 2, axis=0)
    # logger.info("The mean squared error of stock {} is ".format(label_name) + str(loss_norm))

    loss = np.mean((label_data[config.predict_day:] - predict_data[:-config.predict_day] ) ** 2, axis=0)
    loss_norm = loss/(origin_data.std[config.label_in_feature_index] ** 2)
    logger.info("The mean squared error of stock {} is ".format(label_name) + str(loss_norm))

    label_X = range(origin_data.data_num - origin_data.train_num - origin_data.start_num_in_test)
    predict_X = [ x + config.predict_day for x in label_X]

    payload = {
        "label_names": label_name,
        "actual": label_data.tolist(),
        "predicted": predict_data.tolist(),
        "actual_x": list(label_X),
        "predicted_x": list(predict_X),
        "predict_day": int(config.predict_day),
    }
    if getattr(origin_data, "dates", None) is not None:
        try:
            aligned_dates = origin_data.dates.iloc[
                origin_data.train_num + origin_data.start_num_in_test :
            ]
            payload["dates"] = [
                d.isoformat() if isinstance(d, (pd.Timestamp, datetime)) else str(d)
                for d in aligned_dates
            ]
        except Exception:
            payload["dates"] = None
    config.last_prediction_result = payload

    backend = plt.get_backend().lower()
    interactive_backend = "agg" not in backend and backend not in {"pdf", "ps", "svg", "cairo", "template"}
    has_display = True
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        has_display = False

    if config.show_plots and interactive_backend and has_display:
        logger.info("开始绘制预测图表...")
        for i in range(label_column_num):
            fig = plt.figure(i+1, figsize=(12, 6))  # 增大图表尺寸
            ax = fig.add_subplot(111)

            # 绘制实际值（蓝色实线）
            ax.plot(label_X, label_data[:, i], 'b-', linewidth=1.5, label='实际值 (Actual)', alpha=0.8)

            # 绘制预测值（红色虚线）
            ax.plot(predict_X, predict_data[:, i], 'r--', linewidth=1.5, label='预测值 (Predicted)', alpha=0.8)

            title_text = "预测 {} 走势 (框架: {})".format(label_name[i], config.used_frame)
            if _GLOBAL_FONT_PROP is not None:
                ax.set_title(title_text, fontproperties=_GLOBAL_FONT_PROP, fontsize=14, fontweight='bold')
                ax.set_xlabel("样本序号", fontproperties=_GLOBAL_FONT_PROP, fontsize=12)
                ax.set_ylabel("价格", fontproperties=_GLOBAL_FONT_PROP, fontsize=12)
                ax.legend(prop=_GLOBAL_FONT_PROP, loc='best', fontsize=10)
            else:
                ax.set_title(title_text, fontsize=14, fontweight='bold')
                ax.set_xlabel("样本序号", fontsize=12)
                ax.set_ylabel("价格", fontsize=12)
                ax.legend(loc='best', fontsize=10)

            ax.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()

            logger.info(
                "The predicted stock {} for the next {} day(s) is: ".format(label_name[i], config.predict_day)
                + str(np.squeeze(predict_data[-config.predict_day:, i]))
            )

            if config.do_figure_save:
                figure_name = f"{config.continue_flag}predict_{label_name[i]}_with_{config.used_frame}.png"
                save_path = os.path.join(config.figure_save_path, figure_name)
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                logger.info(f"图表已保存至: {save_path}")

        logger.info("所有图表绘制完成，正在显示...")
        plt.show()
    elif config.show_plots and not interactive_backend:
        logger.info("当前 Matplotlib 后端 '%s' 为非交互模式，已跳过图形展示。", backend)
    elif config.show_plots and not has_display:
        logger.info("未检测到图形显示环境 (DISPLAY 未设置)，已跳过图形展示。")

def main(
    config: Config,
    logger: Optional[logging.Logger] = None,
    extra_logger_handlers: Optional[Iterable[logging.Handler]] = None,
):
    config.refresh_dependent_attributes()
    config.prepare_runtime_dirs()
    logger = logger or load_logger(config, extra_logger_handlers)

    # --- 设备选择逻辑 ---
    device = config.device_preference
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("自动检测到 CUDA，将使用 GPU。")
            else:
                device = "cpu"
                logger.info("自动检测未发现可用 CUDA，将使用 CPU。")
        except ImportError:
            device = "cpu"
            logger.warning("PyTorch 未安装，自动回退到 CPU。")
    else:
        logger.info(f"使用用户指定的设备: {device}")
    # ------------------

    try:
        train_func, predict_func = import_backend(config.used_frame)
    except BackendUnavailableError as exc:
        logger.error(str(exc))
        raise SystemExit(1) from exc

    try:
        np.random.seed(config.random_seed)
        data_gainer = Data(config)
        logger.info("Loaded %d rows of data from %s", data_gainer.data_num, config.train_data_path)

        if (config.do_train or config.do_predict) and data_gainer.data_num <= config.time_step:
            raise ValueError(
                "Insufficient data rows after preprocessing. Please provide more historical records."
            )

        if config.do_train:
            train_X, valid_X, train_Y, valid_Y = data_gainer.get_train_and_valid_data()
            train_func(config, logger, [train_X, train_Y, valid_X, valid_Y], device=device)

        if config.do_predict:
            model_path = Path(config.model_save_path) / config.model_name
            if not model_path.exists():
                raise FileNotFoundError(
                    f"未找到预测所需的模型文件: {model_path}. 请先运行训练阶段 (移除 --no-train) 或指定有效的模型路径。"
                )
            test_X, test_Y = data_gainer.get_test_data(return_label_data=True)
            pred_result = predict_func(config, test_X, device=device)
            draw(config, data_gainer, logger, pred_result)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        raise SystemExit(1) from exc
    except ValueError as exc:
        logger.error(str(exc))
        raise SystemExit(1) from exc
    except Exception:
        logger.error("Run Error", exc_info=True)
        raise
    finally:
        for handler in logger.handlers:
            handler.flush()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--frame", choices=list(FRAME_MODULES.keys()), default="pytorch")
    parser.add_argument("-d", "--train_data_path", help="Path to stock CSV file")
    parser.add_argument("--no-train", dest="do_train", action="store_false", help="Disable training phase")
    parser.add_argument("--no-predict", dest="do_predict", action="store_false", help="Disable prediction phase")
    parser.add_argument(
        "--no-show-plots",
        dest="show_plots",
        action="store_false",
        help="Disable interactive Matplotlib windows (useful for headless environments)",
    )
    parser.set_defaults(do_train=True, do_predict=True)
    args = parser.parse_args()

    config = Config(used_frame=args.frame)
    for key, value in vars(args).items():
        if key == "frame" or value is None:
            continue
        setattr(config, key, value)

    config.refresh_dependent_attributes()
    config.prepare_runtime_dirs()

    main(config)
