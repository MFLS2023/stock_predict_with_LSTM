import importlib
import logging
import os
import re
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
import mplfinance as mpf
from plot_utils import (
    ensure_chinese_fonts,
    get_chinese_font_prop,
    get_chinese_rc_params,
)

ensure_chinese_fonts()

import numpy as np
import pandas as pd

from backtest import run_ai_comparison_backtest, run_backtest, run_grid_backtest

from train_agent import train_ppo_model

from main import Config, FRAME_MODULES, main as run_pipeline

if TYPE_CHECKING:  # pragma: no cover
    import torch  # type: ignore
    import tensorflow as tf  # type: ignore

from PyQt5.QtCore import QObject, QThread, pyqtSignal, QTranslator, QLocale, QLibraryInfo, Qt, QTimer, QDate
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QTabWidget,  # Added QTabWidget
    QGroupBox,
    QVBoxLayout,
    QWidget,
)


FRAME_DISPLAY_NAMES = {
    "pytorch": "PyTorch",
    "keras": "Keras",
    "tensorflow": "TensorFlow",
}

AI_DEFAULT_TRADE_FEE = 0.001

LOG_PREFIX_PATTERN = re.compile(r"^\[\d{2}:\d{2}:\d{2}\]")

PRICE_COLUMN_ALIASES: Dict[str, str] = {
    "date": "Date",
    "datetime": "Date",
    "timestamp": "Date",
    "交易日期": "Date",
    "time": "Date",
    "open": "Open",
    "开盘": "Open",
    "openprice": "Open",
    "high": "High",
    "最高": "High",
    "low": "Low",
    "最低": "Low",
    "close": "Close",
    "收盘": "Close",
    "收盘价": "Close",
    "last": "Close",
    "adjclose": "Close",
    "volume": "Volume",
    "vol": "Volume",
    "成交量": "Volume",
    "amount": "Amount",
    "成交额": "Amount",
    "turnover": "Amount",
    "amount_wan": "AmountWan",
}

NUMERIC_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "Amount", "AmountWan"]


class LogSignal(QObject):
    message = pyqtSignal(str)


class QtLogHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.signal = LogSignal()
        self.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            self.signal.message.emit(message)
        except Exception:
            self.handleError(record)


class WorkerThread(QThread):
    succeeded = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, config: Config, handler: logging.Handler):
        super().__init__()
        self.config = config
        self.handler = handler

    def run(self) -> None:
        logger = logging.getLogger()
        try:
            run_pipeline(self.config, extra_logger_handlers=[self.handler])
            self.succeeded.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            if self.handler in logger.handlers:
                logger.removeHandler(self.handler)
            self.handler.close()


class PpoTrainingThread(QThread):
    succeeded = pyqtSignal(str)
    failed = pyqtSignal(str)
    log_message = pyqtSignal(str)

    def __init__(
        self,
        df_train: pd.DataFrame,
        total_timesteps: int,
        output_dir: Path,
        model_filename: str,
        device: str = "auto",
    ) -> None:
        super().__init__()
        self.df_train = df_train
        self.total_timesteps = int(total_timesteps)
        self.output_dir = output_dir
        self.model_filename = model_filename
        self.device = device

    def run(self) -> None:
        def emit_log(message: str) -> None:
            self.log_message.emit(message)

        try:
            saved_path = train_ppo_model(
                df_train=self.df_train,
                total_timesteps=self.total_timesteps,
                output_dir=self.output_dir,
                model_filename=self.model_filename,
                device=self.device,
                log_callback=emit_log,
            )
            self.succeeded.emit(str(saved_path))
        except Exception:
            self.failed.emit(traceback.format_exc())


class AiBacktestThread(QThread):
    succeeded = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
        self,
        model_path: Path,
        df_test: pd.DataFrame,
        initial_cash: float,
        monthly_invest: float,
        fee: float,
    ) -> None:
        super().__init__()
        self.model_path = Path(model_path)
        self.df_test = df_test.copy()
        self.initial_cash = float(initial_cash)
        self.monthly_invest = float(monthly_invest)
        self.fee = float(fee)

    def run(self) -> None:
        try:
            result = run_ai_comparison_backtest(
                model_path=str(self.model_path),
                df_test=self.df_test,
                initial_cash=self.initial_cash,
                monthly_invest=self.monthly_invest,
                fee=self.fee,
            )
            self.succeeded.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("股票预测与回测平台")
        self.resize(1280, 800)

        # --- Data State Management ---
        self.full_df: Optional[pd.DataFrame] = None
        self.current_display_rows: int = 300
        self.current_data_path: Optional[str] = None
        # ---------------------------

        self.framework_status: Dict[str, Dict[str, str]] = {}

        self.worker: Optional[WorkerThread] = None
        self.log_handler: Optional[QtLogHandler] = None
        self._kline_canvas: Optional[FigureCanvas] = None
        self._kline_toolbar: Optional[NavigationToolbar] = None
        self._kline_mpl_cids: list[int] = []
        self._last_backtest_result: Optional[dict] = None
        self.lstm_result_fig: Optional[Figure] = None
        self.lstm_result_canvas: Optional[FigureCanvas] = None
        self.lstm_result_toolbar: Optional[NavigationToolbar] = None
        self.lstm_result_placeholder: Optional[QLabel] = None
        self._last_lstm_prediction: Optional[Dict[str, Any]] = None
        self.log_view: Optional[QPlainTextEdit] = None
        self.status_label: Optional[QLabel] = None
        self._log_buffer: list[str] = []

        # AI 策略线程与状态
        self.ai_training_thread: Optional[PpoTrainingThread] = None
        self.ai_backtest_thread: Optional[AiBacktestThread] = None
        self.ai_model_path: Optional[Path] = None
        self.ai_benchmark_df: Optional[pd.DataFrame] = None
        self.ai_last_result: Optional[Dict[str, Any]] = None
        self.ai_comparison_result: Optional[Dict[str, Any]] = None

        self.crosshair_timer = QTimer(self)
        self.crosshair_timer.setSingleShot(True)

        self._build_ui()

        # 中文字体属性，供图例/标题明确使用，避免某些系统下的回退导致乱码
        self._chinese_font_prop = get_chinese_font_prop()

        # Startup Auto-Load
        default_path = self._default_data_path()
        if default_path:
            self.data_path_edit.setText(default_path)
            self._load_new_data_file(default_path)

    def _build_ui(self) -> None:
        """构建主用户界面。"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # 提前准备日志和状态栏，防止初始化阶段调用 append_log 时属性未定义
        self.log_view = QPlainTextEdit(central)
        self.log_view.setReadOnly(True)
        self.status_label = QLabel("准备就绪 (Ready)", central)

        # 1. 判断各框架是否可用
        self.framework_status = self._detect_framework_status()

        # 2. 创建主选项卡控件
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, stretch=5)  # 主要区域，占据更多空间

        # 创建三个核心选项卡
        self._create_chart_tab()
        self._create_training_tab()
        self._create_backtest_tab()

        # 3. 日志和状态栏 (位于选项卡外部，全局可见)
        main_layout.addWidget(self.log_view, stretch=1)  # 次要区域，占据较少空间
        main_layout.addWidget(self.status_label)
        self._flush_log_buffer()

        # 4. 收集所有可禁用的控件
        self._collect_all_controls()

        for key, info in self.framework_status.items():
            if not info["available"]:
                self.append_log(
                    f"{FRAME_DISPLAY_NAMES.get(key, key.title())} 框架不可用：{info['error']}"
                )

    def append_log(self, message: str) -> None:
        """Append a formatted log message to the GUI log view."""
        if message is None:
            return

        text = str(message).rstrip()
        if not text:
            return

        lines = text.splitlines() or [""]
        time_tag = datetime.now().strftime("%H:%M:%S")
        formatted: list[str] = []
        for line in lines:
            stripped = line.rstrip()
            if not stripped:
                formatted.append("")
                continue
            if LOG_PREFIX_PATTERN.match(stripped):
                formatted.append(stripped)
            else:
                formatted.append(f"[{time_tag}] {stripped}")

        if not formatted:
            return

        if self.log_view is None:
            self._log_buffer.extend(formatted)
            return

        self.log_view.appendPlainText("\n".join(formatted))
        scroll_bar = self.log_view.verticalScrollBar()
        if scroll_bar is not None:
            scroll_bar.setValue(scroll_bar.maximum())

    def _flush_log_buffer(self) -> None:
        if self.log_view is None or not self._log_buffer:
            return

        self.log_view.appendPlainText("\n".join(self._log_buffer))
        self._log_buffer.clear()
        scroll_bar = self.log_view.verticalScrollBar()
        if scroll_bar is not None:
            scroll_bar.setValue(scroll_bar.maximum())

    def _default_data_path(self) -> str:
        """返回默认的数据文件路径，如果存在的话。"""
        path = r"C:\Users\20577\Neverflandre\stock\data\external\tdx_newdata\sh510300.csv"
        return path if os.path.exists(path) else ""

    def _populate_device_combo(self, combo: QComboBox) -> None:
        """填充设备选择下拉框，检测可用的GPU。"""
        combo.clear()
        combo.addItem("自动 (Auto)", "auto")
        combo.addItem("CPU", "cpu")

        pytorch_gpu_count = 0
        tf_gpu_count = 0

        # 检测可用的 PyTorch GPU
        try:
            torch = importlib.import_module("torch")

            torch_version = getattr(torch, "__version__", "未知版本")
            cuda_build = getattr(torch.version, "cuda", None)
            if torch.cuda.is_available():
                pytorch_gpu_count = torch.cuda.device_count()
                for i in range(pytorch_gpu_count):
                    try:
                        gpu_name = torch.cuda.get_device_name(i)
                    except Exception:
                        gpu_name = "未知 GPU"
                    combo.addItem(f"GPU {i}: {gpu_name}", f"cuda:{i}")
                self.append_log(f"✅ PyTorch {torch_version} 检测到 {pytorch_gpu_count} 个GPU")
            else:
                reason = "CPU only"
                if cuda_build:
                    reason = f"CUDA 构建 {cuda_build} / is_available=False"
                self.append_log(
                    f"⚠️ PyTorch {torch_version} 未检测到GPU ({reason})"
                )
        except Exception as exc:
            self.append_log(f"⚠️ PyTorch GPU检测失败: {exc}")

        # 检测可用的 TensorFlow GPU
        try:
            tf = importlib.import_module("tensorflow")

            gpus = tf.config.list_physical_devices('GPU')
            tf_gpu_count = len(gpus)
            if gpus and pytorch_gpu_count == 0:
                for i, gpu in enumerate(gpus):
                    combo.addItem(f"GPU {i} (TF)", f"gpu:{i}")
                self.append_log(f"✅ TensorFlow {tf.__version__} 检测到 {len(gpus)} 个GPU")
            elif not gpus:
                self.append_log(
                    f"ℹ️ TensorFlow {tf.__version__} 未检测到GPU (is_built_with_cuda={tf.test.is_built_with_cuda()})"
                )
        except Exception as exc:
            self.append_log(f"⚠️ TensorFlow GPU检测失败: {exc}")

        if pytorch_gpu_count == 0 and tf_gpu_count == 0:
            summary, _ = self._collect_gpu_diagnostics()
            self.append_log(f"ℹ️ GPU诊断: {summary}")
            self.append_log("🛠 提示: 点击“ℹ️”按钮查看详细GPU诊断信息。")
            self.append_log("ℹ️ 未检测到可用GPU，将使用CPU训练")
    
    def _refresh_device_combo(self, combo: QComboBox) -> None:
        """刷新设备列表。"""
        current_selection = combo.currentData()
        self.append_log("🔄 刷新设备列表...")
        self._populate_device_combo(combo)
        
        # 尝试恢复之前的选择
        for i in range(combo.count()):
            if combo.itemData(i) == current_selection:
                combo.setCurrentIndex(i)
                break
        
        self.append_log(f"✅ 设备列表已刷新，共 {combo.count()} 个选项")

    def _collect_gpu_diagnostics(self) -> tuple[str, str]:
        """收集当前环境的 GPU 诊断信息。"""

        summary_parts = []
        details: list[str] = []

        details.append(f"Python: {sys.version.split()[0]} ({sys.executable})")
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        details.append(f"CUDA_VISIBLE_DEVICES: {cuda_visible if cuda_visible is not None else '未设置'}")

        # PyTorch 信息
        try:
            torch = importlib.import_module("torch")

            torch_version = getattr(torch, "__version__", "未知版本")
            cuda_build = getattr(torch.version, "cuda", None)
            cuda_available = torch.cuda.is_available()
            device_count = torch.cuda.device_count()

            summary_parts.append(
                f"PyTorch {torch_version} (CUDA: {cuda_build or 'CPU only'}, GPUs: {device_count if cuda_available else 0})"
            )
            details.extend(
                [
                    f"[PyTorch] 版本: {torch_version}",
                    f"[PyTorch] 构建 CUDA: {cuda_build or 'CPU only'}",
                    f"[PyTorch] CUDA 可用: {cuda_available}",
                    f"[PyTorch] GPU 数量: {device_count}",
                ]
            )

            if cuda_available and device_count > 0:
                for idx in range(device_count):
                    try:
                        props = torch.cuda.get_device_properties(idx)
                        mem_gb = props.total_memory / (1024 ** 3)
                        details.append(f"[PyTorch] GPU {idx}: {props.name} ({mem_gb:.1f} GB)")
                    except Exception as exc:
                        details.append(f"[PyTorch] GPU {idx} 信息读取失败: {exc}")
            else:
                if cuda_build:
                    details.append("[PyTorch] 检测到 CUDA 构建但 is_available=False，可能是驱动或权限问题。")
                else:
                    details.append("[PyTorch] 当前为 CPU 版本，需重新安装 GPU 版 PyTorch。")
        except Exception as exc:
            summary_parts.append("PyTorch: 导入失败")
            details.append(f"[PyTorch] 导入失败: {exc}")

        # TensorFlow 信息
        try:
            tf = importlib.import_module("tensorflow")

            tf_version = getattr(tf, "__version__", "未知版本")
            gpus = tf.config.list_physical_devices('GPU')
            summary_parts.append(f"TensorFlow {tf_version} (GPUs: {len(gpus)})")
            details.append(f"[TensorFlow] 版本: {tf_version}")
            details.append(f"[TensorFlow] 构建支持 CUDA: {tf.test.is_built_with_cuda()}")
            if gpus:
                for gpu in gpus:
                    details.append(f"[TensorFlow] GPU: {gpu.name}")
            else:
                details.append("[TensorFlow] 未检测到 GPU 设备。")
        except Exception as exc:
            summary_parts.append("TensorFlow: 导入失败")
            details.append(f"[TensorFlow] 导入失败: {exc}")

        if not summary_parts:
            summary_parts.append("未检测到深度学习框架")

        summary = " | ".join(summary_parts)
        detail_text = "\n".join(details)
        return summary, detail_text

    def _show_gpu_diagnostics(self) -> None:
        """显示 GPU 诊断信息弹窗。"""

        summary, detail_text = self._collect_gpu_diagnostics()
        self.append_log(f"🩺 GPU诊断 -> {summary}")

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("GPU 诊断")
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(summary)
        msg_box.setInformativeText("详细诊断信息请展开“详细信息”。")
        msg_box.setDetailedText(detail_text)
        msg_box.exec()

    def _on_browse(self) -> None:
        start_dir = os.path.dirname(self.data_path_edit.text()) or os.path.join(os.getcwd(), "data")
        path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", start_dir, "CSV 文件 (*.csv);;所有文件 (*.*)")
        if path:
            self.data_path_edit.setText(path)
            self._load_new_data_file(path)

    def _load_new_data_file(self, path: str) -> None:
        """加载一个全新的数据文件，重置状态并显示初始图表。"""
        self.append_log(f"开始加载新数据文件: {path}")
        try:
            self.full_df = self._load_price_dataframe(path)
        except Exception as e:
            QMessageBox.critical(self, "读取失败", f"读取数据失败：{e}")
            self.append_log(f"数据文件加载失败: {e}")
            self.full_df = None
            self.current_data_path = None
            return

        if not self._validate_dataframe(self.full_df):
            self.full_df = None
            self.current_data_path = None
            self.chart_placeholder.setText("K线/回测区 —— 数据无效或不足，无法绘制图表。")
            self._clear_kline_canvas(keep_placeholder=True)
            return

        # 重置显示窗口并绘制图表
        self.current_display_rows = 300
        self.show_kline()
        self._update_ai_date_bounds()
        self.current_data_path = path

    def _load_price_dataframe(self, path: str) -> pd.DataFrame:
        if not path:
            raise ValueError("未提供数据文件路径。")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"找不到数据文件: {path}")

        encodings = ["utf-8", "utf-8-sig", "gbk", "ansi"]
        last_error: Optional[Exception] = None
        df: Optional[pd.DataFrame] = None
        for enc in encodings:
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except UnicodeDecodeError as exc:
                last_error = exc
        if df is None:
            raise ValueError(
                "无法使用常见编码读取 CSV 文件，请确认文件编码是否为 UTF-8 或 GBK。"
                + (f" 最近的错误: {last_error}" if last_error else "")
            )

        if df.empty:
            raise ValueError("数据文件为空，无法加载。")

        rename_map: Dict[str, str] = {}
        for col in df.columns:
            alias = PRICE_COLUMN_ALIASES.get(str(col).strip().lower())
            if alias:
                rename_map[col] = alias
        if rename_map:
            df = df.rename(columns=rename_map)

        if "Date" not in df.columns:
            first_col = df.columns[0]
            if str(first_col).strip().lower() not in {"open", "high", "low", "close"}:
                df = df.rename(columns={first_col: "Date"})

        if "Date" not in df.columns:
            raise ValueError("数据文件缺少日期列（Date）。")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.loc[df["Date"].notna()].copy()
        if df.empty:
            raise ValueError("日期列全部无效，请检查数据格式。")

        for col in NUMERIC_PRICE_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df.sort_values("Date", inplace=True)
        df.drop_duplicates(subset="Date", keep="last", inplace=True)
        df.reset_index(drop=True, inplace=True)

        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].fillna(0.0)
        if "Amount" in df.columns:
            df["Amount"] = df["Amount"].fillna(0.0)

        column_names = ", ".join(str(col) for col in df.columns)
        self.append_log(f"数据文件读取成功，共 {len(df)} 行，列: {column_names}")
        return df

    def _validate_dataframe(self, df: Optional[pd.DataFrame]) -> bool:
        if df is None or df.empty:
            self.append_log("数据验证失败：数据为空。")
            return False

        required = ["Date", "Open", "High", "Low", "Close"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            self.append_log(f"数据验证失败：缺少必要列 {missing}")
            return False

        if df["Date"].isna().any():
            self.append_log("数据验证失败：存在无法解析的日期。")
            return False

        numeric_checks = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
        for col in numeric_checks:
            if not pd.api.types.is_numeric_dtype(df[col]):
                self.append_log(f"数据验证失败：列 {col} 不是数值类型。")
                return False

        if len(df) < 50:
            self.append_log(
                f"⚠️ 数据行数仅 {len(df)} 行，图表与训练可能不稳定。建议使用更多历史数据。"
            )

        return True

    @staticmethod
    def compute_kdj(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 9,
        k_smooth: int = 3,
        d_smooth: int = 3,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        high = pd.to_numeric(high, errors="coerce").ffill().bfill()
        low = pd.to_numeric(low, errors="coerce").ffill().bfill()
        close = pd.to_numeric(close, errors="coerce").ffill().bfill()

        low_min = low.rolling(window=period, min_periods=1).min()
        high_max = high.rolling(window=period, min_periods=1).max()
        denominator = high_max - low_min
        rsv = (close - low_min) / denominator.replace(0, np.nan) * 100
        rsv = rsv.clip(lower=-1e6, upper=1e6).fillna(0.0)

        k = rsv.ewm(alpha=1 / k_smooth, adjust=False).mean()
        d = k.ewm(alpha=1 / d_smooth, adjust=False).mean()
        j = 3 * k - 2 * d

        return k.fillna(0.0), d.fillna(0.0), j.fillna(0.0)

    def _update_ai_date_bounds(self) -> None:
        if self.full_df is None or self.full_df.empty:
            return

        if 'Date' not in self.full_df.columns:
            return

        dates = pd.to_datetime(self.full_df['Date'], errors='coerce').dropna()
        if dates.empty:
            return

        min_date = dates.min().date()
        max_date = dates.max().date()
        qmin = QDate(min_date.year, min_date.month, min_date.day)
        qmax = QDate(max_date.year, max_date.month, max_date.day)

        for widget in (self.ai_train_start, self.ai_train_end, self.ai_test_start, self.ai_test_end):
            widget.setMinimumDate(qmin)
            widget.setMaximumDate(qmax)

        sorted_dates = dates.sort_values()
        candidate_split = sorted_dates.iloc[-1] - pd.DateOffset(years=2)

        if candidate_split <= sorted_dates.iloc[0]:
            # 回退到 70/30 拆分
            median_idx = int(len(sorted_dates) * 0.7)
            median_idx = min(max(median_idx, 0), len(sorted_dates) - 1)
            split_date = sorted_dates.iloc[median_idx].date()
            qsplit = QDate(split_date.year, split_date.month, split_date.day)

            self.ai_train_start.setDate(qmin)
            self.ai_train_end.setDate(qsplit)
            self.ai_test_start.setDate(qsplit.addDays(1))
            self.ai_test_end.setDate(qmax)
        else:
            test_start = sorted_dates[sorted_dates >= candidate_split].iloc[0].date()
            train_end = test_start - timedelta(days=1)
            if train_end < min_date:
                train_end = min_date

            qtrain_end = QDate(train_end.year, train_end.month, train_end.day)
            qtest_start = QDate(test_start.year, test_start.month, test_start.day)

            self.ai_train_start.setDate(qmin)
            self.ai_train_end.setDate(qtrain_end)
            self.ai_test_start.setDate(qtest_start)
            self.ai_test_end.setDate(qmax)
            self.append_log(
                "AI 日期拆分: 默认使用历史至两年前作为训练集，最近两年作为测试集。"
            )

    def show_kline(self) -> None:
        """根据当前状态(full_df, current_display_rows)绘制K线图。"""
        if self.full_df is None:
            # This can happen if the initial load failed.
            # We don't show a message box here as one would have been shown already.
            self.append_log("绘图失败：没有已加载的有效数据。")
            return

        self.append_log(f"准备使用全部 {len(self.full_df)} 行数据绘制图表...")

        # 使用完整数据进行绘图，以便平移
        df = self.full_df.copy()
        df = df.set_index('Date')

        self.append_log("计算指标并准备绘图...")
        k, d, j = self.compute_kdj(df['High'], df['Low'], df['Close'])
        df['K'] = k
        df['D'] = d
        df['J'] = j

        ap = [
            mpf.make_addplot(df['K'], panel=2, color='fuchsia', ylabel='KDJ'),
            mpf.make_addplot(df['D'], panel=2, color='b'),
            mpf.make_addplot(df['J'], panel=2, color='g'),
        ]

        mc = mpf.make_marketcolors(up='r', down='g', edge='i', wick='i', volume='in')
        s = mpf.make_mpf_style(
            base_mpf_style='yahoo',
            marketcolors=mc,
            rc=get_chinese_rc_params(),
        )

        if 'Amount' in df.columns:
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
            df['AmountWan'] = df['Amount'] / 10000.0

        mav = (5, 10, 20)

        try:
            self.append_log("调用 mplfinance.plot()...")
            fig, axes = mpf.plot(
                df,
                type='candle',
                mav=mav,
                volume=True,
                addplot=ap,
                style=s,
                returnfig=True,
                figscale=1.0,
                tight_layout=True,
                datetime_format='%Y/%m/%d',
                panel_ratios=(6, 1, 1),
                warn_too_much_data=99999,
            )
            self.append_log("mplfinance.plot() 调用成功。")
        except Exception as e:
            self.append_log(f"!!! K线图绘制失败: {e}")
            QMessageBox.critical(self, "绘图失败", f"K线图绘制失败，可能是由于数据格式问题。详情请见日志。\n\n{e}")
            self.chart_placeholder.setText("K线/回测区 —— 绘图失败，请检查数据格式。")
            self._clear_kline_canvas(keep_placeholder=True)
            return

        if isinstance(axes, (list, tuple)) and len(axes) >= 2:
            vol_ax = axes[1]
        elif isinstance(axes, np.ndarray) and axes.size >= 2:
            vol_ax = axes.flat[1]
        else:
            vol_ax = None

        if isinstance(axes, (list, tuple)) and axes:
            main_ax = axes[0]
        elif isinstance(axes, np.ndarray) and axes.size:
            main_ax = axes.flat[0]
        else:
            main_ax = fig.axes[0] if fig.axes else None

        if main_ax is not None:
            if getattr(self, '_chinese_font_prop', None) is not None:
                main_ax.set_ylabel('价格 (元)', fontproperties=self._chinese_font_prop)
                main_ax.set_xlabel('日期', fontproperties=self._chinese_font_prop)
            else:
                main_ax.set_ylabel('价格 (元)')
                main_ax.set_xlabel('日期')

        if vol_ax is not None:
            if getattr(self, '_chinese_font_prop', None) is not None:
                vol_ax.set_ylabel('成交量 (手)', fontproperties=self._chinese_font_prop)
            else:
                vol_ax.set_ylabel('成交量 (手)')

        if isinstance(axes, (list, tuple)):
            target_axes = [ax for ax in axes if ax is not None]
        elif isinstance(axes, np.ndarray):
            target_axes = [ax for ax in axes.flatten() if ax is not None]
        else:
            target_axes = fig.axes
        for ax in target_axes:
            ax.tick_params(axis='x', rotation=0)


        self.append_log("将图表添加到GUI布局...")
        self._clear_kline_canvas()
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, self.kline_widget)
        self.kline_layout.addWidget(canvas)
        self.kline_layout.addWidget(toolbar)
        self.kline_layout.addWidget(self.chart_info_label)
        self.chart_info_label.setText("提示：鼠标移动查看报价，Ctrl+滚轮缩放，拖动平移。")
        self._kline_canvas = canvas
        self._kline_toolbar = toolbar
        self._last_backtest_result = None
        self.export_backtest_button.setEnabled(False)
        self._setup_crosshair(canvas, axes, df)
        self._setup_ctrl_zoom(canvas, target_axes, df.index)
        canvas.draw_idle()

        # --- 设置初始显示范围为最后300个数据点 ---
        if main_ax is not None:
            total_points = len(df)
            initial_view_points = 300
            start_point = max(0, total_points - initial_view_points)
            
            # 为所有子图设置x轴范围
            for ax in target_axes:
                ax.set_xlim(start_point, total_points)
            canvas.draw_idle() # 再次绘制以应用范围



    def _create_chart_tab(self) -> None:
        """创建“图表分析”选项卡"""
        chart_tab = QWidget()
        self.tabs.addTab(chart_tab, "图表分析 (Chart Analysis)")
        layout = QVBoxLayout(chart_tab)

        # 数据源选择
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("数据文件 (Data File):"))
        self.data_path_edit = QLineEdit()
        self.browse_btn = QPushButton("浏览... (Browse...)")
        self.browse_btn.clicked.connect(self._on_browse)
        path_layout.addWidget(self.data_path_edit)
        path_layout.addWidget(self.browse_btn)
        layout.addLayout(path_layout)

        # 图表显示区域
        self.kline_widget = QWidget()
        self.kline_layout = QVBoxLayout(self.kline_widget)
        self.kline_layout.setContentsMargins(0, 0, 0, 0)
        self.chart_placeholder = QLabel(
            "图表分析区 - 请选择数据文件以自动加载图表\nChart Analysis Area - Select a data file to automatically load the chart"
        )
        self.chart_placeholder.setAlignment(Qt.AlignCenter)
        self.chart_placeholder.setStyleSheet("color: #888888;")
        self.kline_layout.addWidget(self.chart_placeholder)
        self.chart_info_label = QLabel(" ")
        self.chart_info_label.setStyleSheet("color: #555555;")
        self.kline_layout.addWidget(self.chart_info_label)
        layout.addWidget(self.kline_widget, stretch=1)

    def _create_training_tab(self) -> None:
        """创建“模型训练”选项卡"""
        train_tab = QWidget()
        self.tabs.addTab(train_tab, "模型训练 (Model Studio)")
        layout = QVBoxLayout(train_tab)

        form_layout = QFormLayout()

        self.frame_combo = QComboBox()
        self._populate_framework_combo(self.frame_combo)
        self._select_first_available_framework(self.frame_combo)

        self.device_combo = QComboBox()
        self._populate_device_combo(self.device_combo)
        
        # 添加设备刷新按钮
        self.device_refresh_btn = QPushButton("🔄")
        self.device_refresh_btn.setMaximumWidth(40)
        self.device_refresh_btn.setToolTip("刷新GPU列表")
        self.device_refresh_btn.clicked.connect(lambda: self._refresh_device_combo(self.device_combo))

        self.device_diag_btn = QPushButton("ℹ️")
        self.device_diag_btn.setMaximumWidth(40)
        self.device_diag_btn.setToolTip("查看GPU诊断信息")
        self.device_diag_btn.clicked.connect(self._show_gpu_diagnostics)
        
        device_widget = QWidget()
        device_layout = QHBoxLayout(device_widget)
        device_layout.setContentsMargins(0, 0, 0, 0)
        device_layout.addWidget(self.device_combo)
        device_layout.addWidget(self.device_refresh_btn)
        device_layout.addWidget(self.device_diag_btn)

        self.train_check = QCheckBox("训练 (Train)")
        self.train_check.setChecked(True)
        self.predict_check = QCheckBox("预测 (Predict)")
        self.predict_check.setChecked(True)

        task_widget = QWidget()
        task_layout = QHBoxLayout(task_widget)
        task_layout.setContentsMargins(0, 0, 0, 0)
        task_layout.addWidget(self.train_check)
        task_layout.addWidget(self.predict_check)
        task_layout.addStretch()

        self.time_step_spin = QSpinBox()
        self.time_step_spin.setRange(1, 2_000)
        self.time_step_spin.setValue(20)
        self._prepare_spinbox(self.time_step_spin, "输入1-2000")

        self.predict_day_spin = QSpinBox()
        self.predict_day_spin.setRange(1, 365)
        self.predict_day_spin.setValue(1)
        self._prepare_spinbox(self.predict_day_spin, "预测天数 1-365")

        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1, 100_000)
        self.epoch_spin.setSingleStep(10)
        self.epoch_spin.setValue(20)
        self._prepare_spinbox(self.epoch_spin, "训练轮次（可直接输入）")

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 16_384)
        self.batch_spin.setSingleStep(8)
        self.batch_spin.setValue(64)
        self._prepare_spinbox(self.batch_spin, "Batch Size 支持自定义")

        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setDecimals(6)
        self.learning_rate_spin.setRange(0.000001, 10.0)
        self.learning_rate_spin.setSingleStep(0.0001)
        self.learning_rate_spin.setValue(0.001)
        self._prepare_spinbox(self.learning_rate_spin, "学习率，可直接键入 1e-6~10")

        self.train_rate_spin = QDoubleSpinBox()
        self.train_rate_spin.setDecimals(4)
        self.train_rate_spin.setRange(0.0, 0.99)
        self.train_rate_spin.setSingleStep(0.01)
        self.train_rate_spin.setValue(0.80)
        self._prepare_spinbox(self.train_rate_spin, "训练集比例 0~0.99")

        self.valid_rate_spin = QDoubleSpinBox()
        self.valid_rate_spin.setDecimals(4)
        self.valid_rate_spin.setRange(0.0, 0.99)
        self.valid_rate_spin.setSingleStep(0.01)
        self.valid_rate_spin.setValue(0.15)
        self._prepare_spinbox(self.valid_rate_spin, "验证集比例 0~0.99")

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 1_000_000_000)
        self.seed_spin.setValue(42)
        self._prepare_spinbox(self.seed_spin, "随机种子，可键入任意整数")

        form_layout.addRow("框架 (Framework):", self.frame_combo)
        form_layout.addRow("设备 (Device):", device_widget)
        form_layout.addRow("任务 (Task):", task_widget)
        form_layout.addRow("时间步 (Time Step):", self.time_step_spin)
        form_layout.addRow("预测天数 (Predict Days):", self.predict_day_spin)
        form_layout.addRow("Epochs:", self.epoch_spin)
        form_layout.addRow("Batch Size:", self.batch_spin)
        form_layout.addRow("学习率 (Learning Rate):", self.learning_rate_spin)
        form_layout.addRow("训练集比例 (Train Ratio):", self.train_rate_spin)
        form_layout.addRow("验证集比例 (Valid Ratio):", self.valid_rate_spin)
        form_layout.addRow("随机种子 (Random Seed):", self.seed_spin)
        layout.addLayout(form_layout)

        self.run_button = QPushButton("开始运行 (Start Run)")
        self.run_button.clicked.connect(self.start_run)
        layout.addWidget(self.run_button)

        self.lstm_result_group = QGroupBox("预测效果 (Actual vs Predicted)")
        lstm_plot_layout = QVBoxLayout(self.lstm_result_group)
        self.lstm_result_fig = Figure(figsize=(8, 3.5))
        self.lstm_result_canvas = FigureCanvas(self.lstm_result_fig)
        self.lstm_result_toolbar = NavigationToolbar(self.lstm_result_canvas, self.lstm_result_group)
        self.lstm_result_placeholder = QLabel("训练完成后，将在此展示预测曲线。")
        self.lstm_result_placeholder.setAlignment(Qt.AlignCenter)
        self.lstm_result_placeholder.setStyleSheet("color: #777777;")
        lstm_plot_layout.addWidget(self.lstm_result_toolbar)
        lstm_plot_layout.addWidget(self.lstm_result_canvas)
        lstm_plot_layout.addWidget(self.lstm_result_placeholder)
        layout.addWidget(self.lstm_result_group)

        self._set_lstm_plot_visible(False, "训练完成后，将在此展示预测曲线。")
        layout.addStretch()

    def _create_backtest_tab(self) -> None:
        """创建“策略回测”选项卡"""
        backtest_tab = QWidget()
        self.tabs.addTab(backtest_tab, "策略回测 (Backtest Engine)")
        layout = QVBoxLayout(backtest_tab)

        # --- 占位符，未来实现网格策略时将替换为真实控件 ---
        # --- Placeholder, will be replaced with real controls for grid strategy ---
        form_layout = QFormLayout()
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["均线策略 (MA Crossover)", "网格策略 (Grid Trading)"])
        form_layout.addRow("选择策略 (Strategy):", self.strategy_combo)

        # --- Grid Trading Strategy Parameters ---
        self.grid_initial_cash_spin = QDoubleSpinBox()
        self.grid_initial_cash_spin.setRange(0.0, 1_000_000_000.0)
        self.grid_initial_cash_spin.setDecimals(2)
        self.grid_initial_cash_spin.setSingleStep(1_000.0)
        self.grid_initial_cash_spin.setValue(100000.0)
        self._prepare_spinbox(self.grid_initial_cash_spin, "初始资金，可直接输入")
        form_layout.addRow("初始资金 (Initial Cash):", self.grid_initial_cash_spin)

        self.grid_fee_spin = QDoubleSpinBox()
        self.grid_fee_spin.setRange(0.0, 1.0)
        self.grid_fee_spin.setDecimals(5)
        self.grid_fee_spin.setSingleStep(0.0001)
        self.grid_fee_spin.setValue(0.001)
        self._prepare_spinbox(self.grid_fee_spin, "手续费比例 0~1")
        form_layout.addRow("交易手续费 (Fee):", self.grid_fee_spin)

        self.grid_interval_percent_spin = QDoubleSpinBox()
        self.grid_interval_percent_spin.setRange(0.0001, 1.0)
        self.grid_interval_percent_spin.setDecimals(4)
        self.grid_interval_percent_spin.setSingleStep(0.001)
        self.grid_interval_percent_spin.setValue(0.01)
        self._prepare_spinbox(self.grid_interval_percent_spin, "网格间距 0.0001~1")
        form_layout.addRow("网格间距 (%) (Grid Interval %):", self.grid_interval_percent_spin)

        self.grid_num_lower_spin = QSpinBox()
        self.grid_num_lower_spin.setRange(1, 200)
        self.grid_num_lower_spin.setValue(5)
        self._prepare_spinbox(self.grid_num_lower_spin, "下方网格数 1-200")
        form_layout.addRow("下方网格数量 (Lower Grids):", self.grid_num_lower_spin)

        self.grid_num_upper_spin = QSpinBox()
        self.grid_num_upper_spin.setRange(1, 200)
        self.grid_num_upper_spin.setValue(5)
        self._prepare_spinbox(self.grid_num_upper_spin, "上方网格数 1-200")
        form_layout.addRow("上方网格数量 (Upper Grids):", self.grid_num_upper_spin)

        self.grid_order_size_spin = QDoubleSpinBox()
        self.grid_order_size_spin.setRange(0.0, 100_000_000.0)
        self.grid_order_size_spin.setDecimals(2)
        self.grid_order_size_spin.setSingleStep(100.0)
        self.grid_order_size_spin.setValue(1000.0)
        self._prepare_spinbox(self.grid_order_size_spin, "单笔金额，可自定义")
        form_layout.addRow("单笔订单金额 (Order Size):", self.grid_order_size_spin)
        # --- End Grid Trading Strategy Parameters ---

        layout.addLayout(form_layout)

        self.backtest_button = QPushButton("开始回测 (Run Backtest)")
        self.backtest_button.clicked.connect(self.show_backtest) # 复用旧的函数名，但现在它只负责回测
        layout.addWidget(self.backtest_button)

        self.export_backtest_button = QPushButton("导出回测报告 (Export Report)")
        self.export_backtest_button.clicked.connect(self.export_backtest_report)
        self.export_backtest_button.setEnabled(False)
        layout.addWidget(self.export_backtest_button)

        # === AI 动态策略 ===
        ai_group = QGroupBox("AI 动态策略对比 (Reinforcement Learning)")
        ai_form = QFormLayout(ai_group)

        self.ai_framework_combo = QComboBox()
        self._populate_framework_combo(self.ai_framework_combo)
        self._select_first_available_framework(self.ai_framework_combo)

        today = QDate.currentDate()
        self.ai_train_start = QDateEdit(calendarPopup=True)
        self.ai_train_end = QDateEdit(calendarPopup=True)
        self.ai_test_start = QDateEdit(calendarPopup=True)
        self.ai_test_end = QDateEdit(calendarPopup=True)
        for widget in (self.ai_train_start, self.ai_train_end, self.ai_test_start, self.ai_test_end):
            widget.setDisplayFormat("yyyy-MM-dd")
            widget.setDate(today)

        self.ai_monthly_spin = QDoubleSpinBox()
        self.ai_monthly_spin.setRange(0.0, 10_000_000_000.0)
        self.ai_monthly_spin.setDecimals(2)
        self.ai_monthly_spin.setSingleStep(100.0)
        self.ai_monthly_spin.setValue(2000.0)
        self._prepare_spinbox(self.ai_monthly_spin, "月度注资，可直接输入")

        self.ai_initial_cash_spin = QDoubleSpinBox()
        self.ai_initial_cash_spin.setRange(0.0, 10_000_000_000.0)
        self.ai_initial_cash_spin.setDecimals(2)
        self.ai_initial_cash_spin.setSingleStep(1_000.0)
        self.ai_initial_cash_spin.setValue(100000.0)
        self._prepare_spinbox(self.ai_initial_cash_spin, "初始资金，可直接输入")

        # AI 训练设备选择
        self.ai_device_combo = QComboBox()
        self._populate_device_combo(self.ai_device_combo)
        self.ai_device_combo.setToolTip("选择训练使用的设备：自动、CPU或强制使用特定GPU")
        
        # AI 训练设备刷新按钮
        self.ai_device_refresh_btn = QPushButton("🔄")
        self.ai_device_refresh_btn.setMaximumWidth(40)
        self.ai_device_refresh_btn.setToolTip("刷新GPU列表")
        self.ai_device_refresh_btn.clicked.connect(lambda: self._refresh_device_combo(self.ai_device_combo))

        self.ai_device_diag_btn = QPushButton("ℹ️")
        self.ai_device_diag_btn.setMaximumWidth(40)
        self.ai_device_diag_btn.setToolTip("查看GPU诊断信息")
        self.ai_device_diag_btn.clicked.connect(self._show_gpu_diagnostics)
        
        ai_device_widget = QWidget()
        ai_device_layout = QHBoxLayout(ai_device_widget)
        ai_device_layout.setContentsMargins(0, 0, 0, 0)
        ai_device_layout.addWidget(self.ai_device_combo)
        ai_device_layout.addWidget(self.ai_device_refresh_btn)
        ai_device_layout.addWidget(self.ai_device_diag_btn)

        # AI 训练轮次手动设置
        self.ai_epoch_spin = QSpinBox()
        self.ai_epoch_spin.setRange(1, 200_000)
        self.ai_epoch_spin.setSingleStep(10)
        self.ai_epoch_spin.setValue(100)
        self.ai_epoch_spin.setToolTip("设置训练轮次(Epoch)，建议100-500，可扩大范围")
        self._prepare_spinbox(self.ai_epoch_spin, "AI Epoch 支持自定义输入")

        self.ai_total_timesteps_spin = QSpinBox()
        self.ai_total_timesteps_spin.setRange(100, 10_000_000)
        self.ai_total_timesteps_spin.setSingleStep(1000)
        self.ai_total_timesteps_spin.setValue(200000)
        self.ai_total_timesteps_spin.setToolTip("设置总训练步数，此参数是训练时长的主要决定因素。")
        self._prepare_spinbox(self.ai_total_timesteps_spin, "例如: 2500 用于快速测试")

        self.ai_hyperparam_label = QLabel("其他超参数将根据历史波动自动设定。")
        self.ai_hyperparam_label.setStyleSheet("color: #666666;")
        self.ai_hyperparam_label.setWordWrap(True)

        self.ai_fee_hint_label = QLabel("手续费固定为千分之一，可在配置文件中进一步调整。")
        self.ai_fee_hint_label.setStyleSheet("color: #888888;")
        self.ai_fee_hint_label.setWordWrap(True)

        self.ai_benchmark_path_edit = QLineEdit()
        self.ai_benchmark_browse_btn = QPushButton("浏览基准... (Benchmark)")
        self.ai_benchmark_browse_btn.clicked.connect(self._on_ai_benchmark_browse)
        benchmark_box = QWidget()
        benchmark_layout = QHBoxLayout(benchmark_box)
        benchmark_layout.setContentsMargins(0, 0, 0, 0)
        benchmark_layout.addWidget(self.ai_benchmark_path_edit)
        benchmark_layout.addWidget(self.ai_benchmark_browse_btn)

        self.ai_train_button = QPushButton("训练 AI 策略")
        self.ai_train_button.clicked.connect(self._start_ai_training)
        self.ai_backtest_button = QPushButton("AI 策略回测对比")
        self.ai_backtest_button.clicked.connect(self._start_ai_backtest)

        self.ai_artifact_label = QLabel("尚未训练 AI 模型")
        self.ai_artifact_label.setStyleSheet("color: #666666;")

        ai_form.addRow("框架 (Framework):", self.ai_framework_combo)
        ai_form.addRow("训练设备 (Device):", ai_device_widget)
        ai_form.addRow("训练轮次 (Epochs):", self.ai_epoch_spin)
        ai_form.addRow("总训练步数 (Timesteps):", self.ai_total_timesteps_spin)
        ai_form.addRow("训练开始 (Train From):", self.ai_train_start)
        ai_form.addRow("训练结束 (Train To):", self.ai_train_end)
        ai_form.addRow("测试开始 (Test From):", self.ai_test_start)
        ai_form.addRow("测试结束 (Test To):", self.ai_test_end)
        ai_form.addRow("初始资金 (Initial Cash):", self.ai_initial_cash_spin)
        ai_form.addRow("月度注资 (Monthly Invest):", self.ai_monthly_spin)
        ai_form.addRow("自动调参:", self.ai_hyperparam_label)
        ai_form.addRow("手续费说明:", self.ai_fee_hint_label)
        ai_form.addRow("基准数据 (Benchmark):", benchmark_box)

        ai_button_box = QWidget()
        ai_button_layout = QHBoxLayout(ai_button_box)
        ai_button_layout.setContentsMargins(0, 0, 0, 0)
        ai_button_layout.addWidget(self.ai_train_button)
        ai_button_layout.addWidget(self.ai_backtest_button)
        ai_form.addRow(ai_button_box)
        ai_form.addRow("当前模型:", self.ai_artifact_label)

        self.ai_progress = QProgressBar()
        self.ai_progress.setRange(0, 1)
        self.ai_progress.setValue(0)
        self.ai_progress.setFormat("闲置")
        self.ai_progress.setTextVisible(True)
        ai_form.addRow("任务进度:", self.ai_progress)

        layout.addWidget(ai_group)

        layout.addStretch()

        # Connect strategy combo to toggle parameter visibility
        self.strategy_combo.currentIndexChanged.connect(self._toggle_grid_params_visibility)
        # Initial call to set correct visibility
        self._toggle_grid_params_visibility()

    def _toggle_grid_params_visibility(self) -> None:
        selected_strategy = self.strategy_combo.currentText()
        is_grid_strategy = "网格策略" in selected_strategy

        self.grid_initial_cash_spin.setVisible(is_grid_strategy)
        self.grid_fee_spin.setVisible(is_grid_strategy)
        self.grid_interval_percent_spin.setVisible(is_grid_strategy)
        self.grid_num_lower_spin.setVisible(is_grid_strategy)
        self.grid_num_upper_spin.setVisible(is_grid_strategy)
        self.grid_order_size_spin.setVisible(is_grid_strategy)

        # Also toggle labels for these controls
        # This assumes the labels are directly associated with the widgets in the form layout
        # A more robust way would be to store references to the labels themselves.
        form_layout = self.strategy_combo.parentWidget().layout() # Get the form layout
        if isinstance(form_layout, QFormLayout):
            for i in range(form_layout.rowCount()):
                label_item = form_layout.itemAt(i, QFormLayout.LabelRole)
                field_item = form_layout.itemAt(i, QFormLayout.FieldRole)
                
                if field_item and field_item.widget() in [
                    self.grid_initial_cash_spin,
                    self.grid_fee_spin,
                    self.grid_interval_percent_spin,
                    self.grid_num_lower_spin,
                    self.grid_num_upper_spin,
                    self.grid_order_size_spin,
                ]:
                    if label_item and label_item.widget():
                        label_item.widget().setVisible(is_grid_strategy)

    def _detect_framework_status(self) -> Dict[str, Dict[str, str]]:
        status: Dict[str, Dict[str, str]] = {}
        for key, module_path in FRAME_MODULES.items():
            try:
                importlib.import_module(module_path)
                status[key] = {"available": True, "error": ""}
            except Exception as exc:  # noqa: BLE001
                status[key] = {"available": False, "error": str(exc)}
        return status

    def _populate_framework_combo(self, combo: QComboBox) -> None:
        combo.clear()
        for key, label in FRAME_DISPLAY_NAMES.items():
            info = self.framework_status.get(key, {"available": True, "error": ""})
            display_label = label if info["available"] else f"{label} (不可用)"
            combo.addItem(display_label, key)
            item = combo.model().item(combo.count() - 1)
            if item is not None and not info["available"]:
                item.setEnabled(False)
                item.setToolTip(info["error"])

    def _select_first_available_framework(self, combo: QComboBox) -> None:
        for idx in range(combo.count()):
            key = combo.itemData(idx)
            if self.framework_status.get(key, {"available": True}).get("available", True):
                combo.setCurrentIndex(idx)
                return
        if combo.count():
            combo.setCurrentIndex(0)

    def _build_pipeline_config(self, data_path: str) -> Config:
        framework_key = self.frame_combo.currentData() or self.frame_combo.currentText().lower()
        if not framework_key:
            raise ValueError("请选择需要使用的深度学习框架。")

        info = self.framework_status.get(framework_key, {"available": True, "error": ""})
        if not info["available"]:
            raise ValueError(
                f"{FRAME_DISPLAY_NAMES.get(framework_key, framework_key.title())} 不可用：{info['error']}"
            )

        config = Config(used_frame=str(framework_key))
        config.train_data_path = data_path
        config.do_train = self.train_check.isChecked()
        config.do_predict = self.predict_check.isChecked()
        if not (config.do_train or config.do_predict):
            raise ValueError("请至少勾选训练或预测任务中的一项。")

        config.time_step = int(self.time_step_spin.value())
        config.predict_day = int(self.predict_day_spin.value())
        config.epoch = int(self.epoch_spin.value())
        config.batch_size = int(self.batch_spin.value())
        config.learning_rate = float(self.learning_rate_spin.value())
        train_ratio = float(self.train_rate_spin.value())
        valid_ratio = float(self.valid_rate_spin.value())
        if train_ratio + valid_ratio >= 0.99:
            raise ValueError("训练集比例与验证集比例之和需小于 0.99，以保留充足的测试集。")
        config.train_data_rate = train_ratio
        config.valid_data_rate = valid_ratio
        config.random_seed = int(self.seed_spin.value())

        # 设备选择：从GUI device_combo获取
        device_data = self.device_combo.currentData()
        if device_data is None:
            device_pref = 'auto'
        else:
            device_pref = device_data
        # 将设备偏好存储到config（需要在Config类中支持此属性）
        setattr(config, 'device_preference', device_pref)
        self.append_log(f"LSTM训练设备设置: {device_pref}")

        # GUI 模式下禁用阻塞式图表和标准输出日志
        config.show_plots = False
        config.do_figure_save = False
        config.do_train_visualized = False
        config.do_log_print_to_screen = False

        return config

    def start_run(self) -> None:
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "任务运行中", "已有训练/预测任务正在执行，请等待其完成。")
            return
        if self.ai_training_thread and self.ai_training_thread.isRunning():
            QMessageBox.warning(self, "AI 训练进行中", "请等待 AI 策略训练结束后再执行此操作。")
            return
        if self.ai_backtest_thread and self.ai_backtest_thread.isRunning():
            QMessageBox.warning(self, "AI 回测进行中", "请等待 AI 策略回测完成后再执行此操作。")
            return

        data_path = self.data_path_edit.text().strip()
        if not data_path:
            QMessageBox.warning(self, "缺少数据", "请先在图表分析页选择一个 CSV 数据文件。")
            return
        if not os.path.isfile(data_path):
            QMessageBox.warning(self, "无效路径", f"找不到数据文件：{data_path}")
            return

        try:
            config = self._build_pipeline_config(data_path)
        except ValueError as exc:
            QMessageBox.warning(self, "参数错误", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "配置失败", f"构建训练配置时出现异常：{exc}")
            self.append_log(f"配置构建失败: {exc}")
            return

        # 确保界面上的行情数据与训练文件同步
        if self.current_data_path != data_path:
            self.data_path_edit.setText(data_path)
            self._load_new_data_file(data_path)
            if self.full_df is None:
                return

        if self.log_handler is not None:
            try:
                self.log_handler.signal.message.disconnect(self.append_log)
            except TypeError:
                pass
            self.log_handler = None

        handler = QtLogHandler()
        handler.signal.message.connect(self.append_log)
        self.log_handler = handler

        self._last_lstm_prediction = None
        self._set_lstm_plot_visible(False, "训练/预测任务执行中，请稍候…")

        frame_name = config.used_frame.upper()
        tasks = []
        if config.do_train:
            tasks.append("训练")
        if config.do_predict:
            tasks.append("预测")
        task_label = "/".join(tasks)
        self.append_log(f"启动 LSTM 工作流 ({task_label})，框架={frame_name}，数据源={os.path.basename(data_path)}")
        self.status_label.setText("LSTM 训练/预测任务运行中...")
        self._set_controls_enabled(False)

        self.worker = WorkerThread(config, handler)
        self.worker.succeeded.connect(self._on_worker_success)
        self.worker.failed.connect(self._on_worker_failed)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _collect_all_controls(self) -> None:
        """收集所有需要统一禁用/启用的控件"""
        self._controls = [
            # Chart Tab
            self.data_path_edit,
            self.browse_btn,
            # Training Tab
            self.frame_combo,
            self.device_combo,
            self.train_check,
            self.predict_check,
            self.time_step_spin,
            self.predict_day_spin,
            self.epoch_spin,
            self.batch_spin,
            self.learning_rate_spin,
            self.train_rate_spin,
            self.valid_rate_spin,
            self.seed_spin,
            self.run_button,
            # Backtest Tab
            self.strategy_combo,
            self.grid_initial_cash_spin,
            self.grid_fee_spin,
            self.grid_interval_percent_spin,
            self.grid_num_lower_spin,
            self.grid_num_upper_spin,
            self.grid_order_size_spin,
            self.backtest_button,
            self.export_backtest_button,
            # AI Strategy Controls
            self.ai_framework_combo,
            self.ai_train_start,
            self.ai_train_end,
            self.ai_test_start,
            self.ai_test_end,
            self.ai_initial_cash_spin,
            self.ai_monthly_spin,
            self.ai_benchmark_path_edit,
            self.ai_benchmark_browse_btn,
            self.ai_train_button,
            self.ai_backtest_button,
            self.ai_epoch_spin,
            self.ai_total_timesteps_spin,
        ]

    def _set_ai_controls_enabled(self, enabled: bool) -> None:
        widgets = [
            self.ai_framework_combo,
            self.ai_device_combo,
            self.ai_epoch_spin,
            self.ai_total_timesteps_spin,
            self.ai_train_start,
            self.ai_train_end,
            self.ai_test_start,
            self.ai_test_end,
            self.ai_initial_cash_spin,
            self.ai_monthly_spin,
            self.ai_benchmark_path_edit,
            self.ai_benchmark_browse_btn,
            self.ai_train_button,
            self.ai_backtest_button,
        ]
        for widget in widgets:
            widget.setEnabled(enabled)
        if enabled:
            self.export_backtest_button.setEnabled(self._last_backtest_result is not None)

    def _on_ai_benchmark_browse(self) -> None:
        start_dir = os.path.dirname(self.ai_benchmark_path_edit.text()) or os.path.join(os.getcwd(), "data")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择基准数据文件",
            start_dir,
            "CSV 文件 (*.csv);;所有文件 (*.*)",
        )
        if not path:
            return
        try:
            df = self._load_price_dataframe(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "读取失败", f"导入基准数据失败：{exc}")
            self.append_log(f"基准数据加载失败: {exc}")
            return

        self.ai_benchmark_df = df
        self.ai_benchmark_path_edit.setText(path)
        self.append_log(f"已加载基准数据: {path}")

    def _get_benchmark_series(self) -> Optional[pd.Series]:
        if self.ai_benchmark_df is None:
            return None
        df = self.ai_benchmark_df.copy()
        if 'Date' in df.columns:
            df = df.set_index('Date')
        if 'Close' not in df.columns:
            return None
        return df['Close']

    def _start_ai_training(self) -> None:
        if self.full_df is None or self.full_df.empty:
            QMessageBox.warning(self, "缺少数据", "请先在图表分析页加载目标数据。")
            return

        train_start = pd.Timestamp(self.ai_train_start.date().toPyDate())
        train_end = pd.Timestamp(self.ai_train_end.date().toPyDate())
        if train_start >= train_end:
            QMessageBox.warning(self, "日期错误", "训练开始日期必须早于结束日期。")
            return

        df_indexed = self.full_df.set_index('Date')
        df_train = df_indexed.loc[train_start:train_end]
        if df_train.empty or len(df_train) < 200:
            QMessageBox.warning(self, "样本不足", "训练区间数据量过少，至少需要 200 条记录。")
            return

        framework = self.ai_framework_combo.currentData()
        if framework is None:
            QMessageBox.warning(self, "缺少框架", "当前没有可用的深度学习框架，请检查依赖安装。")
            return
        info = self.framework_status.get(framework, {"available": True, "error": ""})
        if not info["available"]:
            QMessageBox.warning(
                self,
                "框架不可用",
                f"所选框架当前不可用，请检查依赖安装：\n{info['error']}",
            )
            return

        if framework != "pytorch":
            QMessageBox.information(
                self,
                "暂未支持",
                "当前的 PPO 训练仅支持 PyTorch，请在框架选择中选择 PyTorch。",
            )
            return

        total_timesteps = int(self.ai_total_timesteps_spin.value())
        if total_timesteps <= 0:
            QMessageBox.warning(self, "参数错误", "训练步数必须大于 0。")
            return

        device_data = self.ai_device_combo.currentData()
        device = str(device_data) if device_data is not None else "auto"

        output_dir = Path("checkpoint") / "ai_runs" / f"{datetime.now():%Y%m%d_%H%M%S}_{framework}"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_filename = "ppo_model.zip"

        framework_name = FRAME_DISPLAY_NAMES.get(framework, framework.title())
        self.ai_hyperparam_label.setText(
            f"PPO 训练参数：框架 {framework_name} | 总步数 {total_timesteps:,} | 设备 {device}"
        )

        self.append_log(
            f"启动 PPO 训练：样本 {len(df_train)} 条 | 总步数 {total_timesteps:,} | 设备 {device} | 输出目录 {output_dir}"
        )
        self.status_label.setText("AI 策略训练中...")
        self.ai_artifact_label.setText("训练进行中...")
        self._set_ai_controls_enabled(False)
        self.ai_progress.setRange(0, 0)
        self.ai_progress.setFormat("AI 训练中...")

        self.ai_training_thread = PpoTrainingThread(
            df_train=df_train.copy(),
            total_timesteps=total_timesteps,
            output_dir=output_dir,
            model_filename=model_filename,
            device=device,
        )
        self.ai_training_thread.log_message.connect(self.append_log)
        self.ai_training_thread.succeeded.connect(self._on_ai_training_success)
        self.ai_training_thread.failed.connect(self._on_ai_training_failed)
        self.ai_training_thread.start()

    def _on_ai_training_success(self, model_path: str) -> None:
        self.ai_training_thread = None
        self._set_ai_controls_enabled(True)
        self.ai_model_path = Path(model_path)
        self.status_label.setText("AI 策略训练完成")
        self.append_log(f"AI 策略训练完成，模型已保存至 {model_path}")
        self.ai_artifact_label.setText(f"模型路径: {self.ai_model_path}")
        self.ai_progress.setRange(0, 100)
        self.ai_progress.setValue(100)
        self.ai_progress.setFormat("训练完成")

    def _on_ai_training_failed(self, traceback_text: str) -> None:
        self.ai_training_thread = None
        self._set_ai_controls_enabled(True)
        self.status_label.setText("AI 策略训练失败")
        self.ai_artifact_label.setText("训练失败，请重试")
        self.append_log(traceback_text)
        QMessageBox.critical(self, "AI 训练失败", "训练过程中出现异常，详情请查看日志。")
        self.ai_progress.setRange(0, 100)
        self.ai_progress.setValue(0)
        self.ai_progress.setFormat("训练失败")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds <= 0:
            return "--"
        whole = int(seconds)
        hours, remainder = divmod(whole, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}小时{minutes:02d}分{secs:02d}秒"
        if minutes > 0:
            return f"{minutes:02d}分{secs:02d}秒"
        if secs > 0:
            return f"{secs}秒"
        return "<1秒"

    def _start_ai_backtest(self) -> None:
        if self.full_df is None or self.full_df.empty:
            QMessageBox.warning(self, "缺少数据", "请先在图表分析页加载目标数据。")
            return
        if self.ai_model_path is None or not self.ai_model_path.exists():
            QMessageBox.information(self, "缺少模型", "请先训练 AI 策略模型后再执行回测。")
            if self.ai_model_path is not None:
                self.append_log(f"提示：已记录的模型路径不存在：{self.ai_model_path}")
            return

        test_start = pd.Timestamp(self.ai_test_start.date().toPyDate())
        test_end = pd.Timestamp(self.ai_test_end.date().toPyDate())
        if test_start >= test_end:
            QMessageBox.warning(self, "日期错误", "测试开始日期必须早于结束日期。")
            return

        try:
            df_indexed = self.full_df.set_index('Date')
            df_test = df_indexed.loc[test_start:test_end]
        except Exception as e:
            QMessageBox.critical(self, "数据处理错误", f"处理测试数据时出错：{e}")
            self.append_log(f"数据处理错误: {e}")
            return
            
        if df_test.empty or len(df_test) < 60:
            QMessageBox.warning(self, "样本不足", f"测试区间数据量过少（当前{len(df_test)}条），至少需要 60 条记录。")
            self.append_log(f"测试数据: {test_start} 到 {test_end}, 共 {len(df_test)} 条记录")
            return
        
        self.append_log(f"测试数据准备完成: {len(df_test)} 条记录 ({test_start.date()} 到 {test_end.date()})")

        self.ai_comparison_result = None

        initial_cash = float(self.ai_initial_cash_spin.value())
        monthly_invest = float(self.ai_monthly_spin.value())
        fee = AI_DEFAULT_TRADE_FEE

        self.append_log(
            f"启动 AI 策略对比回测: {test_start.date()} ~ {test_end.date()} | 初始资金 {initial_cash:.2f} | 月度注资 {monthly_invest:.2f} | 手续费 {fee:.2%} | 模型 {self.ai_model_path}"
        )
        self.status_label.setText("正在执行策略对比回测...")
        self._set_ai_controls_enabled(False)
        self.ai_progress.setRange(0, 0)
        self.ai_progress.setFormat("策略对比回测中...")

        self.ai_backtest_thread = AiBacktestThread(
            self.ai_model_path,
            df_test,
            initial_cash=initial_cash,
            monthly_invest=monthly_invest,
            fee=fee,
        )
        self.ai_backtest_thread.succeeded.connect(self._on_comparison_backtest_success)
        self.ai_backtest_thread.failed.connect(self._on_comparison_backtest_failed)
        self.ai_backtest_thread.start()

    def _on_comparison_backtest_success(self, results: Dict[str, Any]) -> None:
        self.ai_backtest_thread = None
        self._set_ai_controls_enabled(True)
        self.status_label.setText("对比回测完成")
        self.ai_last_result = results
        self.ai_comparison_result = results

        self.append_log("=== 策略对比回测完成 ===")
        equity_curves = results.get("equity_curves")
        if equity_curves:
            self.append_log(f"获取到 {len(equity_curves)} 条净值曲线：{', '.join(equity_curves.keys())}")
        else:
            self.append_log("警告：回测结果中未包含净值曲线。")

        metrics = results.get("metrics")
        if metrics:
            self.append_log(f"可用指标集合：{', '.join(metrics.keys())}")
        else:
            self.append_log("警告：回测结果中未包含指标数据。")

        self.ai_progress.setRange(0, 1)
        self.ai_progress.setValue(1)
        self.ai_progress.setFormat("回测完成")

        try:
            self._display_comparison_result(results)
        except Exception as exc:
            self.append_log(f"显示回测结果时出错: {exc}")
            self.append_log(traceback.format_exc())
            QMessageBox.critical(self, "显示错误", f"绘制策略对比曲线时出错：{exc}")

    def _on_comparison_backtest_failed(self, traceback_text: str) -> None:
        self.ai_backtest_thread = None
        self._set_ai_controls_enabled(True)
        self.status_label.setText("策略对比回测失败")
        self.append_log(traceback_text)
        QMessageBox.critical(self, "策略对比回测失败", "回测过程中出现异常，详情请查看日志。")
        self.ai_progress.setRange(0, 1)
        self.ai_progress.setValue(0)
        self.ai_progress.setFormat("回测失败")

    def _display_comparison_result(self, results: Dict[str, Any]) -> None:
        equity_curves = results.get("equity_curves") or {}
        if not equity_curves:
            raise ValueError("缺少净值曲线数据，无法绘制比较图。")
        
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        color_cycle = ['#d62728', '#2ca02c', '#1f77b4', '#9467bd', '#ff7f0e']

        for idx, (name, curve) in enumerate(equity_curves.items()):
            if curve is None:
                continue
            if isinstance(curve, pd.DataFrame):
                for col in curve.columns:
                    series = curve[col].astype(float)
                    ax.plot(series.index, series.values, label=f"{name}-{col}", color=color_cycle[idx % len(color_cycle)])
            else:
                series = pd.Series(curve)
                series = series.astype(float)
                if not isinstance(series.index, pd.DatetimeIndex):
                    series.index = pd.RangeIndex(start=0, stop=len(series))
                ax.plot(series.index, series.values, label=name, color=color_cycle[idx % len(color_cycle)])

        if getattr(self, '_chinese_font_prop', None) is not None:
            ax.set_title('AI 动态策略 vs 基准策略 净值对比', fontweight='bold', fontproperties=self._chinese_font_prop)
            ax.set_ylabel('资产净值 (元)', fontproperties=self._chinese_font_prop)
            ax.legend(loc='upper left', title='策略', prop=self._chinese_font_prop, title_fontproperties=self._chinese_font_prop)
        else:
            ax.set_title('AI 动态策略 vs 基准策略 净值对比', fontweight='bold')
            ax.set_ylabel('资产净值 (元)')
            ax.legend(loc='upper left', title='策略')
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
        ax.grid(True, linestyle='--', alpha=0.3)
        fig.autofmt_xdate()

        self._clear_kline_canvas()
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, self.kline_widget)
        self.kline_layout.addWidget(canvas)
        self.kline_layout.addWidget(toolbar)
        self.kline_layout.addWidget(self.chart_info_label)

        self._kline_canvas = canvas
        self._kline_toolbar = toolbar
        self._kline_mpl_cids = []
        canvas.draw_idle()

        self._last_backtest_result = results
        self.export_backtest_button.setEnabled(True)
        self.tabs.setCurrentIndex(0)

        metrics_combined = results.get('metrics', {}) or {}

        def format_currency(val: Any) -> str:
            return f"{float(val):,.2f}" if isinstance(val, (float, int)) and pd.notna(val) else "--"

        def format_pct(val: Any) -> str:
            return f"{val * 100:.2f}%" if isinstance(val, (float, int)) and pd.notna(val) else "--"

        summary_lines: list[str] = ["净值/收益概览："]
        for name, metric in metrics_combined.items():
            if not metric:
                continue
            parts = [name]
            if "Final Equity" in metric:
                parts.append(f"最终净值 {format_currency(metric['Final Equity'])}")
            if "Total Return" in metric:
                parts.append(f"总收益 {format_pct(metric['Total Return'])}")
            summary_lines.append(" | ".join(parts))

        if len(summary_lines) == 1:
            summary_lines = ["暂未获取到指标数据"]

        self.chart_info_label.setText('\n'.join(summary_lines))

    def _clear_kline_canvas(self, keep_placeholder: bool = False) -> None:
        # 断开旧的 matplotlib 事件连接
        if self._kline_canvas is not None:
            for cid in self._kline_mpl_cids:
                self._kline_canvas.mpl_disconnect(cid)
        self._kline_mpl_cids = []
        self._kline_canvas = None
        self._kline_toolbar = None

        # 清空布局内的组件
        while self.kline_layout.count():
            item = self.kline_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        if keep_placeholder:
            self.kline_layout.addWidget(self.chart_placeholder)
            self.chart_info_label.setText(" ")
            self.kline_layout.addWidget(self.chart_info_label)


    def _prepare_spinbox(self, spinbox: QAbstractSpinBox, placeholder: Optional[str] = None) -> None:
        getter = getattr(spinbox, "lineEdit", None)
        try:
            spinbox.setKeyboardTracking(False)
        except AttributeError:
            pass
        try:
            spinbox.setAccelerated(True)
        except AttributeError:
            pass
        if callable(getter):
            editor = getter()
            if editor is not None:
                try:
                    editor.setClearButtonEnabled(True)
                except AttributeError:
                    pass
                if placeholder:
                    editor.setPlaceholderText(placeholder)

    def _set_lstm_plot_visible(self, has_data: bool, placeholder_text: str = "") -> None:
        if self.lstm_result_group is None:
            return
        if self.lstm_result_canvas is not None:
            self.lstm_result_canvas.setVisible(has_data)
        if self.lstm_result_toolbar is not None:
            self.lstm_result_toolbar.setVisible(has_data)
        if self.lstm_result_placeholder is not None:
            if placeholder_text:
                self.lstm_result_placeholder.setText(placeholder_text)
            self.lstm_result_placeholder.setVisible(not has_data)

    def _render_lstm_prediction(self, payload: Optional[Dict[str, Any]]) -> None:
        if payload is None:
            self._set_lstm_plot_visible(False, "暂无预测结果，请先运行训练/预测任务。")
            return

        actual = np.asarray(payload.get("actual"), dtype=float) if payload.get("actual") is not None else None
        predicted = (
            np.asarray(payload.get("predicted"), dtype=float) if payload.get("predicted") is not None else None
        )

        if actual is None or predicted is None or actual.size == 0 or predicted.size == 0:
            self._set_lstm_plot_visible(False, "预测结果为空，请确认已执行预测任务。")
            return

        if actual.ndim == 1:
            actual = actual[:, np.newaxis]
        if predicted.ndim == 1:
            predicted = predicted[:, np.newaxis]

        min_labels = min(actual.shape[1], predicted.shape[1])
        if min_labels == 0:
            self._set_lstm_plot_visible(False, "预测结果缺少可绘制的标签数据。")
            return

        actual_x = payload.get("actual_x")
        predicted_x = payload.get("predicted_x")
        try:
            actual_x_arr = (
                np.asarray(actual_x, dtype=float)
                if actual_x is not None and len(actual_x) == actual.shape[0]
                else np.arange(actual.shape[0], dtype=float)
            )
        except Exception:
            actual_x_arr = np.arange(actual.shape[0], dtype=float)
        try:
            predicted_x_arr = (
                np.asarray(predicted_x, dtype=float)
                if predicted_x is not None and len(predicted_x) == predicted.shape[0]
                else np.arange(predicted.shape[0], dtype=float)
            )
        except Exception:
            predicted_x_arr = np.arange(predicted.shape[0], dtype=float)

        label_names = payload.get("label_names") or []
        if not label_names or len(label_names) < min_labels:
            label_names = [f"标签 {i+1}" for i in range(min_labels)]

        predict_day = int(payload.get("predict_day", 0) or 0)
        dates = payload.get("dates")
        if isinstance(dates, (list, tuple)) and len(dates) != actual.shape[0]:
            dates = None

        self.lstm_result_fig.clear()
        ax = self.lstm_result_fig.add_subplot(111)

        for idx in range(min_labels):
            ax.plot(
                actual_x_arr,
                actual[:, idx],
                label=f"{label_names[idx]} 实际",
                linewidth=1.6,
                alpha=0.85,
            )
            ax.plot(
                predicted_x_arr,
                predicted[:, idx],
                linestyle="--",
                label=f"{label_names[idx]} 预测",
                linewidth=1.5,
                alpha=0.8,
            )

        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xlabel("样本序号")
        ax.set_ylabel("价格")

        title = "LSTM 预测 vs 实际走势"
        subtitle = ""
        if len(label_names) == 1:
            subtitle = label_names[0]
        if predict_day > 0:
            subtitle = f"{subtitle} (预测间隔 {predict_day} 天)" if subtitle else f"预测间隔 {predict_day} 天"
        if subtitle:
            title = f"{title} - {subtitle}"
        if getattr(self, '_chinese_font_prop', None) is not None:
            ax.set_title(title, fontproperties=self._chinese_font_prop)
            ax.legend(prop=self._chinese_font_prop)
        else:
            ax.set_title(title)
            ax.legend()

        if dates is not None:
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))

            def format_fn(value, tick_number):
                idx = int(value)
                if 0 <= idx < len(dates):
                    return dates[idx]
                return ""

            ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_fn))
            self.lstm_result_fig.autofmt_xdate()

        self.lstm_result_canvas.draw_idle()
        self._set_lstm_plot_visible(True)

    def _on_worker_success(self) -> None:
        self.status_label.setText("LSTM 任务完成")
        self.append_log("LSTM 工作流执行成功。")
        self._set_controls_enabled(True)

        # 尝试从结果文件中加载预测数据
        try:
            result_path = Path("logs") / "result.npz"
            if result_path.exists():
                self.append_log(f"正在从 {result_path} 加载预测结果...")
                with np.load(result_path, allow_pickle=True) as data:
                    payload = {k: v for k, v in data.items()}
                self._last_lstm_prediction = payload
                self._render_lstm_prediction(payload)
                self.append_log("预测结果图表已更新。")
                self.tabs.setCurrentIndex(1)  # 切换到训练/预测选项卡
            else:
                self.append_log("未找到预测结果文件 (result.npz)，跳过图表更新。")
                self._set_lstm_plot_visible(False, "任务完成，但未生成预测图表。")
        except Exception as e:
            self.append_log(f"加载或渲染预测结果时出错: {e}")
            self._set_lstm_plot_visible(False, "加载预测结果失败。")

    def _on_worker_failed(self, traceback_text: str) -> None:
        self.status_label.setText("LSTM 任务失败")
        self.append_log(traceback_text)
        QMessageBox.critical(self, "任务失败", "执行过程中出现异常，详情请查看日志。")
        self._set_controls_enabled(True)
        self._set_lstm_plot_visible(False, "任务失败，无法生成预测图表。")

    def _on_worker_finished(self) -> None:
        self.worker = None
        self._set_controls_enabled(True)
        if self.log_handler:
            try:
                self.log_handler.signal.message.disconnect(self.append_log)
            except TypeError:
                pass
            self.log_handler = None

    def _set_controls_enabled(self, enabled: bool) -> None:
        """启用或禁用所有交互控件"""
        for widget in self._controls:
            widget.setEnabled(enabled)
        # 特殊处理：只有在有回测结果时才启用导出按钮
        if enabled:
            self.export_backtest_button.setEnabled(self._last_backtest_result is not None)

    def show_backtest(self) -> None:
        if self.full_df is None:
            QMessageBox.warning(self, "缺少数据", "请先在图表分析页加载数据。")
            return

        selected_strategy = self.strategy_combo.currentText()
        self.append_log(f"开始执行回测: {selected_strategy}")
        self.status_label.setText(f"正在回测: {selected_strategy}...")
        self._set_controls_enabled(False)

        try:
            if "网格策略" in selected_strategy:
                result = run_grid_backtest(
                    self.full_df,
                    initial_cash=self.grid_initial_cash_spin.value(),
                    fee=self.grid_fee_spin.value(),
                    grid_interval_percent=self.grid_interval_percent_spin.value(),
                    num_lower_grids=self.grid_num_lower_spin.value(),
                    num_upper_grids=self.grid_num_upper_spin.value(),
                    order_size=self.grid_order_size_spin.value(),
                )
            else: # 默认均线策略
                result = run_backtest(self.full_df)

            self._last_backtest_result = result
            self.export_backtest_button.setEnabled(True)
            self._display_backtest_on_kline(result)
            self.append_log(f"回测完成: {selected_strategy}")
            self.status_label.setText("回测完成")

        except Exception as e:
            self.append_log(f"回测失败: {e}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "回测失败", f"执行回测时出错: {e}")
            self.status_label.setText("回测失败")
        finally:
            self._set_controls_enabled(True)

    def _display_backtest_on_kline(self, result: Dict[str, Any]) -> None:
        """在现有的K线图上叠加回测结果"""
        if self._kline_canvas is None:
            self.show_kline() # 如果没有图，先画一个
        if self._kline_canvas is None:
            self.append_log("无法绘制回测结果，因为K线图不存在。")
            return

        fig = self._kline_canvas.figure
        main_ax = fig.axes[0]

        # 清除旧的回测标记
        for child in main_ax.get_children():
            if isinstance(child, matplotlib.collections.PathCollection) and child.get_label() in ['buy', 'sell']:
                child.remove()
        for ax in fig.axes:
            if ax != main_ax and ax.get_label() == 'equity_ax':
                ax.remove()

        trades = result.get('trades')
        if trades is not None and not trades.empty:
            buy_trades = trades[trades['Size'] > 0]
            sell_trades = trades[trades['Size'] < 0]
            main_ax.scatter(buy_trades.index, buy_trades['Price'] * 0.99, marker='^', color='magenta', s=100, label='buy', zorder=5)
            main_ax.scatter(sell_trades.index, sell_trades['Price'] * 1.01, marker='v', color='cyan', s=100, label='sell', zorder=5)

        equity_curve = result.get('equity_curve')
        if equity_curve is not None:
            equity_ax = main_ax.twinx()
            equity_ax.set_label('equity_ax')
            equity_ax.plot(equity_curve.index, equity_curve, color='blue', alpha=0.6, label='策略净值')
            equity_ax.set_ylabel('策略净值', color='blue')
            equity_ax.tick_params(axis='y', labelcolor='blue')
            # 调整Y轴范围，使其不与价格重叠太多
            min_eq, max_eq = equity_curve.min(), equity_curve.max()
            min_price, max_price = main_ax.get_ylim()
            # 尝试将净值曲线放在价格下方
            if min_eq > max_price:
                 pass # 净值远高于价格，正常显示
            else:
                 # 压缩净值显示范围，避免与价格重叠
                 padding = (max_eq - min_eq) * 0.1
                 equity_ax.set_ylim(min_eq - padding, max_eq + padding)


        self._kline_canvas.draw_idle()
        self.tabs.setCurrentIndex(0) # 切换到图表

        # 更新信息标签
        metrics = result.get('metrics', {})
        info_text = (
            f"最终净值: {metrics.get('Final Equity', 0):,.2f} | "
            f"总收益: {metrics.get('Total Return', 0):.2%} | "
            f"年化收益: {metrics.get('Annualized Return', 0):.2%} | "
            f"最大回撤: {metrics.get('Max Drawdown', 0):.2%} | "
            f"夏普比率: {metrics.get('Sharpe Ratio', 0):.2f} | "
            f"交易次数: {metrics.get('Total Trades', 0)}"
        )
        self.chart_info_label.setText(info_text)
        self.append_log(f"回测指标: {info_text}")

    def export_backtest_report(self) -> None:
        if self.ai_comparison_result is not None:
            export_data = self.ai_comparison_result
        else:
            export_data = self._last_backtest_result

        if export_data is None:
            QMessageBox.warning(self, "无结果", "没有可导出的回测结果。")
            return

        start_dir = os.path.join(os.getcwd(), "reports")
        os.makedirs(start_dir, exist_ok=True)

        default_filename = f"backtest_report_{datetime.now():%Y%m%d_%H%M%S}.html"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存回测报告",
            os.path.join(start_dir, default_filename),
            "HTML 文件 (*.html)",
        )

        if not path:
            return

        # 对比回测导出：自定义 HTML
        if isinstance(export_data, dict) and "equity_curves" in export_data:
            try:
                equity_curves = export_data.get("equity_curves", {}) or {}
                metrics = export_data.get("metrics", {}) or {}

                curve_map: Dict[str, pd.Series] = {}
                for name, curve in equity_curves.items():
                    if curve is None:
                        continue
                    if isinstance(curve, pd.DataFrame):
                        for col in curve.columns:
                            curve_map[f"{name}-{col}"] = pd.Series(curve[col]).astype(float)
                    else:
                        curve_map[name] = pd.Series(curve).astype(float)

                equity_df = pd.DataFrame(curve_map)
                equity_df.index.name = "Index"

                metrics_rows = []
                for strategy, metric in metrics.items():
                    row = {"策略": strategy}
                    for key, value in (metric or {}).items():
                        if isinstance(value, (float, int)):
                            row[key] = f"{value:.6f}" if abs(value) < 1 else f"{value:.4f}"
                        else:
                            row[key] = value
                    metrics_rows.append(row)
                metrics_df = pd.DataFrame(metrics_rows)

                html_parts = [
                    "<html><head><meta charset='utf-8'><title>策略对比回测报告</title>",
                    "<style>body{font-family:Segoe UI, Arial, sans-serif;padding:20px;}table{border-collapse:collapse;width:100%;margin-bottom:20px;}th,td{border:1px solid #ccc;padding:8px;text-align:right;}th{background:#f5f5f5;text-align:center;}td:first-child,th:first-child{text-align:left;}</style>",
                    "</head><body>",
                    "<h1>策略对比回测报告</h1>",
                    f"<p>导出时间：{datetime.now():%Y-%m-%d %H:%M:%S}</p>",
                ]

                if not metrics_df.empty:
                    html_parts.append("<h2>核心指标</h2>")
                    html_parts.append(metrics_df.to_html(index=False, escape=False))
                else:
                    html_parts.append("<p>暂无指标数据。</p>")

                if not equity_df.empty:
                    html_parts.append("<h2>净值曲线数据</h2>")
                    html_parts.append(equity_df.to_html())
                else:
                    html_parts.append("<p>暂无净值曲线数据。</p>")

                html_parts.append("</body></html>")

                with open(path, "w", encoding="utf-8") as fout:
                    fout.write("\n".join(html_parts))

                self.append_log(f"策略对比回测报告已保存至: {path}")
                QMessageBox.information(self, "导出成功", f"策略对比回测报告已保存至:\n{path}")
            except Exception as exc:  # noqa: BLE001
                self.append_log(f"导出策略对比报告失败: {exc}\n{traceback.format_exc()}")
                QMessageBox.critical(self, "导出失败", f"导出报告时出错: {exc}")
            return

        # 传统 backtesting 导出
        try:
            from backtesting import plotting

            plotting.plot(
                export_data,
                filename=path,
                open_browser=True,
                resample=False,
            )
            self.append_log(f"回测报告已保存至: {path}")
            QMessageBox.information(self, "导出成功", f"报告已保存至:\n{path}")
        except Exception as exc:  # noqa: BLE001
            self.append_log(f"导出报告失败: {exc}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "导出失败", f"导出报告时出错: {exc}")

    def closeEvent(self, event) -> None:
        """确保在关闭窗口时，所有后台线程都能被正确终止。"""
        threads = [self.worker, self.ai_training_thread, self.ai_backtest_thread]
        running_threads = [t for t in threads if t and t.isRunning()]

        if not running_threads:
            event.accept()
            return

        reply = QMessageBox.question(
            self,
            "确认退出",
            "有后台任务正在运行，确定要强制退出吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            for t in running_threads:
                try:
                    self.append_log(f"正在终止线程: {t.__class__.__name__}...")
                    t.quit()
                    t.wait(2000)  # 等待最多2秒
                except Exception as e:
                    self.append_log(f"终止线程时出错: {e}")
            event.accept()
        else:
            event.ignore()

    def _setup_crosshair(self, canvas: FigureCanvas, axes: list, df: pd.DataFrame) -> None:
        """在图表上设置十字光标和信息显示"""
        if not axes:
            return

        main_ax = axes[0]
        lines = [ax.axvline(df.index[0], color='k', linestyle='--', linewidth=0.5, visible=False) for ax in axes]
        line_h = main_ax.axhline(df['Close'].iloc[0], color='k', linestyle='--', linewidth=0.5, visible=False)

        def on_mouse_move(event):
            if event.inaxes is None:
                if any(line.get_visible() for line in lines) or line_h.get_visible():
                    for line in lines:
                        line.set_visible(False)
                    line_h.set_visible(False)
                    canvas.draw_idle()
                return

            x_val = event.xdata
            idx = int(round(x_val))

            if 0 <= idx < len(df):
                for line in lines:
                    line.set_xdata([idx])
                    line.set_visible(True)

                y_val = df['Close'].iloc[idx]
                line_h.set_ydata([y_val])
                line_h.set_visible(True)

                self._update_chart_info(df.iloc[idx])
                canvas.draw_idle()

        cid = canvas.mpl_connect('motion_notify_event', on_mouse_move)
        self._kline_mpl_cids.append(cid)

    def _update_chart_info(self, series: pd.Series) -> None:
        """更新图表下方的详细信息标签"""
        date_str = series.name.strftime('%Y-%m-%d')
        parts = [
            f"日期: {date_str}",
            f"开: {series['Open']:.2f}",
            f"高: {series['High']:.2f}",
            f"低: {series['Low']:.2f}",
            f"收: {series['Close']:.2f}",
            f"量: {series['Volume'] / 100:.2f}万手"
        ]
        if 'AmountWan' in series:
            parts.append(f"额: {series['AmountWan']:.2f}万元")
        if 'K' in series and 'D' in series and 'J' in series:
            parts.append(f"KDJ: {series['K']:.1f}, {series['D']:.1f}, {series['J']:.1f}")

        self.chart_info_label.setText(" | ".join(parts))

    def _setup_ctrl_zoom(self, canvas: FigureCanvas, axes: list, dates: pd.DatetimeIndex) -> None:
        """通过 Ctrl+滚轮 实现K线图的缩放"""
        if not axes:
            return

        main_ax = axes[0]

        def on_scroll(event):
            if event.key != 'control' or event.inaxes not in axes:
                return

            base_scale = 1.1
            cur_xlim = main_ax.get_xlim()
            cur_xrange = cur_xlim[1] - cur_xlim[0]
            xdata = event.xdata

            if event.button == 'up': # 放大
                scale_factor = 1 / base_scale
            elif event.button == 'down': # 缩小
                scale_factor = base_scale
            else:
                return

            new_width = cur_xrange * scale_factor
            relx = (cur_xlim[1] - xdata) / cur_xrange
            
            new_xlim = [
                xdata - new_width * (1 - relx),
                xdata + new_width * relx
            ]
            
            # 限制缩放范围
            if new_xlim[0] < -0.5: new_xlim[0] = -0.5
            if new_xlim[1] > len(dates) - 0.5: new_xlim[1] = len(dates) - 0.5
            
            for ax in axes:
                ax.set_xlim(new_xlim)
            
            canvas.draw_idle()

        cid = canvas.mpl_connect('scroll_event', on_scroll)
        self._kline_mpl_cids.append(cid)


def main_gui() -> None:
    """主GUI入口函数"""
    app = QApplication(sys.argv)

    # --- 国际化/汉化 ---
    translator = QTranslator()
    locale = QLocale.system().name()
    # 尝试加载Qt官方中文翻译
    path = QLibraryInfo.location(QLibraryInfo.TranslationsPath)
    if translator.load(f"qt_{locale}", path):
        app.installTranslator(translator)
    else:
        print(f"未能加载Qt官方翻译文件: qt_{locale}.qm at {path}")

    # --- 启动主窗口 ---
    main_win = MainWindow()
    main_win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/gui_session.log", mode="a", encoding="utf-8"),
        ],
    )
    main_gui()
