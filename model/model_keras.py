# -*- coding: UTF-8 -*-
"""
Keras 模型实现，兼容独立 Keras 与 TensorFlow 2.x 集成的 keras。
"""

from __future__ import annotations

import warnings

try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras as keras_module  # type: ignore
except Exception:  # noqa: BLE001
    tf = None  # type: ignore
    try:
        import keras as keras_module  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "未检测到 Keras 或 TensorFlow，请先安装相关依赖：pip install tensorflow"
        ) from exc

layers = keras_module.layers
models = keras_module.models
callbacks = keras_module.callbacks


def get_keras_model(config):
    inputs = layers.Input(shape=(config.time_step, config.input_size))
    lstm = inputs
    for _ in range(config.lstm_layers):
        lstm = layers.LSTM(
            units=config.hidden_size,
            dropout=config.dropout_rate,
            return_sequences=True,
        )(lstm)
    outputs = layers.Dense(config.output_size)(lstm)
    model = models.Model(inputs, outputs)
    model.compile(loss="mse", optimizer="adam")
    return model


def gpu_train_init(logger) -> bool:
    """初始化GPU设置，返回是否成功"""
    if tf is None:
        return False
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU 初始化成功，检测到 {len(gpus)} 个GPU设备")
            return True
        else:
            logger.info("未检测到GPU设备")
            return False
    except Exception as e:  # noqa: BLE001
        logger.warning(f"GPU 初始化失败，已退回 CPU 模式: {e}")
        return False


def train(config, logger, train_and_valid_data):
    # 设备选择逻辑
    device_pref = getattr(config, 'device_preference', 'auto').lower()
    use_gpu = False
    
    if device_pref == 'cpu':
        logger.info("使用 CPU 进行训练")
        # 强制TensorFlow使用CPU
        if tf is not None:
            try:
                tf.config.set_visible_devices([], 'GPU')
            except Exception:
                pass
    elif device_pref == 'gpu':
        use_gpu = gpu_train_init(logger)
        if not use_gpu:
            logger.warning("GPU不可用，但用户强制要求使用GPU，训练可能失败")
    else:  # auto
        if config.use_cuda:
            use_gpu = gpu_train_init(logger)

    train_X, train_Y, valid_X, valid_Y = train_and_valid_data
    model = get_keras_model(config)
    model.summary(print_fn=logger.info)
    if config.add_train:
        model.load_weights(config.model_save_path + config.model_name)

    check_point = callbacks.ModelCheckpoint(
        filepath=config.model_save_path + config.model_name,
        monitor="val_loss",
        save_best_only=True,
        mode="auto",
    )
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=config.patience, mode="auto")
    model.fit(
        train_X,
        train_Y,
        batch_size=config.batch_size,
        epochs=config.epoch,
        verbose=2,
        validation_data=(valid_X, valid_Y),
        callbacks=[check_point, early_stop],
    )


def predict(config, test_X):
    model = get_keras_model(config)
    model.load_weights(config.model_save_path + config.model_name)
    result = model.predict(test_X, batch_size=1)
    result = result.reshape((-1, config.output_size))
    return result
