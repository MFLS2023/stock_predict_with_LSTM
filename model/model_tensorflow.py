# -*- coding: UTF-8 -*-
"""
@author: hichenway
@知乎: 海晨威
@contact: lyshello123@163.com
@time: 2020/5/9 17:00
@license: Apache
tensorflow 模型
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

_TF_IMPORT_ERROR: Optional[Exception] = None
try:  # pragma: no cover - optional dependency
    import tensorflow as tf  # type: ignore
except Exception as exc:  # noqa: BLE001
    tf = None  # type: ignore[assignment]
    _TF_IMPORT_ERROR = exc
    tfv1 = None  # type: ignore[assignment]
else:  # pragma: no branch
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.compat.v1.disable_v2_behavior()
    tfv1 = tf.compat.v1


def _require_tf() -> Tuple[Any, Any]:
    if tf is None or tfv1 is None:
        message = "未检测到 TensorFlow，请先安装相关依赖：pip install tensorflow"
        raise ImportError(message) from _TF_IMPORT_ERROR
    return tf, tfv1

class Model:
    def __init__(self, config):
        self.config = config
        self._tf, self._tfv1 = _require_tf()

        self.placeholders()
        self.net()
        self.operate()


    def placeholders(self):
        self.X = self._tfv1.placeholder(self._tf.float32, [None,self.config.time_step,self.config.input_size])
        self.Y = self._tfv1.placeholder(self._tf.float32, [None,self.config.time_step,self.config.output_size])


    def net(self):

        def dropout_cell():
            basicLstm = self._tfv1.nn.rnn_cell.LSTMCell(self.config.hidden_size)
            dropoutLstm = self._tfv1.nn.rnn_cell.DropoutWrapper(basicLstm, output_keep_prob=1-self.config.dropout_rate)
            return dropoutLstm

        cell = self._tfv1.nn.rnn_cell.MultiRNNCell([dropout_cell() for _ in range(self.config.lstm_layers)])

        output_rnn, _ = self._tfv1.nn.dynamic_rnn(cell=cell, inputs=self.X, dtype=self._tf.float32)

        # shape of output_rnn is: [batch_size, time_step, hidden_size]
        self.pred = self._tfv1.layers.dense(inputs=output_rnn, units=self.config.output_size)


    def operate(self):
        self.loss = self._tfv1.reduce_mean(self._tf.square(self._tf.reshape(self.pred, [-1]) - self._tf.reshape(self.Y, [-1])))
        self.optim = self._tfv1.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)
        self.saver = self._tfv1.train.Saver(self._tfv1.global_variables())


def train(config, logger, train_and_valid_data):
    _, tfv1 = _require_tf()
    if config.do_train_visualized:  # loss可视化
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            logger.warning("未安装 tensorboardX，已跳过可视化记录。")
            config.do_train_visualized = False
        else:
            train_writer = SummaryWriter(config.log_save_path + "Train")
            eval_writer = SummaryWriter(config.log_save_path + "Eval")

    with tfv1.variable_scope("stock_predict"):
        model = Model(config)

    train_X, train_Y, valid_X, valid_Y = train_and_valid_data
    train_len = len(train_X)
    valid_len = len(valid_X)

    # 设备选择逻辑
    device_pref = getattr(config, 'device_preference', 'auto').lower()
    
    if device_pref == 'cpu':
        logger.info("使用 CPU 进行训练")
        sess_config = tfv1.ConfigProto(device_count={'GPU': 0})
    elif device_pref == 'gpu':
        logger.info("强制使用 GPU 进行训练")
        sess_config = tfv1.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.7
        sess_config.gpu_options.allow_growth = True
    else:  # auto
        if config.use_cuda:
            logger.info("自动模式：尝试使用GPU训练")
            sess_config = tfv1.ConfigProto(log_device_placement=True, allow_soft_placement=True)
            sess_config.gpu_options.per_process_gpu_memory_fraction = 0.7
            sess_config.gpu_options.allow_growth = True
        else:
            logger.info("自动模式：使用CPU训练")
            sess_config = None

    with tfv1.Session(config=sess_config) as sess:
        sess.run(tfv1.global_variables_initializer())

        valid_loss_min = float("inf")
        bad_epoch = 0
        global_step = 0
        for epoch in range(config.epoch):
            logger.info("Epoch {}/{}".format(epoch, config.epoch))
            # 训练
            train_loss_array = []
            for step in range(train_len//config.batch_size):
                feed_dict = {model.X: train_X[step*config.batch_size : (step+1)*config.batch_size],
                             model.Y: train_Y[step*config.batch_size : (step+1)*config.batch_size]}
                train_loss, _ = sess.run([model.loss,model.optim], feed_dict=feed_dict)
                train_loss_array.append(train_loss)
                if config.do_train_visualized and global_step % 100 == 0:   # 每一百步显示一次
                    train_writer.add_scalar('Train_Loss', train_loss, global_step+1)
                global_step += 1

            # 验证与早停
            valid_loss_array = []
            for step in range(valid_len//config.batch_size):
                feed_dict = {model.X: valid_X[step * config.batch_size: (step + 1) * config.batch_size],
                             model.Y: valid_Y[step * config.batch_size: (step + 1) * config.batch_size]}
                valid_loss = sess.run(model.loss, feed_dict=feed_dict)
                valid_loss_array.append(valid_loss)

            train_loss_cur = np.mean(train_loss_array)
            valid_loss_cur = np.mean(valid_loss_array)
            logger.info("The train loss is {:.6f}. ".format(train_loss_cur) +
                        "The valid loss is {:.6f}.".format(valid_loss_cur))
            if config.do_train_visualized:
                train_writer.add_scalar('Epoch_Loss', train_loss_cur, epoch + 1)
                eval_writer.add_scalar('Epoch_Loss', valid_loss_cur, epoch + 1)

            if valid_loss_cur < valid_loss_min:
                valid_loss_min = valid_loss_cur
                bad_epoch = 0
                model.saver.save(sess, config.model_save_path + config.model_name)
            else:
                bad_epoch += 1
                if bad_epoch >= config.patience:
                    logger.info(" The training stops early in epoch {}".format(epoch))
                    break


def predict(config, test_X):
    _, tfv1 = _require_tf()
    config.dropout_rate = 0     # 预测模式要调为1

    tfv1.reset_default_graph()    # # 清除默认图的堆栈，并设置全局图为默认图
    with tfv1.variable_scope("stock_predict", reuse=tfv1.AUTO_REUSE):
        model = Model(config)

    test_len = len(test_X)
    with tfv1.Session() as sess:
        module_file = tfv1.train.latest_checkpoint(config.model_save_path)
        model.saver.restore(sess, module_file)

        result = np.zeros((test_len*config.time_step, config.output_size))
        for step in range(test_len):
            feed_dict = {model.X: test_X[step : (step + 1)]}
            test_pred = sess.run(model.pred, feed_dict=feed_dict)
            result[step*config.time_step : (step + 1)*config.time_step] = test_pred[0,:,:]
    return result


