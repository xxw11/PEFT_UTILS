import logging
import os
import random

import numpy as np
import torch


def get_logger(logger_name="peft_log",log_path="./log/"):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    # 创建一个FileHandler，并设置日志级别和日志格式化器
    file_handler = logging.FileHandler(os.path.join(log_path, logger_name))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    stream_handler.delay = False
    file_handler.delay = False
    return logger

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False