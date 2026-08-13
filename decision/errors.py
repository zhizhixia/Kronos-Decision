"""Kronos 决策系统异常体系。"""


class KronosError(Exception):
    """所有 Kronos 异常的基类。"""


class ConfigError(KronosError):
    """配置错误（参数缺失、类型不匹配）。"""


class DataSourceError(KronosError):
    """数据源不可用（网络错误、接口异常、限流）。"""


class DataValidationError(KronosError):
    """数据校验失败（格式错误、列缺失、空值过多）。"""


class ModelNotReadyError(KronosError):
    """模型未加载或加载失败。"""


class PredictionTimeoutError(KronosError, TimeoutError):
    """模型推理超时。"""
