"""测试错误类型体系。"""
import pytest
from decision.errors import (
    KronosError, ConfigError, DataSourceError,
    DataValidationError, ModelNotReadyError, PredictionTimeoutError,
)


def test_errors_are_exceptions():
    """所有错误类型都是 Exception 的子类。"""
    assert issubclass(KronosError, Exception)
    assert issubclass(ConfigError, KronosError)
    assert issubclass(DataSourceError, KronosError)
    assert issubclass(DataValidationError, KronosError)
    assert issubclass(ModelNotReadyError, KronosError)
    assert issubclass(PredictionTimeoutError, KronosError)


def test_errors_can_be_raised_and_caught():
    """错误可以被正确抛出和捕获。"""
    with pytest.raises(KronosError):
        raise DataSourceError("数据源连接超时")

    try:
        raise DataValidationError("缺少必需列: open")
    except KronosError as e:
        assert "open" in str(e)


def test_error_message_preserved():
    """错误消息在捕获后保持完整。"""
    msg = "测试错误消息 —— 包含特殊字符: !@#$%^&*()"
    err = ConfigError(msg)
    assert str(err) == msg