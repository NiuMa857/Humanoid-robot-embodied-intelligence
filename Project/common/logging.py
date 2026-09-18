"""统一日志(结构化 JSON 输出)。

特性:
  - 名字空间(每个模块用自己的 logger)
  - JSON 行输出(方便 Loki/Elasticsearch 收集)
  - 自动加 timestamp / level / module / line
  - 支持 trace_id / skill_id 等自定义字段(通过 extra={...})
"""
from __future__ import annotations
import json
import logging
import sys
import time
from typing import Any


class JsonFormatter(logging.Formatter):
    """JSON 行格式。"""

    DEFAULT_FIELDS = {
        "timestamp", "level", "name", "message", "module", "func", "line"
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }
        # 自定义 extra 字段(非 LogRecord 内置)
        for k, v in record.__dict__.items():
            if k not in self.DEFAULT_FIELDS and k not in (
                "args", "msg", "levelname", "levelno", "pathname", "filename",
                "exc_info", "exc_text", "stack_info", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "taskName",
            ):
                try:
                    json.dumps(v)  # 确认可序列化
                    payload[k] = v
                except (TypeError, ValueError):
                    payload[k] = repr(v)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


_initialized = set()


def setup_logger(
    name: str = "oli",
    level: str = "INFO",
    json: bool = True,
    stream=None,
) -> logging.Logger:
    """初始化 logger(单次设置,重复调用返回同一个 logger)。

    Args:
        name: logger 名字(模块用 `setup_logger(__name__)`)。
        level: 日志级别("DEBUG" / "INFO" / "WARNING" / "ERROR")。
        json: True 用 JSON formatter,False 用纯文本。
        stream: 输出流,默认 sys.stderr。
    """
    log = logging.getLogger(name)
    if name in _initialized:
        return log
    log.setLevel(getattr(logging, level.upper(), logging.INFO))
    log.propagate = False  # 避免 root logger 重复输出

    h = logging.StreamHandler(stream or sys.stderr)
    if json:
        h.setFormatter(JsonFormatter())
    else:
        h.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s | %(message)s"
        ))
    log.addHandler(h)
    _initialized.add(name)
    return log


__all__ = ["setup_logger", "JsonFormatter"]
