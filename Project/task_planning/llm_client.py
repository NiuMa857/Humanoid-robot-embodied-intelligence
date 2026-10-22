"""云端 / 本地 LLM/VLM 客户端(完整实装)。

用 httpx(支持 async / HTTP/2 / 流式)。如果没装会自动 fallback 到 urllib。
"""
from __future__ import annotations
import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any

from common.logging import setup_logger

_log = setup_logger("llm_client")


# ============================================================
# HTTP 后端(优先 httpx,fallback urllib)
# ============================================================

def _http_post(url: str, json_body: dict, headers: dict, timeout_s: float) -> dict:
    """POST JSON,返回响应 dict。"""
    try:
        import httpx
        with httpx.Client(timeout=timeout_s) as client:
            r = client.post(url, json=json_body, headers=headers)
            r.raise_for_status()
            return r.json()
    except ImportError:
        import urllib.request, urllib.error
        req = urllib.request.Request(
            url, data=json.dumps(json_body).encode("utf-8"),
            headers={**headers, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}") from e


def _http_get(url: str, headers: dict, timeout_s: float) -> bytes:
    """GET 字节流。"""
    try:
        import httpx
        with httpx.Client(timeout=timeout_s) as client:
            r = client.get(url, headers=headers)
            r.raise_for_status()
            return r.content
    except ImportError:
        import urllib.request
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.read()


def _chat_completions_url(endpoint: str) -> str:
    """endpoint 可写成 https://host 或 https://host/v1,避免拼出 /v1/v1。"""
    base = endpoint.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


_SKILL_IDS_HINT = "可用技能:SK_WALK / SK_CLIMB / SK_REACH / SK_GRASP / SK_PLACE / SK_ASSEMBLE / SK_STAND / SK_TURN"


def _extract_skills(result: Any) -> Any:
    """从 LLM 返回中取技能列表:兼容 {"skills": [...]} 和裸 list;其余原样返回交给校验器报错。"""
    if isinstance(result, dict) and "skills" in result:
        return result["skills"]
    return result


# ============================================================
# LLM 抽象基类
# ============================================================

class LLMError(Exception):
    """LLM 调用失败(超时 / 解析错误 / HTTP 错误)。"""
    pass


class BaseLLMClient(ABC):
    """LLM 客户端统一接口。"""

    def __init__(self, timeout_s: float = 2.0, max_retries: int = 1, **kwargs):
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    @abstractmethod
    def understand_instruction(self, text: str) -> dict:
        """自然语言 → TaskGoal dict。"""
        ...

    @abstractmethod
    def decompose(self, goal: dict, scene: dict) -> list[dict]:
        """任务分解 → Skill 列表(原始 dict)。"""
        ...

    @abstractmethod
    def replan(self, failure: str, state: dict) -> list[dict]:
        """重规划(失败后调用)。"""
        ...

    def _retry(self, fn, *args, **kwargs) -> Any:
        """带重试的调用(指数 backoff)。"""
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    wait = 0.2 * (2 ** attempt)
                    time.sleep(wait)
        raise LLMError(f"after {self.max_retries + 1} attempts: {last_err}") from last_err

    def close(self) -> None:
        pass


# ============================================================
# Remote 实现:OpenAI-compatible 协议
# ============================================================

class RemoteLLMClient(BaseLLMClient):
    """云端 LLM 客户端(POST/JSON)。

    适配 OpenAI-compatible `/v1/chat/completions`(覆盖 OpenAI/DeepSeek/智谱/自建等)。
    """

    def __init__(
        self,
        endpoint: str = "",
        api_key: str = "",
        model: str = "gpt-4o-mini",
        timeout_s: float = 2.0,
        max_retries: int = 1,
    ):
        super().__init__(timeout_s=timeout_s, max_retries=max_retries)
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.model = model

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _chat_url(self) -> str:
        return _chat_completions_url(self.endpoint)

    def _chat(self, system: str, user: str, json_mode: bool = True) -> dict:
        if not self.endpoint:
            raise LLMError("LLM endpoint not configured")
        url = self._chat_url()
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        resp = self._retry(_http_post, url, body, self._headers(), self.timeout_s)
        content = resp["choices"][0]["message"]["content"]
        return json.loads(content) if json_mode else {"text": content}

    def understand_instruction(self, text: str) -> dict:
        system = (
            "你是机器人任务规划助手。把自然语言指令解析为结构化 JSON:\n"
            '{"type": "...", "target": "...", "location": "...", "params": {}}'
        )
        user = f"指令:{text}\n返回 JSON,只包含字段。"
        return self._chat(system, user, json_mode=True)

    def decompose(self, goal: dict, scene: dict) -> list[dict]:
        # response_format=json_object 要求顶层是对象,所以约定 {"skills": [...]}
        system = (
            "你是任务分解器。把目标分解为技能序列,返回 JSON 对象:\n"
            '{"skills": [{"id": "SK_WALK", "params": {...}}, ...]}\n'
            f"{_SKILL_IDS_HINT}"
        )
        user = f"目标:{json.dumps(goal, ensure_ascii=False)}\n场景:{json.dumps(scene, ensure_ascii=False)}"
        return _extract_skills(self._chat(system, user, json_mode=True))

    def replan(self, failure: str, state: dict) -> list[dict]:
        system = (
            "你是重规划器。state 中包含原始目标 goal、已完成技能 completed_skills、"
            "失败技能 failed_skill、剩余计划 remaining_skills 和场景 scene。\n"
            "请给出从当前状态出发、最终完成 goal 的新技能序列(不要重复已完成的技能),返回 JSON 对象:\n"
            '{"skills": [{"id": "...", "params": {...}}, ...]}\n'
            f"{_SKILL_IDS_HINT}"
        )
        user = f"失败:{failure}\n状态:{json.dumps(state, ensure_ascii=False)}"
        return _extract_skills(self._chat(system, user, json_mode=True))


# ============================================================
# Local 实现(同样走 OpenAI-compatible,目标地址是 localhost)
# ============================================================

class LocalLLMClient(RemoteLLMClient):
    """本地 LLM 客户端(vLLM / TGI / llama.cpp server,OpenAI-compatible)。"""

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8000",
        model: str = "local-model",
        api_key: str = "EMPTY",
        timeout_s: float = 5.0,
        max_retries: int = 1,
    ):
        super().__init__(
            endpoint=endpoint, api_key=api_key, model=model,
            timeout_s=timeout_s, max_retries=max_retries,
        )


# ============================================================
# VLM(基类 + Remote/Local)
# ============================================================

class BaseVLMClient(ABC):
    def __init__(self, timeout_s: float = 2.0, **kwargs):
        self.timeout_s = timeout_s

    @abstractmethod
    def understand_scene(self, image_bytes: bytes, prompt: str = "") -> dict:
        ...


class RemoteVLMClient(BaseVLMClient):
    """云端 VLM(走 OpenAI-compatible vision 接口)。"""

    def __init__(self, endpoint: str = "", api_key: str = "", model: str = "gpt-4o", timeout_s: float = 2.0):
        super().__init__(timeout_s=timeout_s)
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.model = model

    def understand_scene(self, image_bytes: bytes, prompt: str = "描述场景中的物体和它们的空间关系,返回 JSON。") -> dict:
        import base64
        b64 = base64.b64encode(image_bytes).decode("ascii")
        if not self.endpoint:
            raise LLMError("VLM endpoint not configured")
        url = _chat_completions_url(self.endpoint)
        body = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]}
            ],
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        resp = _http_post(url, body, headers, self.timeout_s)
        return json.loads(resp["choices"][0]["message"]["content"])

    def close(self) -> None:
        # HTTP client 是 stateless 的,无需释放
        pass


class LocalVLMClient(RemoteVLMClient):
    def __init__(self, endpoint: str = "http://127.0.0.1:8001", model: str = "llava", timeout_s: float = 5.0):
        super().__init__(endpoint=endpoint, api_key="EMPTY", model=model, timeout_s=timeout_s)


# ============================================================
# 工厂
# ============================================================

def _resolve_api_key(cfg: dict, default: str) -> str:
    """显式 api_key 优先;为空时从 api_key_env 指定的环境变量读取。"""
    if cfg.get("api_key"):
        return cfg["api_key"]
    env_name = cfg.get("api_key_env")
    return os.environ.get(env_name, default) if env_name else default


def make_llm_client(config: dict) -> BaseLLMClient:
    cfg = config.get("llm", {})
    mode = cfg.get("mode", "remote")
    if mode == "local":
        return LocalLLMClient(
            endpoint=cfg.get("endpoint", "http://127.0.0.1:8000"),
            model=cfg.get("model", "local-model"),
            api_key=_resolve_api_key(cfg, "EMPTY"),
            timeout_s=cfg.get("timeout_s", 5.0),
            max_retries=cfg.get("max_retries", 1),
        )
    return RemoteLLMClient(
        endpoint=cfg.get("endpoint", ""),
        api_key=_resolve_api_key(cfg, ""),
        model=cfg.get("model", "gpt-4o-mini"),
        timeout_s=cfg.get("timeout_s", 2.0),
        max_retries=cfg.get("max_retries", 1),
    )


def make_vlm_client(config: dict) -> BaseVLMClient:
    cfg = config.get("vlm", {})
    mode = cfg.get("mode", "remote")
    if mode == "local":
        return LocalVLMClient(
            endpoint=cfg.get("endpoint", "http://127.0.0.1:8001"),
            model=cfg.get("model", "llava"),
            timeout_s=cfg.get("timeout_s", 5.0),
        )
    return RemoteVLMClient(
        endpoint=cfg.get("endpoint", ""),
        api_key=_resolve_api_key(cfg, ""),
        model=cfg.get("model", "gpt-4o"),
        timeout_s=cfg.get("timeout_s", 2.0),
    )


__all__ = [
    "BaseLLMClient", "BaseVLMClient",
    "RemoteLLMClient", "LocalLLMClient",
    "RemoteVLMClient", "LocalVLMClient",
    "make_llm_client", "make_vlm_client",
    "LLMError",
]
