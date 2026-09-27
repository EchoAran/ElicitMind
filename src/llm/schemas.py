from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator


class LLMCallRecord(BaseModel):
    call_id: str
    turn_id: Optional[str] = None
    module: Optional[str] = None
    prompt_name: Optional[str] = None
    model: str = "default"
    temperature: float = 0.2
    prompt: str = ""
    query: Optional[str] = None
    request: dict[str, Any] = Field(default_factory=dict)
    raw_response: Optional[str] = None
    parsed_response: Optional[Any] = None
    response_time_ms: float = 0.0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    status: str = "ok"  # "ok" | "success" | "error"
    error_message: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _sync_metrics(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        # 1. Sync response time (latency_ms <-> response_time_ms)
        lat = data.get("latency_ms")
        resp_time = data.get("response_time_ms")
        if lat is not None and resp_time is None:
            data["response_time_ms"] = float(lat)
        elif resp_time is not None and lat is None:
            data["latency_ms"] = float(resp_time)
        elif lat is not None and resp_time is not None:
            if float(lat) != 0.0 and float(resp_time) == 0.0:
                data["response_time_ms"] = float(lat)
            elif float(resp_time) != 0.0 and float(lat) == 0.0:
                data["latency_ms"] = float(resp_time)

        # 2. Sync total tokens
        tot = data.get("total_tokens")
        p_val = int(data.get("prompt_tokens") or 0)
        c_val = int(data.get("completion_tokens") or 0)
        calc_tot = p_val + c_val
        if tot is None or int(tot) == 0:
            if calc_tot > 0:
                data["total_tokens"] = calc_tot

        return data
