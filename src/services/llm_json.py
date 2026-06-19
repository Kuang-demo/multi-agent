from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from src.config import get_chat_llm


SchemaT = TypeVar("SchemaT", bound=BaseModel)


def _extract_json_block(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    raise ValueError("模型响应中没有找到合法的 JSON 对象。")


async def invoke_json_schema(
    system_prompt: str,
    user_prompt: str,
    schema: type[SchemaT],
    temperature: float = 0.2,
) -> SchemaT:
    llm = get_chat_llm(temperature=temperature)
    format_hint = schema.model_json_schema()
    prompt = (
        f"{system_prompt}\n\n"
        "输出要求：\n"
        "- 只返回一个 JSON 对象。\n"
        "- 不要使用 markdown 代码块包裹 JSON。\n"
        "- 不要输出 JSON 之外的解释文字。\n"
        f"- 严格遵守这个 JSON Schema：{json.dumps(format_hint, ensure_ascii=False)}\n\n"
        f"任务说明：\n{user_prompt}"
    )
    response = await llm.ainvoke(prompt)
    payload = json.loads(_extract_json_block(response.content))
    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"结构化输出校验失败：{exc}") from exc
