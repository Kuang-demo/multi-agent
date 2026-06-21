from __future__ import annotations

import json
from json import JSONDecodeError
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from src.config import get_chat_llm


SchemaT = TypeVar("SchemaT", bound=BaseModel)
MAX_RESPONSE_PREVIEW_CHARS = 500


def _preview(text: str) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= MAX_RESPONSE_PREVIEW_CHARS:
        return compact
    return compact[:MAX_RESPONSE_PREVIEW_CHARS].rstrip() + "..."


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
    schema_text = json.dumps(format_hint, ensure_ascii=False)
    prompt = (
        f"{system_prompt}\n\n"
        "输出要求：\n"
        "- 只返回一个 JSON 对象。\n"
        "- 不要使用 markdown 代码块包裹 JSON。\n"
        "- 不要输出 JSON 之外的解释文字。\n"
        f"- 严格遵守这个 JSON Schema：{schema_text}\n\n"
        f"任务说明：\n{user_prompt}"
    )
    response = await llm.ainvoke(prompt)
    content = str(getattr(response, "content", "") or "").strip()
    if not content:
        raise ValueError("模型响应为空，无法解析结构化 JSON。")

    try:
        return _parse_and_validate(content=content, schema=schema)
    except ValueError as first_error:
        repair_prompt = (
            "下面是一个模型输出，但它没有通过 JSON 解析或 JSON Schema 校验。\n"
            "请只返回修复后的 JSON 对象，不要输出解释文字，不要使用 markdown 代码块。\n\n"
            f"JSON Schema：{schema_text}\n\n"
            f"原始输出：\n{content}"
        )
        repair_response = await llm.ainvoke(repair_prompt)
        repaired_content = str(getattr(repair_response, "content", "") or "").strip()
        if not repaired_content:
            raise ValueError(
                f"结构化输出解析失败，且修复响应为空。首次错误：{first_error}"
            ) from first_error
        try:
            return _parse_and_validate(content=repaired_content, schema=schema)
        except ValueError as repair_error:
            raise ValueError(
                "结构化输出解析失败。"
                f"首次错误：{first_error}；"
                f"修复错误：{repair_error}；"
                f"原始响应片段：{_preview(content)}；"
                f"修复响应片段：{_preview(repaired_content)}"
            ) from repair_error


def _parse_and_validate(content: str, schema: type[SchemaT]) -> SchemaT:
    try:
        json_text = _extract_json_block(content)
        payload = json.loads(json_text)
    except JSONDecodeError as exc:
        raise ValueError(
            f"模型响应不是合法 JSON：{exc}. 响应片段：{_preview(content)}"
        ) from exc
    except ValueError as exc:
        raise ValueError(f"{exc} 响应片段：{_preview(content)}") from exc

    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(
            f"结构化输出校验失败：{exc}. 响应片段：{_preview(content)}"
        ) from exc
