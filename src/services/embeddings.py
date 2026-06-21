from __future__ import annotations

from typing import Sequence

import requests

from src.config import settings


class DashScopeEmbeddings:
    def __init__(self) -> None:
        self.api_key = settings.embedding_api_key
        self.api_url = settings.embedding_api_url
        self.model = settings.embedding_model
        self.timeout = settings.llm_request_timeout
        self.batch_size = max(1, min(settings.embedding_batch_size, 10))

    def _request_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required for vector embeddings.")

        response = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": {"texts": list(texts)},
                "parameters": {"output_type": "dense"},
            },
            timeout=self.timeout,
        )
        if not response.ok:
            detail = response.text[:1000]
            raise requests.HTTPError(
                f"DashScope embedding request failed: status={response.status_code}, body={detail}",
                response=response,
            )
        payload = response.json()
        embeddings = payload.get("output", {}).get("embeddings", [])
        vectors = [item["embedding"] for item in embeddings if "embedding" in item]
        if len(vectors) != len(texts):
            raise RuntimeError(
                "DashScope embedding response count mismatch: "
                f"expected={len(texts)}, got={len(vectors)}"
            )
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        cleaned_texts = [text for text in texts if text and text.strip()]
        if not cleaned_texts:
            return []

        results: list[list[float]] = []
        for start in range(0, len(cleaned_texts), self.batch_size):
            batch = cleaned_texts[start : start + self.batch_size]
            results.extend(self._request_embeddings(batch))
        return results

    def embed_query(self, text: str) -> list[float]:
        embeddings = self._request_embeddings([text])
        return embeddings[0] if embeddings else []
