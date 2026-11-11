"""Composition root: the only place that picks concrete adapters."""

from __future__ import annotations

from codicil.adapters.onnx_embedder import OnnxEmbedder
from codicil.adapters.onnx_reranker import OnnxReranker
from codicil.adapters.openai_compat import OpenAICompatGenerator
from codicil.application.answer import AnswerConfig, AnswerService
from codicil.application.index import load_index
from codicil.application.retrieve import RetrieveConfig
from codicil.settings import Settings


def default_answer_config(s: Settings | None = None) -> AnswerConfig:
    s = s or Settings()
    return AnswerConfig(
        retrieve=RetrieveConfig(mode="hybrid", dense_only_languages=("hi",)), min_dense=s.min_dense, min_rerank=s.min_rerank, no_score_abstention=("hinglish",)
    )


def build_service(s: Settings) -> AnswerService:
    index = load_index(s.index_dir)
    embedder = OnnxEmbedder(s.models_dir / s.embedder)
    reranker = OnnxReranker(s.models_dir / "ms-marco-MiniLM-L-6-v2") if s.rerank else None
    generator = (
        OpenAICompatGenerator(s.llm_base_url, s.llm_api_key.get_secret_value(), s.llm_model, s.llm_timeout_s, s.llm_reasoning_effort, s.llm_send_temperature)
        if s.llm_enabled
        else None
    )
    store = None
    if s.qdrant_url:
        from codicil.adapters.qdrant_store import QdrantDenseStore  # optional dependency, imported only when configured

        store = QdrantDenseStore.connect(s.qdrant_url, s.qdrant_api_key.get_secret_value(), s.qdrant_collection)
    return AnswerService(index, embedder, reranker, generator, store)
