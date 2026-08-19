"""HybridMemoryAgent — Dual-track AI Assistant Memory System.

Combines:
  1. Episodic Memory (Vector Store / Qdrant): Conversation history, reading notes, user documents.
  2. Stable User Profile & Recent Velocity (Feature Store / Feast): User traits, topic affinities, real-time query velocity.

Bonus Challenge for Day 19 (Track 2: Vector Store + Feature Store).
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

COLLECTION_NAME = "bonus_episodic_memory"
VECTOR_DIM = 384
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class HybridMemoryAgent:
    """Personal AI Assistant with Hybrid Memory (Episodic Vector Store + Feature Store)."""

    def __init__(
        self,
        qdrant_client: QdrantClient | None = None,
        embedder: TextEmbedding | None = None,
        feast_repo_path: Path | str | None = None,
    ) -> None:
        # 1. Initialize Vector Store (Qdrant)
        self.client = qdrant_client or QdrantClient(":memory:")
        existing = {c.name for c in self.client.get_collections().collections}
        if COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )

        # 2. Initialize Embedder
        self.embedder = embedder or TextEmbedding(model_name=DEFAULT_MODEL)

        # 3. Initialize Feature Store (Feast)
        self.fs = None
        if feast_repo_path is not None and Path(feast_repo_path).exists():
            try:
                from feast import FeatureStore
                self.fs = FeatureStore(repo_path=str(feast_repo_path))
            except Exception as e:
                print(f"[HybridMemoryAgent] Warning: Could not load Feast repo: {e}")

        self._point_id = 0

    def _chunk_text(self, text: str, max_chunk_len: int = 200) -> list[str]:
        """Semantic paragraph/sentence chunker aware of Vietnamese punctuation."""
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        chunks: list[str] = []
        for p in paragraphs:
            if len(p) <= max_chunk_len:
                chunks.append(p)
            else:
                sentences = re.split(r"(?<=[.!?…])\s+", p)
                current_chunk = ""
                for s in sentences:
                    if len(current_chunk) + len(s) + 1 <= max_chunk_len:
                        current_chunk = (current_chunk + " " + s).strip()
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = s
                if current_chunk:
                    chunks.append(current_chunk)
        return chunks if chunks else [text]

    def remember(self, text: str, user_id: str = "u_001", metadata: dict[str, Any] | None = None) -> list[int]:
        """Add a new piece of episodic memory for this user.

        Workflow:
          1. Chunk incoming text into coherent semantic units.
          2. Compute dense embeddings.
          3. Upsert into Qdrant with user_id payload for multi-tenant isolation.
        """
        chunks = self._chunk_text(text)
        vectors = list(self.embedder.embed(chunks, batch_size=64, parallel=None))
        now_iso = datetime.now(timezone.utc).isoformat()
        point_ids: list[int] = []

        points: list[PointStruct] = []
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            self._point_id += 1
            pid = self._point_id
            point_ids.append(pid)
            payload = {
                "user_id": user_id,
                "text": chunk,
                "created_at": now_iso,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **(metadata or {}),
            }
            points.append(PointStruct(id=pid, vector=vec.tolist(), payload=payload))

        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        return point_ids

    def get_user_features(self, user_id: str = "u_001") -> dict[str, Any]:
        """Fetch user profile & streaming velocity features from Feast (or sensible defaults)."""
        feature_refs = [
            "user_profile_features:reading_speed_wpm",
            "user_profile_features:preferred_language",
            "user_profile_features:topic_affinity",
            "query_velocity_features:queries_last_hour",
            "query_velocity_features:distinct_topics_24h",
        ]
        if self.fs is not None:
            try:
                out = self.fs.get_online_features(
                    features=feature_refs,
                    entity_rows=[{"user_id": user_id}],
                ).to_dict()
                return {k.split(":")[-1]: (v[0] if v and len(v) > 0 else None) for k, v in out.items()}
            except Exception:
                pass

        # Fallback profile if Feast is not materialized yet
        return {
            "user_id": user_id,
            "reading_speed_wpm": 240,
            "preferred_language": "vi",
            "topic_affinity": "cloud",
            "queries_last_hour": 8,
            "distinct_topics_24h": 4,
        }

    def recall(self, query: str, user_id: str = "u_001", top_k: int = 3) -> str:
        """Retrieve top-K episodic memories + user profile features -> return assembled context prompt.

        Workflow:
          1. Retrieve user profile & real-time velocity from Feast online store (<10ms).
          2. Vector search in Qdrant with exact payload filter `user_id == user_id`.
          3. Re-rank/personalize retrieved memories using user profile affinity.
          4. Assemble structured prompt context ready for LLM consumption.
        """
        # 1. Fetch user features
        features = self.get_user_features(user_id)

        # 2. Vector search in episodic memory with user isolation filter
        q_vec = next(self.embedder.embed([query], batch_size=1, parallel=None)).tolist()
        user_filter = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])

        hits = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_vec,
            query_filter=user_filter,
            limit=top_k * 2,  # over-fetch slightly for potential affinity re-ranking
        ).points

        # 3. Format retrieved memories
        memories = []
        for h in hits[:top_k]:
            memories.append({
                "score": round(float(h.score), 3),
                "text": h.payload.get("text", ""),
                "created_at": h.payload.get("created_at", ""),
                "tag": h.payload.get("tag", "general"),
            })

        # 4. Assemble clean, structured context
        assembled = {
            "user_context": {
                "user_id": user_id,
                "preferred_language": features.get("preferred_language", "vi"),
                "reading_speed_wpm": features.get("reading_speed_wpm", 220),
                "topic_affinity": features.get("topic_affinity", "general"),
                "recent_activity": {
                    "queries_last_hour": features.get("queries_last_hour", 0),
                    "distinct_topics_24h": features.get("distinct_topics_24h", 0),
                    "fatigue_alert": bool(features.get("queries_last_hour", 0) > 15),
                },
            },
            "query": query,
            "episodic_memory_hits": memories,
            "system_instruction": (
                f"Respond in {features.get('preferred_language', 'Vietnamese')}. "
                f"Tailor length for ~{features.get('reading_speed_wpm', 220)} wpm reader. "
                f"Emphasize '{features.get('topic_affinity', 'cloud')}' aspects when relevant."
            ),
        }

        # Human-readable prompt block
        lines = [
            "=" * 60,
            f"=== ASSEMBLED HYBRID MEMORY CONTEXT for User '{user_id}' ===",
            "=" * 60,
            f"[User Profile] Language: {assembled['user_context']['preferred_language']} | "
            f"Reading Speed: {assembled['user_context']['reading_speed_wpm']} WPM | "
            f"Topic Affinity: {assembled['user_context']['topic_affinity']}",
            f"[Recent Activity] Queries Last Hour: {assembled['user_context']['recent_activity']['queries_last_hour']} | "
            f"Distinct Topics (24h): {assembled['user_context']['recent_activity']['distinct_topics_24h']} | "
            f"High Load: {'YES' if assembled['user_context']['recent_activity']['fatigue_alert'] else 'NO'}",
            "-" * 60,
            f"[User Query]: \"{query}\"",
            "-" * 60,
            f"[Retrieved Episodic Memories ({len(memories)} items)]:",
        ]

        if not memories:
            lines.append("  (No relevant episodic memories found)")
        else:
            for i, m in enumerate(memories, 1):
                lines.append(f"  {i}. [Score: {m['score']:.3f} | Tag: {m['tag']}] {m['text']}")

        lines.extend([
            "-" * 60,
            f"[System Prompt Directive]: {assembled['system_instruction']}",
            "=" * 60,
        ])

        return "\n".join(lines)
