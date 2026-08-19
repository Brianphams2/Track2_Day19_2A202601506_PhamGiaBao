"""Đo khả năng phân loại kiểu truy vấn bằng tín hiệu BM25, không dùng nhãn khi tìm kiếm."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from rank_bm25 import BM25Okapi


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def main() -> None:
    docs = [json.loads(line) for line in (DATA / "corpus_vn.jsonl").open(encoding="utf-8")]
    golden = [json.loads(line) for line in (DATA / "golden_set.jsonl").open(encoding="utf-8")]
    tokenized = [(doc["title"] + " " + doc["text"]).lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized)

    scores_by_type: dict[str, list[float]] = defaultdict(list)
    for item in golden:
        confidence = float(max(bm25.get_scores(item["query"].lower().split())))
        scores_by_type[item["mode_hint"]].append(confidence)

    print(f"{'loại':12} {'min':>8} {'median':>8} {'max':>8}")
    for query_type in ("exact", "paraphrase", "mixed"):
        scores = sorted(scores_by_type[query_type])
        print(
            f"{query_type:12} {scores[0]:8.2f} "
            f"{scores[len(scores) // 2]:8.2f} {scores[-1]:8.2f}"
        )

    print("\nCác truy vấn theo confidence tăng dần:")
    for item in sorted(
        golden,
        key=lambda row: max(bm25.get_scores(row["query"].lower().split())),
    ):
        confidence = float(max(bm25.get_scores(item["query"].lower().split())))
        print(f"{confidence:6.2f}  {item['mode_hint']:10}  {item['query']}")


if __name__ == "__main__":
    main()
