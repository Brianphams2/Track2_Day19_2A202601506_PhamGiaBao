import pytest

from app.search import SearchHit, Searcher


def _hit(doc_id: str, score: float = 1.0) -> SearchHit:
    return SearchHit(doc_id=doc_id, title=doc_id, text=doc_id, score=score)


def test_rrf_uses_one_based_ranks_and_sums_both_retrievers(monkeypatch):
    searcher = Searcher()
    keyword = [_hit("shared"), _hit("keyword_only")]
    semantic = [_hit("shared"), _hit("semantic_only")]
    monkeypatch.setattr(searcher, "_search_keyword", lambda query, depth: keyword)
    monkeypatch.setattr(searcher, "_search_semantic", lambda query, depth: semantic)

    hits = searcher._search_hybrid("query", top_k=3, rrf_k=60)

    assert [hit.doc_id for hit in hits] == ["shared", "keyword_only", "semantic_only"]
    assert hits[0].score == pytest.approx(2 / 61)
    assert hits[1].score == pytest.approx(1 / 62)
    assert hits[2].score == pytest.approx(1 / 62)


def test_search_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown mode"):
        Searcher().search("query", mode="khong_hop_le")
