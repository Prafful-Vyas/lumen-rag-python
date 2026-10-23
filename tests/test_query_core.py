import pytest

from lumen.adapters.completion import MockCompletionProvider
from lumen.adapters.reranker import parse_rerank_scores
from lumen.ports import ChatMessage, RerankCandidate
from lumen.query.core import (
    REFUSAL_MESSAGE,
    PromptChunk,
    TenantLeakError,
    assert_no_tenant_leak,
    build_prompt,
    normalize_query,
    reciprocal_rank_fusion,
    should_refuse,
    validate_citations,
)
from lumen.types import FusedChunk, RetrievedChunk, SourceRef


def chunk(cid: str, score: float, workspace: str = "ws-1") -> RetrievedChunk:
    return RetrievedChunk(cid, "doc-1", workspace, f"text-{cid}", 1, "", score)


def fused(cid: str, rrf: float = 0.5, rerank: float | None = None, text: str | None = None) -> FusedChunk:
    return FusedChunk(cid, "doc-1", "ws-1", text or f"text-{cid}", 1, "", 0.0, rrf_score=rrf, rerank_score=rerank)


def source(marker: int, chunk_id: str) -> SourceRef:
    return SourceRef(marker, chunk_id, "doc-1", "Acme MSA", marker, "", "")


class TestNormalizeQuery:
    def test_expands_known_acronyms_in_place(self):
        assert "MSA (Master Services Agreement)" in normalize_query("what does the MSA say about termination")

    def test_collapses_whitespace_and_trims(self):
        assert normalize_query("  what   about   SLA   uptime  ") == "what about SLA (Service Level Agreement) uptime"

    def test_leaves_non_acronym_text_untouched(self):
        assert normalize_query("who is responsible for data breaches") == "who is responsible for data breaches"

    def test_truncates_to_max_length(self):
        assert len(normalize_query("a" * 1000)) == 512


class TestReciprocalRankFusion:
    def test_chunk_in_both_lists_outranks_one_ranked_first_by_only_one(self):
        vector = [chunk("a", 0.9), chunk("b", 0.8), chunk("c", 0.7)]
        lexical = [chunk("b", 50), chunk("d", 40)]
        assert reciprocal_rank_fusion([vector, lexical], 60)[0].chunk_id == "b"

    def test_carries_rank_provenance(self):
        result = {f.chunk_id: f for f in reciprocal_rank_fusion([[chunk("a", 0.9)], [chunk("a", 10), chunk("z", 5)]])}
        assert (result["a"].vector_rank, result["a"].lexical_rank) == (1, 1)
        assert (result["z"].vector_rank, result["z"].lexical_rank) == (None, 2)

    def test_depends_on_rank_position_not_score_magnitude(self):
        high = reciprocal_rank_fusion([[chunk("a", 0.99), chunk("b", 0.01)], []])
        low = reciprocal_rank_fusion([[chunk("a", 0.02), chunk("b", 0.019)], []])
        assert [f.rrf_score for f in high] == [f.rrf_score for f in low]


class TestCitationValidation:
    def test_keeps_valid_citations_and_maps_them_to_chunk_ids(self):
        check = validate_citations(
            "The cap does not apply [1] except in section 9 [2].", [source(1, "a"), source(2, "b")]
        )
        assert check.invalid_count == 0
        assert check.citations == [{"marker": 1, "chunkId": "a"}, {"marker": 2, "chunkId": "b"}]
        assert check.cleaned_text == "The cap does not apply [1] except in section 9 [2]."

    def test_strips_hallucinated_marker(self):
        check = validate_citations("This is supported [1] and this is made up [9].", [source(1, "a")])
        assert check.invalid_count == 1
        assert check.citations == [{"marker": 1, "chunkId": "a"}]
        assert check.cleaned_text == "This is supported [1] and this is made up ."

    def test_no_sources_means_no_citations(self):
        check = validate_citations("No sources were used.", [])
        assert (check.invalid_count, check.citations) == (0, [])


class TestPromptBuilder:
    def test_renders_best_ranked_chunk_last_next_to_the_question(self):
        ranked = [PromptChunk(fused("best"), [1, 0, 0]), PromptChunk(fused("second"), [0, 1, 0])]
        messages, sources = build_prompt("q?", {"doc-1": "Acme"}, ranked, 6000, 0.95)
        assert [s.chunk_id for s in sources] == ["second", "best"]
        user = messages[1].content
        assert user.index("text-best") > user.index("text-second")
        assert user.endswith("Question: q?")

    def test_skips_near_duplicate_chunks(self):
        ranked = [
            PromptChunk(fused("a"), [1, 0]),
            PromptChunk(fused("a-copy"), [1, 0.001]),
            PromptChunk(fused("b"), [0, 1]),
        ]
        _, sources = build_prompt("q", {}, ranked, 6000, 0.95)
        assert {s.chunk_id for s in sources} == {"a", "b"}

    def test_always_includes_one_source_truncated_to_budget(self):
        ranked = [PromptChunk(fused("huge", text="word " * 5000), [1, 0])]
        _, sources = build_prompt("q", {}, ranked, 50, 0.95)
        assert len(sources) == 1
        assert len(sources[0].snippet) <= 240

    def test_unknown_document_gets_placeholder_title(self):
        _, sources = build_prompt("q", {}, [PromptChunk(fused("a"), [1])], 6000, 0.95)
        assert sources[0].document_title == "Untitled document"


class TestRefusalAndTenantGuard:
    def test_refuses_with_no_candidates(self):
        assert should_refuse(None, 0.15)

    def test_uses_rerank_score_over_rrf_score(self):
        assert should_refuse(fused("a", rrf=0.9, rerank=0.1), 0.15)
        assert not should_refuse(fused("a", rrf=0.01, rerank=0.5), 0.15)

    def test_falls_back_to_rrf_score_without_rerank(self):
        assert should_refuse(fused("a", rrf=0.01), 0.15)

    def test_tenant_guard_raises_on_foreign_chunk(self):
        leaked = reciprocal_rank_fusion([[chunk("a", 1), chunk("x", 1, workspace="ws-2")], []])
        with pytest.raises(TenantLeakError):
            assert_no_tenant_leak("ws-1", leaked)
        assert_no_tenant_leak("ws-1", [f for f in leaked if f.workspace_id == "ws-1"])


class TestRerankParsing:
    candidates = [RerankCandidate("a", "x"), RerankCandidate("b", "y")]

    def test_parses_json_embedded_in_prose(self):
        scores = parse_rerank_scores('Sure: [{"id":"b","score":0.9},{"id":"a","score":0.2}]', self.candidates)
        assert {r.id: r.score for r in scores} == {"a": 0.2, "b": 0.9}

    def test_ignores_unknown_ids_and_defaults_missing_to_zero(self):
        scores = parse_rerank_scores('[{"id":"zzz","score":1},{"id":"a","score":0.7}]', self.candidates)
        assert {r.id: r.score for r in scores} == {"a": 0.7, "b": 0.0}

    def test_garbage_falls_back_to_fused_order(self):
        scores = parse_rerank_scores("not json at all", self.candidates)
        assert [r.id for r in scores] == ["a", "b"] and scores[0].score > scores[1].score


async def test_mock_completion_cites_markers_present_in_prompt():
    mock = MockCompletionProvider()
    text, usage = await mock.complete([ChatMessage("user", "Sources:\n[1] a\n\n[2] b\n\n[3] c\n\nQuestion: why?")])
    assert "[1] [2]" in text and '"why?"' in text
    assert usage.prompt_tokens > 0
    no_sources, _ = await mock.complete([ChatMessage("user", "Question: why?")])
    assert no_sources == REFUSAL_MESSAGE
