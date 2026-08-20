"""
Unit tests for text normalization and LLM judge output parsing.
"""

import pytest
from app.services.evaluation_service import normalize, extract_keywords
from app.services.llm_judge import _parse_and_validate


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------
class TestNormalizeEdgeCases:
    def test_tabs_collapsed(self):
        assert normalize("hello\tworld") == "hello world"

    def test_multiple_spaces(self):
        assert normalize("a    b    c") == "a b c"

    def test_leading_trailing_stripped(self):
        assert normalize("  hello  ") == "hello"

    def test_mixed_case(self):
        assert normalize("HELLO World") == "hello world"

    def test_unicode_preserved(self):
        # Non-ASCII preserved but lowercased where possible
        result = normalize("  Über  ")
        assert result == "über"

    def test_empty_string(self):
        assert normalize("") == ""

    def test_only_whitespace(self):
        assert normalize("   \t\n  ") == ""


# ---------------------------------------------------------------------------
# LLM judge output parsing
# ---------------------------------------------------------------------------
class TestParseAndValidate:
    def _valid_json(self, **overrides):
        base = {
            "correctness": 0.9,
            "relevance": 0.85,
            "completeness": 0.8,
            "faithfulness": 0.95,
            "reason": "Good answer.",
        }
        base.update(overrides)
        import json
        return json.dumps(base)

    def test_valid_output_parsed(self):
        result = _parse_and_validate(self._valid_json())
        assert result["correctness"] == 0.9
        assert result["reason"] == "Good answer."

    def test_scores_rounded_to_4_places(self):
        import json
        raw = json.dumps({
            "correctness": 0.912345678,
            "relevance": 0.8,
            "completeness": 0.8,
            "faithfulness": 0.8,
            "reason": "test",
        })
        result = _parse_and_validate(raw)
        assert result["correctness"] == round(0.912345678, 4)

    def test_json_in_markdown_fences_extracted(self):
        raw = '```json\n{"correctness":0.9,"relevance":0.8,"completeness":0.8,"faithfulness":0.8,"reason":"ok"}\n```'
        result = _parse_and_validate(raw)
        assert result["correctness"] == 0.9

    def test_json_embedded_in_prose_extracted(self):
        raw = 'Here is my evaluation: {"correctness":0.9,"relevance":0.8,"completeness":0.8,"faithfulness":0.8,"reason":"ok"} end.'
        result = _parse_and_validate(raw)
        assert result["correctness"] == 0.9

    def test_missing_key_raises_valueerror(self):
        import json
        raw = json.dumps({
            "correctness": 0.9,
            "relevance": 0.8,
            # missing completeness, faithfulness, reason
        })
        with pytest.raises(ValueError, match="missing keys"):
            _parse_and_validate(raw)

    def test_score_out_of_range_raises(self):
        import json
        raw = json.dumps({
            "correctness": 1.5,  # out of range
            "relevance": 0.8,
            "completeness": 0.8,
            "faithfulness": 0.8,
            "reason": "test",
        })
        with pytest.raises(ValueError, match="out of range"):
            _parse_and_validate(raw)

    def test_non_json_raises_valueerror(self):
        with pytest.raises(ValueError, match="non-JSON"):
            _parse_and_validate("This is just plain text with no JSON.")

    def test_wrong_type_for_score_raises(self):
        import json
        raw = json.dumps({
            "correctness": "high",  # string instead of number
            "relevance": 0.8,
            "completeness": 0.8,
            "faithfulness": 0.8,
            "reason": "test",
        })
        with pytest.raises(ValueError, match="must be a number"):
            _parse_and_validate(raw)

    def test_reason_must_be_string(self):
        import json
        raw = json.dumps({
            "correctness": 0.9,
            "relevance": 0.8,
            "completeness": 0.8,
            "faithfulness": 0.8,
            "reason": 42,  # not a string
        })
        with pytest.raises(ValueError, match="reason"):
            _parse_and_validate(raw)

    def test_scores_at_boundaries_valid(self):
        import json
        raw = json.dumps({
            "correctness": 0.0,
            "relevance": 1.0,
            "completeness": 0.0,
            "faithfulness": 1.0,
            "reason": "boundary",
        })
        result = _parse_and_validate(raw)
        assert result["correctness"] == 0.0
        assert result["relevance"] == 1.0
