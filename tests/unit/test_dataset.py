"""
Unit tests for dataset validation.

Tests cover:
  - Valid cases pass through
  - Missing required fields are rejected
  - Malformed types are rejected
  - Invalid difficulty is rejected
  - Auto-assigned IDs
  - Extra fields are tolerated
  - Empty dataset
  - Whitespace-only fields are rejected
  - All errors reported (not just the first)
"""

import pytest
from app.models.dataset import validate_dataset, TestCase


class TestTestCaseValidation:
    def test_valid_minimal_case(self):
        result = validate_dataset([{
            "question": "What is AI?",
            "ground_truth": "Artificial Intelligence.",
        }])
        assert result.valid_count == 1
        assert result.error_count == 0

    def test_valid_full_case(self):
        result = validate_dataset([{
            "question": "What is ML?",
            "ground_truth": "Machine learning is a subset of AI.",
            "context": "AI includes ML as a subfield.",
            "expected_keywords": ["machine", "learning", "subset"],
            "category": "ML Basics",
            "difficulty": "easy",
        }])
        assert result.valid_count == 1
        case = result.valid[0]
        assert case.category == "ML Basics"
        assert case.difficulty == "easy"
        assert case.expected_keywords == ["machine", "learning", "subset"]

    def test_missing_question_rejected(self):
        result = validate_dataset([{"ground_truth": "Some answer."}])
        assert result.error_count == 1
        assert result.valid_count == 0
        assert "question" in result.errors[0]["error"].lower()

    def test_missing_ground_truth_rejected(self):
        result = validate_dataset([{"question": "What is AI?"}])
        assert result.error_count == 1
        assert result.valid_count == 0

    def test_empty_question_rejected(self):
        result = validate_dataset([{"question": "", "ground_truth": "Something."}])
        assert result.error_count == 1

    def test_whitespace_only_question_rejected(self):
        result = validate_dataset([{"question": "   ", "ground_truth": "Something."}])
        assert result.error_count == 1

    def test_invalid_difficulty_rejected(self):
        result = validate_dataset([{
            "question": "What is AI?",
            "ground_truth": "AI is...",
            "difficulty": "very_hard",
        }])
        assert result.error_count == 1
        assert "difficulty" in result.errors[0]["error"].lower()

    def test_valid_difficulty_values(self):
        for diff in ["easy", "medium", "hard"]:
            result = validate_dataset([{
                "question": "What is AI?",
                "ground_truth": "AI is...",
                "difficulty": diff,
            }])
            assert result.valid_count == 1
            assert result.valid[0].difficulty == diff

    def test_difficulty_case_insensitive(self):
        result = validate_dataset([{
            "question": "What is AI?",
            "ground_truth": "AI is...",
            "difficulty": "EASY",
        }])
        assert result.valid_count == 1
        assert result.valid[0].difficulty == "easy"

    def test_auto_id_assigned(self):
        result = validate_dataset([{
            "question": "What is AI?",
            "ground_truth": "AI is...",
        }])
        assert result.valid[0].id == "case_0000"

    def test_explicit_id_preserved(self):
        result = validate_dataset([{
            "id": "my_custom_id",
            "question": "What is AI?",
            "ground_truth": "AI is...",
        }])
        assert result.valid[0].id == "my_custom_id"

    def test_extra_fields_tolerated(self):
        """Extra fields (like 'source', 'author') should not cause rejection."""
        result = validate_dataset([{
            "question": "What is AI?",
            "ground_truth": "AI is...",
            "source": "textbook",
            "author": "John",
            "year": 2024,
        }])
        assert result.valid_count == 1

    def test_empty_dataset(self):
        result = validate_dataset([])
        assert result.total == 0
        assert result.valid_count == 0
        assert result.error_count == 0

    def test_mixed_valid_and_invalid(self):
        result = validate_dataset([
            {"question": "What is AI?", "ground_truth": "AI is..."},  # valid
            {"ground_truth": "Missing question"},                       # invalid
            {"question": "What is ML?", "ground_truth": "ML is..."},   # valid
            {"question": "", "ground_truth": "Empty question"},         # invalid
        ])
        assert result.valid_count == 2
        assert result.error_count == 2
        assert result.total == 4

    def test_all_errors_collected(self):
        """Validation collects ALL errors, not just the first."""
        result = validate_dataset([
            {"ground_truth": "no question"},
            {"question": "no ground truth"},
            {"question": "What?", "ground_truth": "Valid", "difficulty": "extreme"},
        ])
        assert result.error_count == 3

    def test_expected_keywords_must_be_list(self):
        result = validate_dataset([{
            "question": "What is AI?",
            "ground_truth": "AI is...",
            "expected_keywords": "not a list",
        }])
        assert result.error_count == 1

    def test_expected_keywords_strips_empty_strings(self):
        result = validate_dataset([{
            "question": "What is AI?",
            "ground_truth": "AI is...",
            "expected_keywords": ["valid", "", "  ", "keyword"],
        }])
        assert result.valid_count == 1
        kw = result.valid[0].expected_keywords
        assert "" not in kw
        assert "valid" in kw
        assert "keyword" in kw
