"""
Tests for tools/implementations.py.

These aren't just happy-path checks — several tests here codify things
that were verified manually earlier in this project (e.g. the
calculator rejecting code injection, the file reader rejecting path
traversal). Turning those one-off manual checks into real tests means
a future change that accidentally reopens one of those holes gets
caught automatically, instead of relying on someone remembering to
re-check it by hand.

Run with:
    pytest tests/ -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.implementations import calculator, knowledge_base_lookup, read_file


class TestCalculator:
    def test_basic_arithmetic(self):
        assert calculator({"expression": "2 + 2"}) == {"result": 4}

    def test_order_of_operations(self):
        result = calculator({"expression": "(23 * 4) / 2 + 10"})
        assert result["result"] == 56.0

    def test_math_functions(self):
        assert calculator({"expression": "sqrt(144)"}) == {"result": 12.0}
        assert calculator({"expression": "pow(2, 10)"}) == {"result": 1024.0}
        assert calculator({"expression": "abs(-15)"}) == {"result": 15}
        assert calculator({"expression": "factorial(5)"}) == {"result": 120}

    def test_combined_function_and_arithmetic(self):
        result = calculator({"expression": "sqrt(144) * 3"})
        assert result["result"] == 36.0

    def test_division_by_zero_returns_error_not_exception(self):
        result = calculator({"expression": "1/0"})
        assert "error" in result
        assert "zero" in result["error"].lower()

    def test_empty_expression(self):
        result = calculator({"expression": ""})
        assert "error" in result

    # --- Security: these must never be allowed to execute ---

    def test_rejects_arbitrary_function_calls(self):
        result = calculator({"expression": '__import__("os").system("echo pwned")'})
        assert "error" in result

    def test_rejects_attribute_access_calls(self):
        # os.system(...) is an attribute-call, not a bare-name call —
        # must be rejected even though "system" alone isn't in the
        # allowlist and even if an attacker tried to alias it.
        result = calculator({"expression": 'os.system("echo pwned")'})
        assert "error" in result

    def test_rejects_eval_and_open(self):
        assert "error" in calculator({"expression": 'eval("1+1")'})
        assert "error" in calculator({"expression": 'open("/etc/passwd").read()'})

    def test_rejects_lambda(self):
        result = calculator({"expression": "(lambda: 1)()"})
        assert "error" in result

    def test_rejects_unknown_function_name(self):
        result = calculator({"expression": "not_a_real_function(5)"})
        assert "error" in result


class TestKnowledgeBaseLookup:
    def test_known_topic_found(self):
        result = knowledge_base_lookup({"query": "refund policy"})
        assert result["matches"]
        assert "refund policy" in result["matches"]

    def test_partial_match(self):
        # "refund" alone should still match "refund policy" via
        # substring matching
        result = knowledge_base_lookup({"query": "refund"})
        assert result["matches"]

    def test_unknown_topic_returns_no_matches(self):
        result = knowledge_base_lookup({"query": "unicorn husbandry"})
        assert result["matches"] == {}

    def test_empty_query(self):
        result = knowledge_base_lookup({"query": ""})
        assert "error" in result


class TestReadFile:
    def test_reads_existing_file(self):
        result = read_file({"filename": "sample_notes.txt"})
        assert "result" in result
        assert "Q1 2027" in result["result"]

    def test_missing_file_returns_error_not_exception(self):
        result = read_file({"filename": "does_not_exist.txt"})
        assert "error" in result

    def test_empty_filename(self):
        result = read_file({"filename": ""})
        assert "error" in result

    # --- Security: path traversal must never escape the workspace ---

    def test_rejects_path_traversal(self):
        result = read_file({"filename": "../../../etc/passwd"})
        assert "error" in result
        # Should report "not found" (basename-only lookup), not
        # actually have read a file outside the workspace.
        assert "passwd" in result["error"] or "not found" in result["error"].lower()

    def test_rejects_absolute_path_escape(self):
        result = read_file({"filename": "/etc/passwd"})
        assert "error" in result