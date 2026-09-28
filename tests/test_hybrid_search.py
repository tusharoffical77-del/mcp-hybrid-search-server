"""
Tests for tools/hybrid_search.py.

Includes a regression test for a real bug found during manual testing:
search results used to be truncated to 300 characters, which caused
Claude to give incomplete answers (e.g. cutting off the compensation
details in performance_reviews.txt). That was fixed by returning full
document text instead. test_results_are_not_truncated below makes
sure that fix can't silently regress.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.hybrid_search import HybridSearchIndex


@pytest.fixture(scope="module")
def index():
    # Built once per test module, not per test — building the index
    # (TF-IDF + SVD fit) is the expensive part, and the document set
    # doesn't change between tests.
    return HybridSearchIndex()


class TestHybridSearch:
    def test_index_loads_all_documents(self, index):
        assert len(index.doc_names) == 5
        assert "onboarding.txt" in index.doc_names
        assert "remote_work_policy.txt" in index.doc_names

    def test_exact_keyword_match(self, index):
        results = index.search("VPN access", top_k=1)
        assert results[0]["document"] == "onboarding.txt"

    def test_semantic_match_no_exact_overlap(self, index):
        # "working from home" shares almost no exact words with
        # remote_work_policy.txt's actual text — this only works if
        # the semantic (LSA) half of the hybrid search is doing its job.
        results = index.search("working from home", top_k=1)
        assert results[0]["document"] == "remote_work_policy.txt"

    def test_another_semantic_match(self, index):
        # "when do I get a raise" vs. "Compensation adjustments...
        # communicated" — different wording, same topic.
        results = index.search("when do I get a raise", top_k=1)
        assert results[0]["document"] == "performance_reviews.txt"

    def test_results_are_not_truncated(self, index):
        # Regression test: results used to be cut to 300 chars, which
        # dropped real content (e.g. the compensation-timing sentence
        # in performance_reviews.txt). Full documents here are all
        # under ~600 chars, so requiring the full known closing
        # sentence to be present verifies nothing gets cut off.
        results = index.search("performance reviews", top_k=1)
        assert "communicated within two" in results[0]["text"]

    def test_top_k_respected(self, index):
        results = index.search("policy", top_k=2)
        assert len(results) == 2

    def test_top_k_cannot_exceed_document_count(self, index):
        results = index.search("policy", top_k=100)
        assert len(results) == 5  # only 5 documents exist

    def test_results_have_expected_fields(self, index):
        results = index.search("security", top_k=1)
        assert "document" in results[0]
        assert "score" in results[0]
        assert "text" in results[0]