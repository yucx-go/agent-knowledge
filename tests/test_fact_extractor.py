"""Tests for incidental fact extractor.

Test cases derived from real LongMemEval-S failure cases — the patterns
that BM25 over full sessions failed to retrieve.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_knowledge.search.fact_extractor import extract_incidental_facts


class TestByTheWay:
    """The most important pattern — explicit insertion marker."""

    def test_simple_by_the_way(self):
        text = "Need cocktail recipes. By the way, I've been using Spotify lately."
        facts = extract_incidental_facts(text)
        assert any("Spotify" in f for f in facts)

    def test_by_the_way_with_comma(self):
        text = "Looking for advice. By the way, I planted 12 tomato saplings."
        facts = extract_incidental_facts(text)
        assert any("tomato" in f.lower() for f in facts)

    def test_by_the_way_no_comma(self):
        text = "Looking for advice. By the way I just got a smoker."
        facts = extract_incidental_facts(text)
        assert any("smoker" in f.lower() for f in facts)

    def test_by_the_way_capital(self):
        text = "Some long question here. By The Way, I joined a debate team in high school."
        facts = extract_incidental_facts(text)
        assert any("debate team" in f.lower() for f in facts)


class TestIJustVerb:
    """First-person recent action with specific verbs."""

    def test_i_just_got(self):
        text = "I'm looking for BBQ sauce recipes. I just got a smoker today."
        facts = extract_incidental_facts(text)
        assert any("smoker" in f.lower() for f in facts)

    def test_i_just_signed(self):
        text = "Need a contract template. I just signed a contract with my first client today."
        facts = extract_incidental_facts(text)
        assert any("client" in f.lower() for f in facts)

    def test_i_just_completed(self):
        text = "Looking for bike trails. I just completed the Spring Sprint Triathlon today."
        facts = extract_incidental_facts(text)
        assert any("triathlon" in f.lower() for f in facts)

    def test_i_recently_attended(self):
        text = "Talking about gardening. I recently attended a gardening workshop on companion planting."
        facts = extract_incidental_facts(text)
        assert any("workshop" in f.lower() for f in facts)


class TestIveBeenVerb:
    """Ongoing state — 'I've been using/practicing/etc'."""

    def test_ive_been_using(self):
        text = "Looking for music. I've been using Apple Music for the past few months."
        facts = extract_incidental_facts(text)
        assert any("apple music" in f.lower() for f in facts)

    def test_i_have_been_practicing(self):
        text = "Looking for theory tips. I have been practicing guitar for 30 minutes daily."
        facts = extract_incidental_facts(text)
        assert any("guitar" in f.lower() for f in facts)


class TestTemporalAnchors:
    """Time-anchored gold pattern for temporal-reasoning questions."""

    def test_yesterday_i(self):
        text = "Talking about pets. Yesterday I adopted a kitten from the shelter."
        facts = extract_incidental_facts(text)
        assert any("kitten" in f.lower() for f in facts)

    def test_last_week_i(self):
        text = "Considering options. Last week I attended my cousin's wedding in Boston."
        facts = extract_incidental_facts(text)
        assert any("wedding" in f.lower() for f in facts)

    def test_two_weeks_ago(self):
        text = "Need recommendations. Two weeks ago I went to the Museum of Modern Art for a Cubism tour."
        facts = extract_incidental_facts(text)
        assert any("museum" in f.lower() for f in facts)


class TestNoFalsePositives:
    """Make sure we don't match meta-utterances or generic statements."""

    def test_i_just_want(self):
        # "I just want" should not match — it's a meta-utterance, not a fact
        text = "I just want to know more about Python."
        facts = extract_incidental_facts(text)
        assert not any("want" in f.lower() and "python" in f.lower() for f in facts)

    def test_empty_text(self):
        assert extract_incidental_facts("") == []

    def test_no_markers(self):
        text = "Python is a great programming language. Many people use it."
        facts = extract_incidental_facts(text)
        assert facts == []

    def test_short_fragment_filtered(self):
        # Below MIN_FACT_CHARS threshold — pattern matches but extracted text too short
        text = "By the way, OK."
        facts = extract_incidental_facts(text)
        assert facts == []


class TestRealFailureCases:
    """Tests built from actual LongMemEval-S failure cases."""

    def test_spotify_case(self):
        text = (
            "user: I'm looking for some concert recommendations. I just got back "
            "from seeing The 1975 live and I'm itching to see more shows. "
            "By the way, I've been using Spotify lately to discover new artists."
        )
        facts = extract_incidental_facts(text)
        assert any("spotify" in f.lower() for f in facts)

    def test_smoker_case(self):
        text = (
            "user: I'm looking for some new BBQ sauce recipes to try out. "
            "I've been experimenting with making my own from scratch. "
            "By the way, I just got a smoker today and I'm excited to try it."
        )
        facts = extract_incidental_facts(text)
        assert any("smoker" in f.lower() for f in facts)

    def test_tomato_case(self):
        text = (
            "user: I'm trying to figure out what type of fertilizer to use. "
            "I attended a gardening workshop recently where I learned about companion planting. "
            "Two weeks ago I planted 12 new tomato saplings in the back row."
        )
        facts = extract_incidental_facts(text)
        # Either pattern should catch it
        assert any("tomato" in f.lower() for f in facts)

    def test_dedup_identical_matches(self):
        # Same fact text mentioned twice should dedupe within a pattern.
        # Different patterns may both match — that's fine, but each
        # identical extraction should only appear once.
        text = (
            "By the way, I've been using Spotify lately. "
            "Some other content. "
            "By the way, I've been using Spotify lately."
        )
        facts = extract_incidental_facts(text)
        # Dedup ensures we don't have 4 entries (2 patterns × 2 occurrences)
        # — at most one per unique pattern capture
        assert len(facts) <= 2
        # Each unique fact appears once
        assert len(facts) == len(set(f.lower() for f in facts))
