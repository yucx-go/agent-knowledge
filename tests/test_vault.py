"""Tests for Vault operations."""

import os
import shutil
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_knowledge.core.vault import Vault
from agent_knowledge.core.models import Source, Entity, CompiledTruth, Claim, Evidence

TEST_DIR = "/tmp/ak-test-vault"


@pytest.fixture(autouse=True)
def clean_vault():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    yield
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


class TestVaultInit:
    def test_init_creates_dirs(self):
        vault = Vault(TEST_DIR)
        vault.init()
        assert vault.is_initialized
        for d in vault.DIRS:
            assert (vault.root / d).is_dir()

    def test_init_idempotent(self):
        vault = Vault(TEST_DIR)
        vault.init()
        vault.init()  # should not raise
        assert vault.is_initialized

    def test_not_initialized_raises(self):
        vault = Vault("/tmp/ak-nonexistent-dir-99999")
        with pytest.raises(FileNotFoundError):
            vault.save_source(Source(id="x", title="x", content="x"))


class TestVaultSource:
    def test_save_and_load_source(self):
        vault = Vault(TEST_DIR)
        vault.init()

        source = Source(
            id="test-001",
            title="Test Source",
            content="This is a test document about MCP protocol.",
            source_type="text",
        )
        vault.save_source(source)

        loaded = vault.load_source("test-001")
        assert loaded is not None
        assert loaded.title == "Test Source"
        assert "MCP" in loaded.content

    def test_list_sources(self):
        vault = Vault(TEST_DIR)
        vault.init()

        for i in range(3):
            vault.save_source(Source(id=f"src-{i}", title=f"Source {i}", content=f"Content {i}"))

        sources = vault.list_sources()
        assert len(sources) == 3

    def test_load_nonexistent_returns_none(self):
        vault = Vault(TEST_DIR)
        vault.init()
        assert vault.load_source("nonexistent") is None


class TestVaultEntity:
    def test_save_and_load_entity(self):
        vault = Vault(TEST_DIR)
        vault.init()

        claim = Claim(text="MCP is the standard protocol", confidence=0.8)
        ct = CompiledTruth(summary="MCP protocol for agent communication", claims=[claim])
        entity = Entity(
            id="ent-mcp",
            name="MCP",
            entity_type="tech",
            compiled_truth=ct,
            backlinks=["src-001"],
        )
        vault.save_entity(entity)

        loaded = vault.load_entity("ent-mcp")
        assert loaded is not None
        assert loaded.name == "MCP"
        assert loaded.compiled_truth.summary == "MCP protocol for agent communication"
        assert len(loaded.compiled_truth.claims) == 1
        assert loaded.backlinks == ["src-001"]

    def test_list_entities(self):
        vault = Vault(TEST_DIR)
        vault.init()

        for name in ["MCP", "OpenClaw", "React"]:
            vault.save_entity(Entity(id=f"ent-{name.lower()}", name=name))

        entities = vault.list_entities()
        assert len(entities) == 3


class TestVaultCache:
    def test_source_cache_hit(self):
        vault = Vault(TEST_DIR)
        vault.init()
        source = Source(id="cached-1", title="Cached", content="cache test")
        vault.save_source(source)

        # First load populates cache; second load should come from cache
        loaded1 = vault.load_source("cached-1")
        # Remove file to prove cache is serving
        (vault.root / "sources" / "cached-1.yaml").unlink()
        loaded2 = vault.load_source("cached-1")
        assert loaded2 is not None
        assert loaded2.title == "Cached"

    def test_entity_cache_hit(self):
        vault = Vault(TEST_DIR)
        vault.init()
        entity = Entity(id="ent-c1", name="CacheTest")
        vault.save_entity(entity)

        (vault.root / "entities" / "ent-c1.yaml").unlink()
        loaded = vault.load_entity("ent-c1")
        assert loaded is not None
        assert loaded.name == "CacheTest"

    def test_clear_cache(self):
        vault = Vault(TEST_DIR)
        vault.init()
        vault.save_source(Source(id="s-clr", title="Clear", content="x"))

        assert "s-clr" in vault._source_cache
        vault.clear_cache()
        assert "s-clr" not in vault._source_cache

    def test_cache_eviction(self):
        vault = Vault(TEST_DIR, cache_maxsize=3)
        vault.init()

        for i in range(5):
            vault.save_source(Source(id=f"ev-{i}", title=f"E{i}", content=f"c{i}"))

        # maxsize=3, so only last 3 should be in cache
        assert len(vault._source_cache) == 3
        assert "ev-0" not in vault._source_cache
        assert "ev-1" not in vault._source_cache
        assert "ev-4" in vault._source_cache

    def test_save_updates_cache(self):
        vault = Vault(TEST_DIR)
        vault.init()
        s = Source(id="upd-1", title="V1", content="version 1")
        vault.save_source(s)
        assert vault._source_cache["upd-1"].title == "V1"

        s2 = Source(id="upd-1", title="V2", content="version 2")
        vault.save_source(s2)
        assert vault._source_cache["upd-1"].title == "V2"


class TestVaultStats:
    def test_stats_empty(self):
        vault = Vault(TEST_DIR)
        vault.init()
        stats = vault.stats()
        assert stats["sources"] == 0
        assert stats["entities"] == 0

    def test_stats_counts(self):
        vault = Vault(TEST_DIR)
        vault.init()
        vault.save_source(Source(id="s1", title="S1", content="test"))
        vault.save_entity(Entity(id="e1", name="E1"))

        stats = vault.stats()
        assert stats["sources"] == 1
        assert stats["entities"] == 1
