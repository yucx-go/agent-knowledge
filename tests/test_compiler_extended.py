"""Extended tests for Knowledge Compiler — prose extraction, entity extraction edge cases."""

import os
import shutil
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_knowledge.core.vault import Vault
from agent_knowledge.core.compiler import Compiler
from agent_knowledge.core.models import ClaimStatus

TEST_DIR = "/tmp/ak-test-compiler-ext"


@pytest.fixture(autouse=True)
def clean_vault():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    vault = Vault(TEST_DIR)
    vault.init()
    yield vault
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


class TestProseExtraction:
    """Tests for _extract_prose_claims with decision/fact/risk signals."""

    def test_decision_signal_prose(self, clean_vault):
        compiler = Compiler(clean_vault)
        text = """## Summary
We decided to migrate from REST API to MCP protocol for all agent communication.
The team selected React over Vue for the frontend rewrite.
"""
        source = compiler.ingest(text, title="Decisions")
        # Should extract prose claims with decision signals
        assert len(source.claims_extracted) >= 1

    def test_fact_signal_prose(self, clean_vault):
        compiler = Compiler(clean_vault)
        text = """## 报告
Revenue increased by 15.3% this quarter compared to last year.
Total cost was $2,500,000 with a 完成率 of 74.5%.
"""
        source = compiler.ingest(text, title="Facts")
        assert len(source.claims_extracted) >= 1

    def test_risk_signal_prose(self, clean_vault):
        compiler = Compiler(clean_vault)
        text = """## Analysis
There is a critical risk that the production database may fail under load.
A warning was issued about the authentication bug in the login module.
"""
        source = compiler.ingest(text, title="Risks")
        assert len(source.claims_extracted) >= 1

    def test_no_signal_prose_skipped(self, clean_vault):
        compiler = Compiler(clean_vault)
        text = """## Random
The weather today is pleasant and comfortable for outdoor activities.
Birds are singing in the park near the office building entrance area.
"""
        source = compiler.ingest(text, title="NoSignal")
        # Low-signal prose in non-special section should be skipped
        assert len(source.claims_extracted) == 0

    def test_prose_in_high_value_section(self, clean_vault):
        compiler = Compiler(clean_vault)
        text = """## 决策
这个架构方案经过团队讨论后最终确定使用微服务模式。
"""
        source = compiler.ingest(text, title="Section")
        # In 决策 section, even without explicit signal keywords, prose gets extracted
        assert len(source.claims_extracted) >= 1

    def test_short_prose_ignored(self, clean_vault):
        compiler = Compiler(clean_vault)
        text = """## Notes
Short line.
Also brief.
But this is a sufficiently long sentence that should be evaluated for signals about a critical failure in the system.
"""
        source = compiler.ingest(text, title="Short")
        # Short sentences < 15 chars should be ignored
        # Only the long one with risk signal should be extracted
        assert len(source.claims_extracted) >= 1


class TestEntityExtractionEdgeCases:
    def test_empty_text(self, clean_vault):
        compiler = Compiler(clean_vault)
        source = compiler.ingest("", title="Empty")
        assert len(source.entities_extracted) == 0

    def test_pure_headers_only(self, clean_vault):
        compiler = Compiler(clean_vault)
        text = "# Title\n## Section\n### Subsection"
        source = compiler.ingest(text, title="Headers")
        assert len(source.claims_extracted) == 0

    def test_long_text_extraction(self, clean_vault):
        compiler = Compiler(clean_vault)
        # Very long text with repeated entities
        text = "## 决策\n" + "\n".join(
            f"- 决策 {i}: MCP 协议配置项 config_option_{i} 需要更新"
            for i in range(50)
        )
        source = compiler.ingest(text, title="Long")
        assert len(source.claims_extracted) == 50

    def test_pascal_case_entity(self, clean_vault):
        compiler = Compiler(clean_vault)
        text = "## 决策\n- 使用 OpenClaw 和 SearchEngine 进行知识管理"
        source = compiler.ingest(text, title="Pascal")
        # OpenClaw and SearchEngine are PascalCase entities
        assert len(source.entities_extracted) >= 1

    def test_snake_case_entity(self, clean_vault):
        compiler = Compiler(clean_vault)
        text = "## 决策\n- collapsible_panel 的 height 参数必须带 px 单位"
        source = compiler.ingest(text, title="Snake")
        assert len(source.entities_extracted) >= 1

    def test_acronym_entity(self, clean_vault):
        compiler = Compiler(clean_vault)
        text = "## 学习\n- BM25 和 RRF 都是信息检索中的重要算法"
        source = compiler.ingest(text, title="Acronym")
        # BM25 and RRF should be detected as entities
        entity_ids = set(source.entities_extracted)
        assert len(entity_ids) >= 1

    def test_cjk_compound_entity(self, clean_vault):
        compiler = Compiler(clean_vault)
        text = "## 决策\n- 记忆系统和知识平台需要整合"
        source = compiler.ingest(text, title="CJK")
        # 记忆系统 and 知识平台 should match CJK compound patterns
        assert len(source.entities_extracted) >= 1

    def test_entity_config_file(self, clean_vault):
        """Test custom entity config from .ak-entities.yaml."""
        import yaml
        config = {
            "entities": ["CustomProject", "TestEntity"],
            "patterns": {"custom": [r"PROJ-\d+"]},
        }
        config_path = clean_vault.root / ".ak-entities.yaml"
        config_path.write_text(yaml.dump(config), encoding="utf-8")

        compiler = Compiler(clean_vault)
        text = "## 决策\n- CustomProject 的 PROJ-123 任务已完成"
        source = compiler.ingest(text, title="Config")
        assert len(source.entities_extracted) >= 1


class TestClaimConsolidation:
    def test_duplicate_claims_superseded(self, clean_vault):
        compiler = Compiler(clean_vault)
        # Ingest same claim twice
        compiler.ingest("## 决策\n- MCP 协议是标准通信方案", title="Doc1")
        compiler.ingest("## 决策\n- MCP 协议是标准通信方案", title="Doc2")

        # Check that at least one entity has superseded claims
        for eid in clean_vault.list_entities():
            entity = clean_vault.load_entity(eid)
            if entity and entity.name == "MCP":
                statuses = [c.status for c in entity.compiled_truth.claims]
                # Should have some claims, potentially with SUPERSEDED
                assert len(entity.compiled_truth.claims) >= 1
                break

    def test_compile_summary_groups(self, clean_vault):
        compiler = Compiler(clean_vault)
        text = """## 决策
- MCP 协议是 Agent 通信标准
## 学习
- MCP 延迟比预期低
## 待办
- MCP 文档需要更新
"""
        compiler.ingest(text, title="MultiSection")
        for eid in clean_vault.list_entities():
            entity = clean_vault.load_entity(eid)
            if entity and entity.name == "MCP":
                assert entity.compiled_truth.summary != ""
                break


class TestDeriveTitle:
    def test_derive_from_header(self, clean_vault):
        compiler = Compiler(clean_vault)
        source = compiler.ingest("# My Great Title\nSome content here.", title="")
        assert source.title == "My Great Title"

    def test_derive_from_short_text(self, clean_vault):
        compiler = Compiler(clean_vault)
        source = compiler.ingest("ab\ncd\nThis is the real first line of the document", title="")
        assert "real first line" in source.title
