"""Quick demo: init → ingest → dream → query → lint in 30 seconds."""

import sys
import os
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_knowledge.core.vault import Vault
from agent_knowledge.core.compiler import Compiler
from agent_knowledge.search.engine import SearchEngine
from agent_knowledge.dreaming.cycle import DreamCycle

DEMO_DIR = "/tmp/ak-demo"

# Clean slate
if os.path.exists(DEMO_DIR):
    shutil.rmtree(DEMO_DIR)

print("=" * 50)
print("agent-knowledge Demo")
print("=" * 50)

# 1. Init
print("\n📁 Step 1: Initialize vault")
vault = Vault(DEMO_DIR)
vault.init(lang="en")
print(f"   Vault created at {DEMO_DIR}")

# 2. Ingest
print("\n📥 Step 2: Ingest knowledge")
compiler = Compiler(vault)

sources = [
    (
        "We decided to use React instead of Vue for the dashboard project. "
        "The team has more React experience, and the component library ecosystem is richer. "
        "Risk: migration from existing Vue codebase will take 2 sprints.",
        "React vs Vue Decision"
    ),
    (
        "The API gateway was migrated from REST to MCP protocol. "
        "MCP provides better agent interoperability and standardized tool calling. "
        "Latency increased by 15ms but agent compatibility improved significantly.",
        "MCP Migration"
    ),
    (
        "Q1 revenue hit $4.2M, exceeding budget by 3.6%. "
        "International markets contributed 62% of total revenue. "
        "Gross margin at 11.3%, 0.9pp above budget target. "
        "Risk: currency fluctuation may inflate international numbers.",
        "Q1 Financial Results"
    ),
    (
        "The MCP protocol showed a critical bug: timeout handling causes silent failures. "
        "When MCP server doesn't respond within 30s, the agent retries indefinitely. "
        "Fix: added circuit breaker with 3-retry limit and exponential backoff.",
        "MCP Timeout Bug"
    ),
]

for text, title in sources:
    source = compiler.ingest(text, title=title)
    print(f"   ✅ {title}: {len(source.claims_extracted)} claims, {len(source.entities_extracted)} entities")

# 3. Dream
print("\n💤 Step 3: Run dream cycle")
cycle = DreamCycle(vault)
report = cycle.run(since_hours=9999)
print(f"   Light: {report.light_candidates} candidates")
print(f"   Deep:  {report.deep_promoted} promoted / {report.deep_discarded} discarded")

# 4. Query
print("\n🔍 Step 4: Query")
engine = SearchEngine(vault)

queries = ["MCP protocol", "revenue budget", "React decision"]
for q in queries:
    results = engine.search(q, top_k=3)
    print(f"\n   Q: \"{q}\"")
    for i, r in enumerate(results, 1):
        print(f"   [{i}] ({r.page_type}) {r.title} [score: {r.score}]")

# 5. Stats & Lint
print("\n📊 Step 5: Vault stats")
stats = vault.stats()
for k, v in stats.items():
    print(f"   {k}: {v}")

contradictions = compiler.detect_contradictions()
print(f"\n🔍 Contradictions: {len(contradictions)}")

# 6. Show compiled entity
print("\n📖 Step 6: Compiled entity example")
for eid in vault.list_entities():
    entity = vault.load_entity(eid)
    if entity and entity.compiled_truth.summary:
        print(f"\n   Entity: {entity.name} ({entity.entity_type})")
        print(f"   Compiled Truth: {entity.compiled_truth.summary[:200]}")
        print(f"   Claims: {len(entity.compiled_truth.active_claims)}")
        print(f"   Timeline: {len(entity.compiled_truth.timeline)} events")
        print(f"   Backlinks: {len(entity.backlinks)} sources")
        break

print(f"\n{'='*50}")
print("✅ Demo complete. Explore the vault:")
print(f"   ls {DEMO_DIR}/sources/")
print(f"   ls {DEMO_DIR}/entities/")
print(f"   cat {DEMO_DIR}/entities/*.yaml")
