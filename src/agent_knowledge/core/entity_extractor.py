"""Entity extraction — reverse-filtering pipeline.

Strategy: cast a wide net (extract any noun-shaped span), then filter out
known non-entities via curated stopword lists. This inverts the older
positive-rule approach ("match these specific patterns").

Why reverse filtering
---------------------
The space of "what is an entity" is unbounded — every domain (finance,
medical, legal, …) brings new vocabulary. A pattern catalogue can never
keep up.

The space of "what is NOT an entity" is bounded — modal verbs, pronouns,
demonstratives, common adverbs. These are language-level concerns, not
domain-level. A few hundred Chinese stopwords + a few dozen English ones
covers all domains.

Pipeline
--------
1. ``typed_extract``    — high-precision pass for emails / URLs / dates /
                          money / quantities / versions / IDs (already
                          shaped, just need to identify the type)
2. ``candidate_extract``— wide-net pass for noun-shaped strings:
                          CJK 2-6 char compounds, PascalCase, ALL-CAPS,
                          snake/kebab-case, Title-case English, quoted
3. ``filter``           — drop names that are entirely composed of
                          stopwords, start with sentence-fragment
                          particles, or are pure numbers
4. ``tag_type``         — optional suffix-based typing of survivors
                          (公司/部 → org; 市/省 → place; 报告/协议 →
                          document; 先生/老师 → person; …)
5. ``merge``            — typed entities take precedence on type when a
                          name appears in both passes

Public API
----------
``extract_entities(text, extra_patterns=None, extra_names=None)`` returns
a list of ``(name, entity_type)`` tuples. The Compiler turns these into
:class:`Entity` objects.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Stopwords
# ─────────────────────────────────────────────────────────────────────────────

# Chinese stopwords — covers ~95% of high-frequency function words and
# common abstract nouns. These are language-level (not domain-level) and
# should rarely need extension.
STOPWORDS_CN: frozenset[str] = frozenset({
    # ── Pronouns ──
    "我", "你", "他", "她", "它", "您", "咱",
    "我们", "你们", "他们", "她们", "它们", "您们", "咱们", "大家",
    "自己", "本人", "彼此", "互相",

    # ── Demonstratives ──
    "这", "那", "此", "该", "本",
    "这个", "那个", "这些", "那些", "这里", "那里", "这边", "那边",
    "这种", "那种", "这样", "那样", "如此", "这般",

    # ── Question words ──
    "什么", "啥", "怎么", "怎样", "为啥", "为什么", "如何",
    "多少", "几个", "哪里", "哪个", "哪些", "哪样",
    "谁", "谁的", "何时", "何处",

    # ── Time adverbs (no specific referent) ──
    "今天", "明天", "昨天", "前天", "后天", "今年", "明年", "去年",
    "今晚", "明晚", "昨晚", "今早", "明早",
    "现在", "目前", "当前", "如今", "此刻", "立刻", "马上", "随时",
    "以前", "以后", "之前", "之后", "从前", "未来", "将来",
    "上午", "下午", "中午", "凌晨", "早上", "晚上", "傍晚", "夜里",
    "近期", "最近", "最初", "最后", "最终", "终于", "始终", "从来",
    "经常", "总是", "一直", "曾经", "已经", "正在", "即将", "刚刚",
    "偶尔", "有时", "时常", "时刻", "时候", "时段",

    # ── Degree adverbs ──
    "非常", "十分", "极其", "特别", "格外", "比较", "稍微",
    "几乎", "差不多", "相当", "甚至", "尤其",

    # ── Quantity words (vague) ──
    "一些", "一点", "一下", "一定", "一切", "一共", "一同",
    "全部", "全都", "整个", "整体", "所有", "任何",
    "每个", "各个", "各种", "其他", "其它", "另一",
    "多个", "几个", "若干", "许多", "大量", "少量", "部分",

    # ── Conjunctions ──
    "因为", "由于", "所以", "因此", "因而", "故此",
    "如果", "假如", "假设", "要是", "倘若", "万一",
    "即使", "即便", "纵然", "哪怕", "纵使",
    "虽然", "尽管", "固然", "诚然",
    "然而", "但是", "可是", "不过", "只是", "却是",
    "并且", "而且", "况且", "再者", "另外", "此外", "其中",
    "或者", "还是", "不然", "否则",
    "于是", "然后", "接着", "随后", "继而",

    # ── Modal / aux verbs ──
    "可以", "可能", "应该", "应当", "需要", "必须", "一定",
    "能够", "想要", "希望", "愿意", "不能", "不会", "不要", "不必",
    "知道", "理解", "认为", "觉得", "感到", "记得",
    "建议", "推荐", "决定", "选择",

    # ── Common action verbs (2-char) ──
    # These act as natural segment boundaries in CJK runs — without them,
    # adjacent compounds like "系统记录凭证" stay glued together. Trade-off:
    # we lose them as standalone entities, but they're rarely entity-like
    # in their own right.
    "使用", "开发", "创建", "删除", "修改", "增加", "减少", "添加", "移除",
    "更新", "升级", "降级", "记录", "发布", "制作", "完成", "开始", "结束",
    "启动", "停止", "暂停", "恢复", "处理", "管理", "控制", "调整", "配置",
    "设置", "安装", "卸载", "工作", "学习", "研究", "调查", "分析", "评估",
    "实施", "执行", "运行", "操作", "实现", "制定", "编写", "编辑", "审核",
    "审查", "批准", "提供", "支持", "包含", "涉及", "提升", "改进", "优化",
    "确保", "保证", "保持", "维持", "维护", "保护", "监控", "监测", "检查",
    "确认", "验证", "测试", "对比", "比较", "查看", "查询", "搜索", "查找",
    "寻找", "发现", "识别", "判断", "区分", "区别", "匹配", "对应", "对接",
    "集成", "对齐", "同步", "传输", "传送", "发送", "接收", "下载", "上传",
    "分配", "归档", "归集", "拆分", "合并", "整合", "汇总",
    # More — interaction / leadership / collaboration
    "主持", "参加", "出席", "主导", "领导", "负责", "协助", "配合",
    "接待", "接受", "建立", "成立", "组建", "发起", "倡导", "推动",
    "合作", "协作", "交流", "沟通", "讨论", "商议", "协商",
    "推荐", "举荐", "提名", "授予", "任命", "聘任", "罢免",
    # State / occurrence
    "成为", "变成", "属于", "存在", "出现", "出炉", "生成",
    "形成", "构成", "组成", "包括",
    # Reading / writing
    "阅读", "查阅", "阅览", "撰写", "编排",
    # Request / response (common in software/business docs)
    "请求", "响应", "反馈", "答复", "回复", "回应",
    # Common qualifiers / states (NOT 1-char which would over-filter)
    "强制", "可选", "必要", "默认", "自动", "手动", "静默",
    "翻倍", "减半", "加速", "减速", "优先", "落后",
    "节省", "节约", "浪费", "增长", "下降", "提高", "降低",
    "全面", "部分", "完整", "局部", "总体", "整体",
    "直接", "间接", "实时", "异步", "同步",

    # ── Common prepositions (CN-style 2-char) ──
    "通过", "根据", "按照", "基于", "关于", "对于", "针对", "包括",
    "在于", "属于", "对应",

    # ── Common abstract / structural nouns (low entity value) ──
    "情况", "状况", "状态", "情形", "形势",
    "问题", "事情", "事项", "事件",
    "方法", "办法", "方式", "做法", "手段", "途径",
    "结果", "成果", "效果", "影响",
    "过程", "环节",
    "内容", "信息", "数据", "资料", "材料",
    "时间", "期间", "时期",
    "地方", "位置", "地点",
    "原因", "理由", "因素", "条件", "前提",
    "目标", "目的", "宗旨", "意图",
    "事务", "业务",
    "方面", "层面", "领域", "范围",

    # ── Descriptive adjectives (rarely entities standalone) ──
    "重要", "主要", "次要", "首要", "关键", "核心",
    "一般", "普通", "通常", "常规", "正常",
    "具体", "详细", "明确", "清楚", "清晰",
    "简单", "复杂", "困难", "容易",
    "全新", "完整", "完善", "完全",

    # ── Comparatives ──
    "更多", "更少", "更好", "更大", "更小",
    "最多", "最少", "最好", "最大", "最小",

    # ── Markers / hedges ──
    "比如", "例如", "包括", "即可", "也就是",
    "其实", "实际", "事实", "确实", "的确", "当然",
    "只是", "仅仅", "只有", "只要",

    # ── Spatial relatives ──
    "上面", "下面", "里面", "外面", "前面", "后面", "中间",
    "上方", "下方", "里头", "外头",

    # ── Filler / structural ──
    "之类", "等等", "之一", "之中", "之上", "之下",
    "之后", "之前",
})

# English stopwords — small, focused on what gets caught by Title-case /
# PascalCase / ALL-CAPS extractors when they appear at sentence starts.
STOPWORDS_EN: frozenset[str] = frozenset({
    # Articles
    "A", "An", "The",
    # Pronouns
    "I", "You", "He", "She", "It", "We", "They",
    "Me", "Him", "Her", "Us", "Them",
    "My", "Your", "His", "Our", "Their", "Its",
    "This", "That", "These", "Those",
    "Who", "Whom", "Whose", "Which", "What",
    # Conjunctions
    "And", "Or", "But", "Nor", "So", "Yet", "For",
    "If", "Then", "Because", "Since", "While", "Although", "Though",
    # Prepositions
    "In", "On", "At", "To", "From", "By", "With", "Of", "About",
    "As", "Into", "Like", "Through", "After", "Before", "Between",
    "Up", "Down", "Out", "Off", "Over", "Under", "Above", "Below",
    # Auxiliaries / modals
    "Is", "Are", "Was", "Were", "Be", "Been", "Being",
    "Have", "Has", "Had", "Do", "Does", "Did",
    "Can", "Could", "May", "Might", "Will", "Would",
    "Shall", "Should", "Must", "Ought",
    # Determiners / quantifiers
    "All", "Any", "Each", "Every", "Some", "Most", "Both",
    "Few", "Many", "More", "Less", "No", "None", "Other",
    # Adverbs commonly capitalised at sentence start
    "Here", "There", "Now", "Then", "Today", "Tomorrow", "Yesterday",
    "Always", "Never", "Often", "Sometimes", "Usually",
    "Just", "Only", "Also", "Still", "Yet", "Even",
    # Common verbs at sentence start
    "Get", "Got", "Make", "Made", "Take", "Took",
    "Use", "Used", "See", "Saw", "Know", "Knew",
    "Think", "Thought", "Want", "Wanted",
})

STOPWORDS_EN_LOWER = frozenset(s.lower() for s in STOPWORDS_EN)

# Single CJK characters that often start a sentence-fragment compound
# rather than an entity name. e.g. "是个" / "对于" / "给我".
_CJK_FRAGMENT_STARTERS: frozenset[str] = frozenset(
    "是在对给让把被由从向往朝再又也就还都甚却且乃即"
    "时若如即使能让若需有"
)

# Single-char CJK function words / particles. These act as natural
# segment boundaries when scanning CJK runs — what comes between two
# such characters is a candidate compound. They're separate from
# `_CJK_FRAGMENT_STARTERS` (which only block when at position 0) because
# these can break compounds anywhere they appear.
_CJK_SINGLE_PARTICLES: frozenset[str] = frozenset(
    "的了在是和或与及但也都把被由从来去上下"
    "于而对就更最又再已即可需才并很非如同至所"
)


# ─────────────────────────────────────────────────────────────────────────────
# Typed extractors (high-precision)
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (entity_type, list of compiled patterns).
# Order matters — earlier types win when a string matches multiple.
_TYPED_PATTERNS: list[tuple[str, list[re.Pattern]]] = [
    ("identifier", [
        # Email
        re.compile(r"\b([\w.+-]+@[\w-]+\.[\w.-]+)\b"),
        # URL
        re.compile(r"\b(https?://[^\s一-鿿]+)"),
        # Issue / ticket: JIRA-1234, ABC-5678
        re.compile(r"\b([A-Z]{2,5}-\d+)\b"),
        # Lark-style tokens
        re.compile(r"\b((?:tbl|ou_|oc_|om_|img_|file_)[a-zA-Z0-9]+)\b"),
    ]),
    ("time", [
        # ISO date
        re.compile(r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"),
        # Chinese full date / partial date
        re.compile(r"((?:\d{4}年)?\d{1,2}月\d{1,2}日)"),
        re.compile(r"(\d{4}年(?:\d{1,2}月)?)"),
        # Quarter
        re.compile(r"\b(Q[1-4]\s+\d{4}|\d{4}\s+Q[1-4])\b", re.IGNORECASE),
        # Fiscal year
        re.compile(r"\b(FY\s*\d{2,4}|\d{4}\s*财年)\b", re.IGNORECASE),
        # Time-of-day with seconds/minutes
        re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)\b"),
    ]),
    ("money", [
        # Currency-prefixed: $1,234.56 / ¥10万 / €500 / ￥100亿
        re.compile(r"([$¥€£￥]\s*\d[\d,]*(?:\.\d+)?(?:[KMB]|万|亿|千|百)?)"),
        # Currency-suffixed
        re.compile(
            r"\b(\d[\d,]*(?:\.\d+)?\s*(?:USD|EUR|GBP|JPY|CNY|RMB|元|美元|欧元|日元|英镑|港币|人民币))",
            re.IGNORECASE,
        ),
    ]),
    ("quantity", [
        # Percentage
        re.compile(r"(\d+(?:\.\d+)?%)"),
    ]),
    ("version", [
        # Software version: v2.3, V1.0.0. Lookbehind only excludes ASCII
        # letters — CJK is allowed before "v" (e.g. "发布v2.3"). The
        # greedy `(?:\.\d+){1,3}` caps the match at the last digit run.
        re.compile(r"(?<![A-Za-z])([vV]\d+(?:\.\d+){1,3})"),
        # Numbered designations: Strategy 5, Stage 3, 阶段 2, 步骤 1
        re.compile(
            r"\b((?:Strategy|Stage|Phase|Step|Level|Tier|策略|阶段|步骤)\s*\d+)\b",
            re.IGNORECASE,
        ),
    ]),
]


# ─────────────────────────────────────────────────────────────────────────────
# Candidate extractors (wide net)
# ─────────────────────────────────────────────────────────────────────────────

# Pre-compile once. Each pattern's group(1) (or full match if no groups)
# is taken as the candidate name. CJK extraction uses a separate function
# (``_cjk_candidates``) — naive regex over CJK runs splits compounds
# arbitrarily because there are no spaces between words.
_CANDIDATE_PATTERNS: list[re.Pattern] = [
    # PascalCase / CamelCase identifiers (≥2 capitalised segments)
    re.compile(r"(?<![A-Za-z])([A-Z][a-z]+(?:[A-Z][a-z]+)+)(?![A-Za-z])"),
    # Title + ALL-CAPS suffix (MySQL, PostgreSQL, OpenAPI, GraphQL, …)
    re.compile(r"(?<![A-Za-z])([A-Z][a-z]+[A-Z]{2,5})(?![a-z])"),
    # ALL-CAPS acronyms
    re.compile(r"(?<![A-Za-z])([A-Z]{2,8})(?![A-Za-z])"),
    # snake_case / kebab-case (must contain underscore or dash)
    re.compile(r"(?<![A-Za-z0-9_-])([a-z][a-z0-9]*(?:[_-][a-z0-9]+){1,5})(?![A-Za-z0-9_-])"),
    # Title-case sequences (English proper-noun phrases of 2-4 words)
    re.compile(r"\b((?:[A-Z][a-z]+\s+){1,3}[A-Z][a-z]+)\b"),
    # Quoted strings — restricted to 2–15 chars and reject internal
    # punctuation. This catches proper nouns ("iPhone X", 《红楼梦》) but
    # rejects slogans / sentences that happen to be quoted
    # ("对话即工作，让财务效率翻倍" — that's not an entity).
    re.compile(r'"([^"\n,，.。;；!！?？]{2,15})"'),
    re.compile(r"“([^”\n,，.。;；!！?？]{2,15})”"),
    re.compile(r"`([^`\n]{2,40})`"),
]

# Maximal CJK run: any contiguous sequence of CJK chars
_CJK_RUN_RE = re.compile(r"[一-鿿]+")

# Min/max length for a CJK candidate compound. Empirically, 2–4 char
# compounds account for almost all real Chinese named entities; 5+ char
# unbroken chunks are mostly sentence fragments the segmenter couldn't
# resolve (no stopword broke them up). Long chunks that DO end in a
# known suffix (公司/部/报告 etc.) get kept regardless via a separate
# check in :func:`_cjk_candidates`.
_CJK_MIN_LEN = 2
_CJK_MAX_LEN = 4
# When a chunk exceeds the cap but doesn't have a recognizable suffix,
# we keep just its leading and trailing 2–4 char windows as a recall
# fallback (avoids losing partial signal from over-long fragments).
_CJK_LONG_FALLBACK_WIN = 4


def _cjk_candidates(text: str) -> set[str]:
    """Extract CJK candidate compounds via stopword-based segmentation.

    Why this isn't ``re.findall(r'[一-鿿]{2,6}')``:
        Greedy non-overlapping regex chops compounds arbitrarily —
        ``"我们正在开发记忆系统"`` becomes ``["我们正在开发", "记忆系统"]``,
        keeping the first 6-char chunk regardless of word boundary.

    The fix: walk each CJK run; consume known stopwords (multi-char and
    single-char particles); whatever falls between consecutive stopwords
    is a candidate. This uses the stopword set as a free segmenter.

    Stopwords cover common verbs/aux/pronouns/conjunctions, so the
    boundary detection is good for general-domain text. Domain-specific
    compounds (``"凭证生成"``, ``"成本中心"``) survive intact because
    their characters aren't function words.
    """
    candidates: set[str] = set()

    for run_match in _CJK_RUN_RE.finditer(text):
        run = run_match.group(0)
        i = 0
        current = ""
        while i < len(run):
            # Try to consume the longest multi-char stopword starting at i
            matched_len = 0
            for length in range(min(4, len(run) - i), 1, -1):
                if run[i:i + length] in STOPWORDS_CN:
                    matched_len = length
                    break

            # Or a single-char particle
            if matched_len == 0 and run[i] in _CJK_SINGLE_PARTICLES:
                matched_len = 1

            if matched_len:
                _flush_chunk(candidates, current)
                current = ""
                i += matched_len
            else:
                current += run[i]
                i += 1

        # Flush trailing chunk
        _flush_chunk(candidates, current)

    return candidates


def _has_known_suffix(name: str) -> bool:
    """True if ``name`` ends with a recognised type suffix (org/place/etc)."""
    return any(pat.search(name) for pat, _ in _TYPE_SUFFIXES)


def _flush_chunk(candidates: set[str], chunk: str) -> None:
    """Add a CJK chunk to candidates with length-aware policy.

    * 2–4 chars: kept as-is.
    * 5+ chars with a known suffix (公司/部/报告 etc.): kept whole — it's
      likely a real institution / document name.
    * 5+ chars without suffix: kept ONLY as leading and trailing 4-char
      windows. Loses some recall but cuts a lot of sentence-fragment
      noise. The trailing window often captures the meaningful tail
      (``"高优先级规则"`` → ``"优先级规则"``? still noisy, but better
      than the whole 6-char fragment).
    """
    n = len(chunk)
    if n < _CJK_MIN_LEN:
        return
    if n <= _CJK_MAX_LEN:
        candidates.add(chunk)
        return
    # Long chunk
    if _has_known_suffix(chunk):
        candidates.add(chunk)
    else:
        # Recall-preserving fallback: keep edge windows
        candidates.add(chunk[:_CJK_LONG_FALLBACK_WIN])
        candidates.add(chunk[-_CJK_LONG_FALLBACK_WIN:])


# ─────────────────────────────────────────────────────────────────────────────
# Suffix-based type tagging
# ─────────────────────────────────────────────────────────────────────────────

# Each (compiled_pattern, entity_type) — checked in order. First match wins.
_TYPE_SUFFIXES: list[tuple[re.Pattern, str]] = [
    # CJK organisation suffixes
    (re.compile(
        r"(?:公司|集团|银行|学校|大学|学院|医院|协会|联盟|事务所|工厂|"
        r"实验室|研究院|研究所|工作室|部|局|委|处|司|院|厅|室)$"
    ), "org"),
    # CJK place suffixes
    (re.compile(r"(?:市|省|县|州|镇|村|区|国|岛|山|河|江|湖|海)$"), "place"),
    # CJK document suffixes
    (re.compile(
        r"(?:报告|报表|协议|合同|通知|公告|规范|条例|草案|方案|手册|"
        r"指南|流程|制度|准则|规划|文档|手稿)$"
    ), "document"),
    # CJK event suffixes
    (re.compile(r"(?:大会|峰会|论坛|会议|发布会|展会|赛事|庆典|节)$"), "event"),
    # CJK person honorifics
    (re.compile(
        r"(?:先生|女士|老师|博士|教授|总|主任|部长|经理|主席|院长|校长|同学)$"
    ), "person"),
    # English organisation suffix
    (re.compile(r"\s+(?:Inc|LLC|Corp|Co|Ltd|GmbH|AG|Pte|Plc)\.?$"), "org"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Filtering helpers
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=8192)
def _all_stopwords_cn(name: str) -> bool:
    """Return True if `name` decomposes entirely into CJK stopwords.

    Recursive decomposition: ``"我们这些"`` → ``"我们" + "这些"``, both
    stopwords → True. Bounded by ``len(name) <= 6`` from the candidate
    pattern, so the recursion is shallow.
    """
    if name in STOPWORDS_CN:
        return True
    for i in range(2, len(name)):
        left, right = name[:i], name[i:]
        if left in STOPWORDS_CN and _all_stopwords_cn(right):
            return True
    return False


def is_non_entity(name: str) -> bool:
    """Return True if ``name`` is clearly NOT an entity (filter out).

    The tests applied (in order):
      * length < 2
      * pure number (e.g. "12.34", "1,000")
      * exact match in CN/EN stopword sets
      * Title-cased English single word that's a stopword in lowercase
        (e.g. "The")
      * CJK string that decomposes entirely into stopword fragments
      * CJK string starting with a sentence-fragment particle
        (是/在/对/给/让/把/被/由/从/向/朝/再/又/也/就/还/都/甚/却/且/乃/即)
    """
    if not name or len(name) < 2:
        return True

    # Pure numeric (1,000 / 12.34 / 100)
    stripped = name.replace(",", "").replace(".", "").replace(" ", "")
    if stripped.isdigit():
        return True

    # Exact stopword
    if name in STOPWORDS_CN or name in STOPWORDS_EN:
        return True

    # Title-cased single English word that's a stopword
    if " " not in name and name.istitle() and name.lower() in STOPWORDS_EN_LOWER:
        return True

    # CJK-only string: check decomposition + fragment-starter
    is_cjk = all('一' <= c <= '鿿' for c in name)
    if is_cjk:
        if name[0] in _CJK_FRAGMENT_STARTERS:
            return True
        if _all_stopwords_cn(name):
            return True

    return False


def _strip_leading_article(name: str) -> str:
    """Strip leading ``The /A /An`` from English title-case sequences.

    ``"The Northern Hemisphere"`` → ``"Northern Hemisphere"``. Keeps the
    proper-noun core, removes the article that would otherwise pollute
    the entity name.
    """
    parts = name.split(" ", 1)
    if len(parts) == 2 and parts[0] in ("The", "A", "An"):
        return parts[1]
    return name


def tag_type(name: str) -> str:
    """Infer entity_type from suffix patterns. Default ``"named"``."""
    for pat, t in _TYPE_SUFFIXES:
        if pat.search(name):
            return t
    return "named"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_entities(
    text: str,
    extra_patterns: Optional[dict[str, list[str]]] = None,
    extra_names: Optional[list[str]] = None,
) -> list[tuple[str, str]]:
    """Run the full extraction pipeline.

    Args:
        text: Source text to extract from.
        extra_patterns: Optional ``{entity_type: [regex, ...]}`` for
            user-defined typed extractors. Merged after built-in
            typed patterns.
        extra_names: Optional list of literal strings that should always
            be promoted to entities when they appear (case-sensitive
            substring match). Useful for project-specific glossaries.

    Returns:
        List of ``(name, entity_type)`` tuples, deduplicated by name
        with first-seen entity_type winning.
    """
    if not text:
        return []

    # Track names with their assigned types
    typed: dict[str, str] = {}

    # ── Phase 1: typed extractors (high precision) ──
    for entity_type, patterns in _TYPED_PATTERNS:
        for pat in patterns:
            for m in pat.finditer(text):
                name = (m.group(1) if m.lastindex else m.group(0)).strip()
                if name and len(name) >= 2 and name not in typed:
                    typed[name] = entity_type

    # Drop typed candidates that are proper substrings of a longer typed
    # candidate of the same type. e.g. "2026年5月" gets superseded by
    # "2026年5月16日" — keep the longer, more specific one.
    typed_by_type: dict[str, list[str]] = {}
    for n, t in typed.items():
        typed_by_type.setdefault(t, []).append(n)
    for t, names in typed_by_type.items():
        names_sorted = sorted(names, key=len, reverse=True)
        for shorter in names_sorted[1:]:
            for longer in names_sorted:
                if longer is shorter:
                    continue
                if shorter in longer and shorter != longer:
                    typed.pop(shorter, None)
                    break

    # ── Phase 1b: user-defined typed patterns ──
    if extra_patterns:
        for entity_type, pats in extra_patterns.items():
            for raw in pats:
                try:
                    pat = re.compile(raw, re.IGNORECASE)
                except re.error:
                    continue
                for m in pat.finditer(text):
                    name = (m.group(1) if m.lastindex else m.group(0)).strip()
                    if name and len(name) >= 2 and name not in typed:
                        typed[name] = entity_type

    # ── Phase 1c: user-defined exact-match names ──
    if extra_names:
        for n in extra_names:
            if n and n in text and n not in typed:
                # Type-tag it via suffix; default 'named'
                typed[n] = tag_type(n)

    # ── Phase 2a: non-CJK candidates (regex-driven) ──
    candidates: set[str] = set()
    for pat in _CANDIDATE_PATTERNS:
        for m in pat.finditer(text):
            name = (m.group(1) if m.lastindex else m.group(0)).strip()
            if not name:
                continue
            # Strip leading articles for English Title-case sequences
            if " " in name:
                name = _strip_leading_article(name)
            if not is_non_entity(name):
                candidates.add(name)

    # ── Phase 2b: CJK candidates (stopword-segmented) ──
    for name in _cjk_candidates(text):
        if not is_non_entity(name):
            candidates.add(name)

    # ── Phase 3: merge — typed wins on type when name appears in both ──
    result: list[tuple[str, str]] = []
    seen: set[str] = set()

    for name, t in typed.items():
        if name not in seen:
            result.append((name, t))
            seen.add(name)

    for name in candidates:
        if name in seen:
            continue
        result.append((name, tag_type(name)))
        seen.add(name)

    return result
