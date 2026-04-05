"""
Seed data and test case definitions for the Memory Pro Extended Benchmark.
100 test cases across Skill Registry (35), Collision Engine (35), E2E Recall (30).
"""

SOURCE_TAG = "extended-bench"

# ---------------------------------------------------------------------------
# Skill seed data for Phase 2 (promoted ACTIVE skills used in matching tests)
# ---------------------------------------------------------------------------

SKILL_SEEDS = [
    {
        "name": "Docker网络调试",
        "content": "排查Docker容器间网络不通问题的完整流程",
        "procedures": "1. docker network ls 查看网络列表\n2. docker inspect <container> 检查网络配置\n3. docker exec -it <c> ping <target> 测试连通性\n4. 检查iptables规则\n5. 查看docker-compose网络定义",
        "tags": ["devops", "docker", "networking"],
        "prerequisites": "需要docker daemon运行中",
        "applicable_scenarios": "容器间通信失败、DNS解析失败、端口映射问题",
    },
    {
        "name": "Python性能优化",
        "content": "Python应用性能调优方法论",
        "procedures": "1. cProfile定位热点函数\n2. line_profiler逐行分析\n3. 数据结构优化(dict/set替代list查找)\n4. 使用生成器减少内存\n5. 考虑Cython/Numba加速",
        "tags": ["python", "performance", "optimization"],
        "applicable_scenarios": "API响应慢、批处理任务耗时长、内存占用高",
    },
    {
        "name": "用户访谈技巧",
        "content": "产品经理进行用户深度访谈的方法",
        "procedures": "1. 准备开放式问题清单\n2. 建立信任(先聊轻松话题)\n3. 追问'为什么'深挖动机\n4. 记录原话不做解读\n5. 总结确认理解",
        "tags": ["product", "research", "interview"],
        "applicable_scenarios": "新功能需求调研、用户痛点发掘、产品改进验证",
    },
    {
        "name": "React状态管理",
        "content": "React应用中状态管理的最佳实践",
        "procedures": "1. 区分local vs global state\n2. 简单场景用useState+Context\n3. 复杂场景用Zustand/Jotai\n4. 避免prop drilling超过3层\n5. 使用React.memo防止不必要渲染",
        "tags": ["frontend", "react", "state-management"],
        "applicable_scenarios": "组件间数据共享、表单状态管理、缓存管理",
    },
    {
        "name": "PostgreSQL索引优化",
        "content": "PostgreSQL查询性能优化之索引策略",
        "procedures": "1. EXPLAIN ANALYZE找慢查询\n2. 为WHERE/JOIN列创建B-tree索引\n3. 考虑部分索引(WHERE条件)\n4. 复合索引遵循最左前缀\n5. 定期REINDEX维护",
        "tags": ["database", "sql", "postgresql"],
        "applicable_scenarios": "查询超过100ms、全表扫描、排序慢",
    },
]

# ---------------------------------------------------------------------------
# Collision Engine seed memories (3 domains × 5 topics)
# ---------------------------------------------------------------------------

COLLISION_MEMORIES = {
    "fitness": [
        {"content": "今天开始执行新的跑步计划，目标是每周跑步4次，每次5公里，逐步提升到10公里", "days_ago": 28},
        {"content": "力量训练计划调整：周一胸背、周三腿部、周五肩臂，每组做到力竭", "days_ago": 21},
        {"content": "营养师建议的饮食方案：高蛋白低碳水，每天蛋白质摄入体重×1.6克", "days_ago": 14},
        {"content": "跑步时膝盖有轻微不适，医生建议减少跑量，先做康复训练", "days_ago": 7},
        {"content": "决定报名参加下个月的马拉松比赛，需要制定赛前训练计划", "days_ago": 2},
    ],
    "coding": [
        {"content": "决定学习Rust语言，它的所有权模型很独特，内存安全不需要GC", "days_ago": 25},
        {"content": "开始重构后端项目，从单体架构拆分为微服务，用gRPC通信", "days_ago": 18},
        {"content": "搭建了CI/CD流水线：GitHub Actions + Docker + ArgoCD自动部署", "days_ago": 12},
        {"content": "团队代码审查规范：PR不超过400行，必须有单元测试，至少2人approve", "days_ago": 5},
        {"content": "考虑放弃Rust转学Go，Go的学习曲线更平缓，生态也更成熟", "days_ago": 1},
    ],
    "reading": [
        {"content": "读完《思考快与慢》，系统1直觉快但易错，系统2理性慢但准确", "days_ago": 26},
        {"content": "技术博客阅读：分布式系统中CAP定理的实际应用案例", "days_ago": 19},
        {"content": "读了一本存在主义哲学书，'存在先于本质'这个观点很有启发", "days_ago": 13},
        {"content": "读了乔布斯传记，'Stay hungry, stay foolish'的背后是极致的产品追求", "days_ago": 8},
        {"content": "在读《三体》，黑暗森林法则作为博弈论的宇宙版本很震撼", "days_ago": 3},
    ],
}

COLLISION_SEMANTIC_PAIRS = [
    {
        "id": "semantic_pair_1",
        "a": {"content": "冥想练习让我更加专注，每天早上20分钟正念冥想效果显著", "source": "memora"},
        "b": {"content": "Transformer模型的attention机制是深度学习的核心突破，self-attention让模型关注输入的不同部分", "source": "memora"},
        "expected_bridge": "专注/attention",
    },
    {
        "id": "semantic_pair_2",
        "a": {"content": "间歇性断食16:8方案效果不错，身体学会了更高效地利用能量和清理废物", "source": "memora"},
        "b": {"content": "JVM的GC回收策略优化：G1收集器通过分区回收减少停顿时间，清理不再引用的对象", "source": "memora"},
        "expected_bridge": "清理/回收",
    },
]

COLLISION_EMOTIONAL_PAIR = {
    "a": {"content": "最近工作压力很大，感到非常焦虑，晚上经常失眠", "source": "memora"},
    "b": {"content": "终于找到了解决焦虑的方法：规律运动+写日记+限制社交媒体时间", "source": "memora"},
}

# ---------------------------------------------------------------------------
# E2E recall seed conversations (10 groups)
# ---------------------------------------------------------------------------

E2E_CONVERSATIONS = [
    {
        "id": "docker_flow",
        "content": "我刚刚排查了一个Docker网络问题：容器A ping不通容器B。排查步骤是先docker network ls查看网络，然后docker inspect检查IP分配，发现两个容器不在同一个network里，创建共享网络后解决。",
        "expect_skill": True,
    },
    {
        "id": "code_style_pref",
        "content": "我喜欢简洁的代码风格，函数不超过20行，变量名要有意义，不写多余的注释。Clean Code的理念我很认同。",
        "expect_skill": False,
    },
    {
        "id": "graphql_decision",
        "content": "经过团队讨论，我们决定把项目的API从REST迁移到GraphQL。原因是前端需要灵活查询，REST的over-fetching问题严重。计划分3个月完成迁移。",
        "expect_skill": False,
    },
    {
        "id": "rust_learning",
        "content": "Rust的所有权模型笔记：每个值有且仅有一个owner，owner离开作用域时值被drop。借用分为不可变引用(&T)和可变引用(&mut T)，不能同时存在。",
        "expect_skill": False,
    },
    {
        "id": "exercise_conflict",
        "content": "我本来计划每天锻炼一小时，包括跑步和力量训练。但是最近项目太忙了，连续两周都没有时间运动，感觉身体状态在变差。",
        "expect_skill": False,
    },
    {
        "id": "car_update_old",
        "content": "[2026-03-01] 今天提了新车，是一辆红色的Tesla Model 3，非常兴奋！",
        "expect_skill": False,
    },
    {
        "id": "car_update_new",
        "content": "[2026-04-01] 把Tesla Model 3换成了BMW iX3，因为需要更大的后备箱空间。",
        "expect_skill": False,
    },
    {
        "id": "travel_1",
        "content": "上个月去了京都旅行，参观了金阁寺和伏见稻荷大社，在锦市场吃了很多好吃的。",
        "expect_skill": False,
    },
    {
        "id": "travel_2",
        "content": "春节去了冰岛，看到了极光！还泡了蓝湖温泉，冰川徒步也很刺激。",
        "expect_skill": False,
    },
    {
        "id": "travel_3",
        "content": "国庆假期自驾去了云南，从昆明到大理到丽江，走了滇藏线一小段。",
        "expect_skill": False,
    },
    {
        "id": "pypi_flow",
        "content": "发布Python包到PyPI的完整流程：1.确保setup.py/pyproject.toml配置正确 2.python -m build构建 3.twine check检查 4.twine upload上传到TestPyPI验证 5.twine upload上传到正式PyPI 6.pip install验证安装",
        "expect_skill": True,
    },
    {
        "id": "dark_mode_pref",
        "content": "我强烈偏好深色主题(dark mode)，无论是IDE、终端还是浏览器都用深色。浅色主题(light mode)看久了眼睛会很累。",
        "expect_skill": False,
    },
    {
        "id": "running_inspiration",
        "content": "今天跑步的时候突然想到一个代码重构的灵感：可以用策略模式替换那一堆if-else，这样新增支付方式就不用改原有代码了。",
        "expect_skill": False,
    },
]


# ---------------------------------------------------------------------------
# Test case definitions (100 cases)
# ---------------------------------------------------------------------------

def _all_cases():
    """Return list of 100 test case dicts."""
    cases = []

    # ===== Benchmark A: Skill Registry (35 cases) =====

    # Phase 1: Skill Learning & Registration (SK-01 ~ SK-10)
    cases.append({"id": "SK-01", "cat": "skill", "phase": "learning",
        "desc": "基础Skill注册", "type": "api"})
    cases.append({"id": "SK-02", "cat": "skill", "phase": "learning",
        "desc": "带prompt_template的Skill", "type": "api"})
    cases.append({"id": "SK-03", "cat": "skill", "phase": "learning",
        "desc": "带prerequisites的Skill", "type": "api"})
    cases.append({"id": "SK-04", "cat": "skill", "phase": "learning",
        "desc": "重复内容Skill(不同名字)", "type": "api"})
    cases.append({"id": "SK-05", "cat": "skill", "phase": "learning",
        "desc": "空procedures自动生成template", "type": "api"})
    cases.append({"id": "SK-06", "cat": "skill", "phase": "learning",
        "desc": "多标签Skill", "type": "api"})
    cases.append({"id": "SK-07", "cat": "skill", "phase": "learning",
        "desc": "source_memories关联", "type": "api"})
    cases.append({"id": "SK-08", "cat": "skill", "phase": "learning",
        "desc": "中文内容Skill", "type": "api"})
    cases.append({"id": "SK-09", "cat": "skill", "phase": "learning",
        "desc": "长内容Skill(>2000字)", "type": "api"})
    cases.append({"id": "SK-10", "cat": "skill", "phase": "learning",
        "desc": "批量注册5个Skill", "type": "api"})

    # Phase 2: Skill Matching & Retrieval (SK-11 ~ SK-25)
    cases.append({"id": "SK-11", "cat": "skill", "phase": "matching",
        "desc": "精确语义匹配", "query": "我的Docker容器网络不通怎么办",
        "expect_skill": "Docker网络调试", "type": "recall"})
    cases.append({"id": "SK-12", "cat": "skill", "phase": "matching",
        "desc": "同义词匹配", "query": "容器之间ping不通如何排查",
        "expect_skill": "Docker网络调试", "type": "recall"})
    cases.append({"id": "SK-13", "cat": "skill", "phase": "matching",
        "desc": "跨语言匹配", "query": "How to debug container networking issues",
        "expect_skill": "Docker网络调试", "type": "recall"})
    cases.append({"id": "SK-14", "cat": "skill", "phase": "matching",
        "desc": "关键词回退匹配", "query": "docker network debug",
        "expect_skill": "Docker网络调试", "type": "recall"})
    cases.append({"id": "SK-15", "cat": "skill", "phase": "matching",
        "desc": "多Skill匹配", "query": "我的Python Web应用React前端很慢",
        "expect_multi": True, "type": "recall"})
    cases.append({"id": "SK-16", "cat": "skill", "phase": "matching",
        "desc": "不相关查询不匹配", "query": "今天天气怎么样",
        "expect_empty": True, "type": "recall"})
    cases.append({"id": "SK-17", "cat": "skill", "phase": "matching",
        "desc": "L1注入验证", "query": "Docker容器网络故障排查",
        "check_l1": True, "type": "recall"})
    cases.append({"id": "SK-18", "cat": "skill", "phase": "matching",
        "desc": "merged输出包含Skill", "query": "Docker容器网络故障排查",
        "check_merged": True, "type": "recall"})
    cases.append({"id": "SK-19", "cat": "skill", "phase": "matching",
        "desc": "Skill procedures格式", "query": "怎么排查Docker网络",
        "check_procedures": True, "type": "recall"})
    cases.append({"id": "SK-20", "cat": "skill", "phase": "matching",
        "desc": "DRAFT Skill不被召回", "type": "recall_draft"})
    cases.append({"id": "SK-21", "cat": "skill", "phase": "matching",
        "desc": "DEPRECATED Skill不被召回", "type": "recall_deprecated"})
    cases.append({"id": "SK-22", "cat": "skill", "phase": "matching",
        "desc": "5个Skill上限", "type": "recall_limit"})
    cases.append({"id": "SK-23", "cat": "skill", "phase": "matching",
        "desc": "Skill+Evidence混合", "type": "recall_mixed"})
    cases.append({"id": "SK-24", "cat": "skill", "phase": "matching",
        "desc": "偏好类Skill匹配", "type": "recall_preference"})
    cases.append({"id": "SK-25", "cat": "skill", "phase": "matching",
        "desc": "Skill search API", "type": "search_api"})

    # Phase 3: Skill Lifecycle (SK-26 ~ SK-35)
    cases.append({"id": "SK-26", "cat": "skill", "phase": "lifecycle",
        "desc": "DRAFT→ACTIVE promote", "type": "lifecycle"})
    cases.append({"id": "SK-27", "cat": "skill", "phase": "lifecycle",
        "desc": "promote冷却期检测", "type": "lifecycle"})
    cases.append({"id": "SK-28", "cat": "skill", "phase": "lifecycle",
        "desc": "成功反馈记录", "type": "lifecycle"})
    cases.append({"id": "SK-29", "cat": "skill", "phase": "lifecycle",
        "desc": "失败反馈记录", "type": "lifecycle"})
    cases.append({"id": "SK-30", "cat": "skill", "phase": "lifecycle",
        "desc": "utility_rate计算", "type": "lifecycle"})
    cases.append({"id": "SK-31", "cat": "skill", "phase": "lifecycle",
        "desc": "低utility触发rewrite", "type": "lifecycle"})
    cases.append({"id": "SK-32", "cat": "skill", "phase": "lifecycle",
        "desc": "内容更新+版本递增", "type": "lifecycle"})
    cases.append({"id": "SK-33", "cat": "skill", "phase": "lifecycle",
        "desc": "deprecate后不可promote", "type": "lifecycle"})
    cases.append({"id": "SK-34", "cat": "skill", "phase": "lifecycle",
        "desc": "usage_stats统计", "type": "lifecycle"})
    cases.append({"id": "SK-35", "cat": "skill", "phase": "lifecycle",
        "desc": "stats全局统计", "type": "lifecycle"})

    # ===== Benchmark B: Collision Engine (35 cases) =====

    # Phase 1: Strategy Coverage (CL-01 ~ CL-14)
    for i, strat in enumerate([
        "semantic_bridge", "semantic_bridge",
        "chronos_crossref", "chronos_crossref",
        "digest_bridge", "digest_bridge",
        "dormant_revival", "dormant_revival",
        "temporal_echo", "temporal_echo",
        "contradiction_based", "contradiction_based",
        "blind_spot_based", "blind_spot_based",
    ]):
        cases.append({"id": f"CL-{i+1:02d}", "cat": "collision", "phase": "strategy",
            "desc": f"策略覆盖: {strat}", "strategy": strat, "type": "collision"})

    # Phase 2: Insight Quality (CL-15 ~ CL-25)
    quality_ids = [
        ("CL-15", "novelty低分(不相关对)", "quality_low_novelty"),
        ("CL-16", "novelty高分(深层关联)", "quality_high_novelty"),
        ("CL-17", "connection非空", "quality_connection"),
        ("CL-18", "ideas非空", "quality_ideas"),
        ("CL-19", "emotional_relevance", "quality_emotional"),
        ("CL-20", "自碰撞过滤", "quality_self_filter"),
        ("CL-21", "空pool处理", "quality_empty_pool"),
        ("CL-22", "单条目pool", "quality_single_pool"),
        ("CL-23", "to_markdown格式", "quality_markdown"),
        ("CL-24", "save_insights持久化", "quality_persist"),
        ("CL-25", "collisions_per_round限制", "quality_limit"),
    ]
    for cid, desc, qtype in quality_ids:
        cases.append({"id": cid, "cat": "collision", "phase": "quality",
            "desc": desc, "type": qtype})

    # Phase 3: Cross-Layer Discovery (CL-26 ~ CL-35)
    cross_ids = [
        ("CL-26", "Memora↔Chronos跨层", "cross_chronos"),
        ("CL-27", "Memora↔Digest跨层", "cross_digest"),
        ("CL-28", "Memora↔MSA跨层(deep_collide)", "cross_msa"),
        ("CL-29", "KG矛盾传播", "cross_kg_propagate"),
        ("CL-30", "KG线索发现", "cross_kg_threads"),
        ("CL-31", "多策略单轮碰撞", "cross_multi_strategy"),
        ("CL-32", "attention focus影响", "cross_attention"),
        ("CL-33", "bridge.collide()完整流程", "cross_http_collide"),
        ("CL-34", "deep_collide主题聚焦", "cross_deep_collide"),
        ("CL-35", "碰撞insight写入Memora", "cross_insight_recall"),
    ]
    for cid, desc, ctype in cross_ids:
        cases.append({"id": cid, "cat": "collision", "phase": "cross",
            "desc": desc, "type": ctype})

    # ===== Benchmark C: E2E Augmented Recall (30 cases) =====

    # Phase 1: L1 Skill Injection (E2E-01 ~ E2E-10)
    e2e_l1 = [
        ("E2E-01", "Docker容器之间网络不通", "Docker网络排查", "l1_skill"),
        ("E2E-02", "发布Python包到PyPI", "PyPI发布流程", "l1_skill"),
        ("E2E-03", "帮我配置编辑器", "dark mode", "l1_pref"),
        ("E2E-04", "代码风格怎么规范", "简洁代码", "l1_pref"),
        ("E2E-05", "今天吃什么", None, "l1_empty"),
        ("E2E-06", "Python Web应用React前端性能", None, "l1_multi"),
        ("E2E-07", "Docker网络问题排查方法", None, "l1_mixed"),
        ("E2E-08", "查询触发executable_prompt", None, "l1_exec_prompt"),
        ("E2E-09", "DRAFT Skill不泄漏", None, "l1_no_draft"),
        ("E2E-10", "budget控制Skill上限", None, "l1_budget"),
    ]
    for cid, query, expect, etype in e2e_l1:
        cases.append({"id": cid, "cat": "e2e", "phase": "l1",
            "desc": f"L1 Skill: {cid}", "query": query,
            "expect_skill_name": expect, "type": etype})

    # Phase 2: L2/L4 KG Relations (E2E-11 ~ E2E-20)
    e2e_kg = [
        ("E2E-11", "GraphQL迁移进展如何", "l2_relation"),
        ("E2E-12", "Rust适合我的项目吗", "l2_relation"),
        ("E2E-13", "我的锻炼计划", "l4_contradiction"),
        ("E2E-14", "查询触发contradiction", "l4_contradiction"),
        ("E2E-15", "KG supports关系", "l2_supports"),
        ("E2E-16", "KG contradicts进入L4", "l4_edge_type"),
        ("E2E-17", "inference_engine矛盾扫描", "l4_inference"),
        ("E2E-18", "多KG关系聚合", "l2_multi"),
        ("E2E-19", "KG关系+Evidence混合", "l2_mixed"),
        ("E2E-20", "无KG关系的查询", "l2_empty"),
    ]
    for cid, desc, etype in e2e_kg:
        cases.append({"id": cid, "cat": "e2e", "phase": "kg",
            "desc": desc, "type": etype})

    # Phase 3: Full Pipeline (E2E-21 ~ E2E-30)
    e2e_full = [
        ("E2E-21", "Skill+KG+Evidence三层协同", "full_triple"),
        ("E2E-22", "碰撞insight可被recall", "full_collision_recall"),
        ("E2E-23", "knowledge update识别", "full_knowledge_update"),
        ("E2E-24", "多会话聚合", "full_multi_session"),
        ("E2E-25", "intent分类影响层权重", "full_intent"),
        ("E2E-26", "安全门控PII过滤", "full_security"),
        ("E2E-27", "budget控制总token限制", "full_budget"),
        ("E2E-28", "fallback_merge降级", "full_fallback"),
        ("E2E-29", "Skill feedback闭环", "full_feedback_loop"),
        ("E2E-30", "全管线延迟<3000ms", "full_latency"),
    ]
    for cid, desc, etype in e2e_full:
        cases.append({"id": cid, "cat": "e2e", "phase": "full",
            "desc": desc, "type": etype})

    return cases


ALL_CASES = _all_cases()
assert len(ALL_CASES) == 100, f"Expected 100 cases, got {len(ALL_CASES)}"
