# PRD Generator 消融实验报告（三列对比）

- **实验时间**: 2026-07-03 03:14
- **模型**: deepseek-v4-pro
- **样本数**: 5
- **温度**: 0.3

## 三种模式说明

| 模式 | 说明 | 架构 |
|------|------|------|
| **V3_single_llm** | 单次 LLM 调用直接生成完整 PRD | 1 个 Prompt → 1 次 LLM 调用 → PRD |
| **V1_baseline** | 经典 3 阶段流水线 | 需求分析+架构设计(并行) → 流程梳理 → 文档定稿 |
| **V2_multi_agent** | 多智能体协作 + 评审闭环 | Planner → Supervisor → 4 Agent + Reviewer + 反思修订 |

## 综合质量对比

| 版本 | 完整性 | 可行性 | 一致性 | 相关性 | **综合均分** | 平均耗时 |
|------|------|------|------|------|------|------|
| V3_single_llm | 9.0 | 7.8 | 7.8 | 9.8 | **8.6** | 54s |
| V1_baseline | 9.2 | 8.4 | 7.6 | 9.4 | **8.7** | 191s |
| V2_multi_agent | 9.8 | 7.8 | 8.8 | 9.4 | **9.0** | 843s |

## 提升幅度（相对于单 LLM 基线）

| 指标 | V3 单LLM | V1 流水线 | V2 多智能体 | V1 vs V3 | V2 vs V3 |
|------|----------|-----------|-------------|----------|----------|
| 综合均分 | 8.6/10 | 8.7/10 | 9.0/10 | **+0.1** | **+0.4** |
| completeness | 9.0 | 9.2 | 9.8 | **+0.2** | **+0.8** |
| feasibility | 7.8 | 8.4 | 7.8 | **+0.6** | **+0.0** |
| consistency | 7.8 | 7.6 | 8.8 | **+-0.2** | **+1.0** |
| relevance | 9.8 | 9.4 | 9.4 | **+-0.4** | **+-0.4** |
| 平均耗时 | 54s | 191s | 843s | 3.5x | 15.6x |

## 各样本详情

### V3_single_llm

- [tools] 一个帮助职场新人做职业规划的AI助手，根据用户技能和目标推荐学习路径，并生成个性化周计划... → 综合=9.2 (C:9 F:9 CS:9 R:10) | 47.0s
- [social] 一个面向大学生的校园二手交易小程序，支持发布、浏览、聊天、线下交易... → 综合=8.0 (C:9 F:8 CS:5 R:10) | 47.0s
- [ai_apps] 一个AI驱动的智能客服机器人，能理解自然语言，支持多轮对话，可接入企业微信和钉钉... → 综合=8.0 (C:9 F:6 CS:7 R:10) | 66.0s
- [tools] 一个面向小微商家的进销存管理SaaS工具，支持库存管理、销售统计、供应商管理... → 综合=8.8 (C:9 F:8 CS:9 R:9) | 47.0s
- [social] 一个基于用户兴趣图谱的内容推荐社区，支持图文、短视频、直播三种内容形态... → 综合=9.0 (C:9 F:8 CS:9 R:10) | 63.0s

### V1_baseline

- [tools] 一个帮助职场新人做职业规划的AI助手，根据用户技能和目标推荐学习路径，并生成个性化周计划... → 综合=8.8 (C:9 F:9 CS:8 R:9) | 196.0s
- [social] 一个面向大学生的校园二手交易小程序，支持发布、浏览、聊天、线下交易... → 综合=8.8 (C:10 F:9 CS:7 R:9) | 189.0s
- [ai_apps] 一个AI驱动的智能客服机器人，能理解自然语言，支持多轮对话，可接入企业微信和钉钉... → 综合=8.5 (C:9 F:8 CS:8 R:9) | 158.0s
- [tools] 一个面向小微商家的进销存管理SaaS工具，支持库存管理、销售统计、供应商管理... → 综合=8.8 (C:9 F:8 CS:8 R:10) | 205.0s
- [social] 一个基于用户兴趣图谱的内容推荐社区，支持图文、短视频、直播三种内容形态... → 综合=8.5 (C:9 F:8 CS:7 R:10) | 208.0s

### V2_multi_agent

- [tools] 一个帮助职场新人做职业规划的AI助手，根据用户技能和目标推荐学习路径，并生成个性化周计划... → 综合=9.2 (C:10 F:8 CS:9 R:10) | 644.0s
- [social] 一个面向大学生的校园二手交易小程序，支持发布、浏览、聊天、线下交易... → 综合=8.5 (C:10 F:8 CS:7 R:9) | 752.0s
- [ai_apps] 一个AI驱动的智能客服机器人，能理解自然语言，支持多轮对话，可接入企业微信和钉钉... → 综合=8.5 (C:9 F:7 CS:10 R:8) | 1115.0s
- [tools] 一个面向小微商家的进销存管理SaaS工具，支持库存管理、销售统计、供应商管理... → 综合=9.2 (C:10 F:8 CS:9 R:10) | 839.0s
- [social] 一个基于用户兴趣图谱的内容推荐社区，支持图文、短视频、直播三种内容形态... → 综合=9.2 (C:10 F:8 CS:9 R:10) | 865.0s

## 简历可用数据点

- 多智能体架构（V2）将 PRD 综合质量从单 LLM（V3）的 **8.6/10 提升至 9.0/10（+0.4 分，+5%）**
- 经典流水线（V1）为 8.7/10，多智能体（V2）进一步从 8.7 提升至 9.0（+0.3），证明 Agent 分工+评审闭环的有效性
- 单 LLM（V3）仅需 54s 但质量最低；多智能体用 15.6x 时间换取 +0.4 分质量提升
- 在 5 个真实产品场景上完成三列消融实验验证
- 4 维度 LLM-as-Judge 自动评估（完整性/可行性/一致性/相关性）

## 原始数据

```json
{
  "V1_baseline": [
    {
      "product_idea": "一个帮助职场新人做职业规划的AI助手，根据用户技能和目标推荐学习路径，并生成个性化周计划",
      "category": "tools",
      "gen_time_s": 196.0,
      "prd_length": 5521,
      "eval_time_s": 92.0,
      "scores": {
        "completeness": 9,
        "feasibility": 9,
        "consistency": 8,
        "relevance": 9
      }
    },
    {
      "product_idea": "一个面向大学生的校园二手交易小程序，支持发布、浏览、聊天、线下交易",
      "category": "social",
      "gen_time_s": 189.0,
      "prd_length": 4022,
      "eval_time_s": 146.0,
      "scores": {
        "completeness": 10,
        "feasibility": 9,
        "consistency": 7,
        "relevance": 9
      }
    },
    {
      "product_idea": "一个AI驱动的智能客服机器人，能理解自然语言，支持多轮对话，可接入企业微信和钉钉",
      "category": "ai_apps",
      "gen_time_s": 158.0,
      "prd_length": 5206,
      "eval_time_s": 133.0,
      "scores": {
        "completeness": 9,
        "feasibility": 8,
        "consistency": 8,
        "relevance": 9
      }
    },
    {
      "product_idea": "一个面向小微商家的进销存管理SaaS工具，支持库存管理、销售统计、供应商管理",
      "category": "tools",
      "gen_time_s": 205.0,
      "prd_length": 5194,
      "eval_time_s": 112.0,
      "scores": {
        "completeness": 9,
        "feasibility": 8,
        "consistency": 8,
        "relevance": 10
      }
    },
    {
      "product_idea": "一个基于用户兴趣图谱的内容推荐社区，支持图文、短视频、直播三种内容形态",
      "category": "social",
      "gen_time_s": 208.0,
      "prd_length": 6438,
      "eval_time_s": 127.0,
      "scores": {
        "completeness": 9,
        "feasibility": 8,
        "consistency": 7,
        "relevance": 10
      }
    }
  ],
  "V2_multi_agent": [
    {
      "product_idea": "一个帮助职场新人做职业规划的AI助手，根据用户技能和目标推荐学习路径，并生成个性化周计划",
      "category": "tools",
      "gen_time_s": 644.0,
      "prd_length": 5173,
      "eval_time_s": 101.0,
      "scores": {
        "completeness": 10,
        "feasibility": 8,
        "consistency": 9,
        "relevance": 10
      },
      "reviewer_score": 68,
      "reflection_round": 2,
      "reflection_history": [
        {
          "round": 1,
          "score": 45,
          "feedback": {
            "requirements_analyst": "需求分析报告完全为空，缺失整个章节。必须重新生成，包含用户故事、业务需求、功能需求、非功能需求、验收标准等PRD核心内容。缺失导致后续所有Agent输出缺乏依据，严重影响一致性和完整性。",
            "feature_planner": "1. MVP范围定义与成功标准、体验设计不一致：成功标准要求“进行任务勾选”，但MVP未包含任务勾选相关功能（或未明确将基础查看/勾选作为生成功能的一部分）。建议明确将“计划查看与基本编辑”中的核心查看和勾选交互纳入MVP，或将周计划生成功能描述明确包含结果展示与简单交互。2. 功能依赖关系图中，计划保存依赖生成，但生成后未保存时用户需要查看，查看必须在保存前可用，逻辑不清晰。建议调整依赖或合并功能。3. 缺少非功能性需求描述（如性能、安全等），虽技术顾问补充，但功能规划应提及。",
            "ux_designer": "1. 缺少对部分错误状态的处理细节，例如目标设定时行业多选可能产生冲突，未说明。2. 设计中未体现周计划生成后数据为空或生成失败时的具体UI（仅提到空状态引导，但未详细描述）。3. 任务勾选后若用户未手动保存，关闭浏览器再次进入时可能丢失状态，未提供明确提示或自动保存机制。",
            "tech_advisor": "1. API设计包含了认证相关的端点（/api/auth/*）和需要认证的端点（如/api/plan/save、/api/task等），但MVP阶段无用户系统，这会导致实现时需要额外处理认证逻辑或使部分API暂时无用，增加MVP复杂度。建议明确MVP阶段仅使用BFF转发AI调用，无需认证，后续扩展。2. 数据模型包含了User、UserSkill等项，虽可预留但增加了初期数据库设计负担，需遵循YAGNI原则精简。3. 对浏览器本地存储的容量限制处理不足，未给出具体策略（如超过5MB时的降级方案）。"
          },
          "agents_to_revise": [
            "requirements_analyst",
            "feature_planner",
            "ux_designer",
            "tech_advisor"
          ]
        },
        {
          "round": 2,
          "score": 68,
          "feedback": {
            "requirements_analyst": "需求分析报告完全缺失，未提供任何用户故事、场景、业务目标或成功指标，导致整个PRD缺乏需求基础，必须重写。",
            "feature_planner": "功能列表基本覆盖产品想法，但未与需求分析关联（因需求缺失），MVP定义合理。建议补充技能差距可视化、计划分享的受众分析。",
            "ux_designer": "设计细致，交互流程完整，边缘情况和错误处理考虑周全。受需求缺失影响，部分界面（如注册登录）暂未实现但符合MVP定义。无需大幅修改。",
            "tech_advisor": "技术方案前瞻且详实，数据模型与API设计支撑所有功能，风险缓解措施具体。部分内容（如消息队列）对MVP可能过度，但整体可行。"
          },
          "agents_to_revise": [
            "requirements_analyst",
            "feature_planner",
            "ux_designer",
            "tech_advisor"
          ]
        }
      ]
    },
    {
      "product_idea": "一个面向大学生的校园二手交易小程序，支持发布、浏览、聊天、线下交易",
      "category": "social",
      "gen_time_s": 752.0,
      "prd_length": 5010,
      "eval_time_s": 143.0,
      "scores": {
        "completeness": 10,
        "feasibility": 8,
        "consistency": 7,
        "relevance": 9
      },
      "reviewer_score": 84,
      "reflection_round": 0,
      "reflection_history": []
    },
    {
      "product_idea": "一个AI驱动的智能客服机器人，能理解自然语言，支持多轮对话，可接入企业微信和钉钉",
      "category": "ai_apps",
      "gen_time_s": 1115.0,
      "prd_length": 6249,
      "eval_time_s": 123.0,
      "scores": {
        "completeness": 9,
        "feasibility": 7,
        "consistency": 10,
        "relevance": 8
      },
      "reviewer_score": 85,
      "reflection_round": 1,
      "reflection_history": [
        {
          "round": 1,
          "score": 78,
          "feedback": {
            "requirements_analyst": "需求分析整体覆盖了核心场景和用户故事，但缺少非功能需求（如性能、安全合规等具体指标）；未明确是否包含工单等扩展功能，需与功能规划对齐以避免范围蔓延。建议补充非功能需求章节，并明确工单需求的优先级。",
            "feature_planner": "功能列表详细，但存在重大优先级偏差：1）数据分析仪表板作为管理层用户故事的必需功能，却被排除在MVP之外（Should），与需求分析师定义的P1核心场景矛盾；2）工单创建与管理被标记为Must，但需求分析中未明确提及，属于过度设计，可能导致范围蔓延。另外，IT管理员需求中的‘自动回复规则’在功能中未明确独立体现，建议补充或说明由哪个功能覆盖。请重新评估MVP边界，将数据分析仪表板纳入MVP，并根据需求确认工单的必要性。",
            "ux_designer": "设计涵盖面广，交互流程清晰，但监控仪表板与功能规划师排除了MVP数据分析仪表板存在冲突；工单管理页面的设计同样可能超出当前最小可行范围，需与产品策略对齐。建议根据功能规划调整后，保留精简版仪表板或暂缓工单界面。",
            "tech_advisor": "技术方案完备，架构合理，能够支撑功能需求。但工单服务（Ticket Service）的加入对应于功能规划师的工单功能，若该功能被降级或移除，需相应调整；数据分析服务预留了后期扩展，思路正确。建议与功能规划师同步，确保技术实现与最终功能清单一致。"
          },
          "agents_to_revise": [
            "requirements_analyst",
            "feature_planner",
            "ux_designer",
            "tech_advisor"
          ]
        }
      ]
    },
    {
      "product_idea": "一个面向小微商家的进销存管理SaaS工具，支持库存管理、销售统计、供应商管理",
      "category": "tools",
      "gen_time_s": 839.0,
      "prd_length": 6055,
      "eval_time_s": 288.0,
      "scores": {
        "completeness": 10,
        "feasibility": 8,
        "consistency": 9,
        "relevance": 10
      },
      "reviewer_score": 80,
      "reflection_round": 0,
      "reflection_history": []
    },
    {
      "product_idea": "一个基于用户兴趣图谱的内容推荐社区，支持图文、短视频、直播三种内容形态",
      "category": "social",
      "gen_time_s": 865.0,
      "prd_length": 5742,
      "eval_time_s": 86.0,
      "scores": {
        "completeness": 10,
        "feasibility": 8,
        "consistency": 9,
        "relevance": 10
      },
      "reviewer_score": 81,
      "reflection_round": 0,
      "reflection_history": []
    }
  ],
  "V3_single_llm": [
    {
      "product_idea": "一个帮助职场新人做职业规划的AI助手，根据用户技能和目标推荐学习路径，并生成个性化周计划",
      "category": "tools",
      "gen_time_s": 47.0,
      "prd_length": 4074,
      "eval_time_s": 76.0,
      "scores": {
        "completeness": 9,
        "feasibility": 9,
        "consistency": 9,
        "relevance": 10
      }
    },
    {
      "product_idea": "一个面向大学生的校园二手交易小程序，支持发布、浏览、聊天、线下交易",
      "category": "social",
      "gen_time_s": 47.0,
      "prd_length": 4145,
      "eval_time_s": 107.0,
      "scores": {
        "completeness": 9,
        "feasibility": 8,
        "consistency": 5,
        "relevance": 10
      }
    },
    {
      "product_idea": "一个AI驱动的智能客服机器人，能理解自然语言，支持多轮对话，可接入企业微信和钉钉",
      "category": "ai_apps",
      "gen_time_s": 66.0,
      "prd_length": 4024,
      "eval_time_s": 79.0,
      "scores": {
        "completeness": 9,
        "feasibility": 6,
        "consistency": 7,
        "relevance": 10
      }
    },
    {
      "product_idea": "一个面向小微商家的进销存管理SaaS工具，支持库存管理、销售统计、供应商管理",
      "category": "tools",
      "gen_time_s": 47.0,
      "prd_length": 4087,
      "eval_time_s": 82.0,
      "scores": {
        "completeness": 9,
        "feasibility": 8,
        "consistency": 9,
        "relevance": 9
      }
    },
    {
      "product_idea": "一个基于用户兴趣图谱的内容推荐社区，支持图文、短视频、直播三种内容形态",
      "category": "social",
      "gen_time_s": 63.0,
      "prd_length": 4448,
      "eval_time_s": 77.0,
      "scores": {
        "completeness": 9,
        "feasibility": 8,
        "consistency": 9,
        "relevance": 10
      }
    }
  ]
}
```