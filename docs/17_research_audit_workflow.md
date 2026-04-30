# Research First, Audit Before Upgrade

## Flow

1. 新因子先进入 `configs/research_factor_family_*.json` 或独立研究配置。
2. 用 `analysis/run_factor_family_research.py` 做单因子研究：
   - IC / IR
   - 分箱收益
   - 单调性
   - 缺失率
   - 和主策略因子的相关性
3. 新规则策略先在 `analysis/vectorized_strategy_sandbox.py` 里跑通。
4. 所有原型都必须先过 `analysis/strategy_audit.py` 的检查：
   - future leakage
   - signal alignment
   - cost model
   - missing trading days
   - limit edge cases
5. 只有研究结果稳定、审计通过，才允许把因子或规则接进 `pipeline.py` 的研究配置。
6. 讨论是否进入生产配置之前，必须先完成独立研究期和稳定性复核。
7. 观察层标签如果要升级成执行管理规则，必须先做标签组合统计研究，证明“高分 + 标签组合”存在稳定持有路径优势。
8. 所有研究输出先看候选池质量：
   - top20 / top10 / top5 的收益兑现
   - 命中率 / 胜率 / 盈亏比
   - 回撤和无效换手有没有恶化
9. 如果新信息只改善解释、不改善候选池质量，默认停在观察层，不进入主分。

## Rules

- 系统分三层：
  - 主分层：只放已经证明有系统级净增量的核心因子
  - 观察层：只放有解释价值、但没有系统级净增量的标签
  - 执行管理层：只接收跨时间稳定、且动作含义清楚的规则
- 低频基本面因子默认 `low_freq_experimental`
- `low_freq_experimental` 因子默认不得进入 production 配置，除非显式设置 `allow_low_freq_experimental_in_production=true`
- 日 K 收盘信号默认明确说明是否 `T+1`
- 研究配置、观察标签、候选支线可以新增
- 正式主策略默认配置不得直接被新 prototype 覆盖
- `research_audit_required=true` 但没有 audit 结果时，研究输出必须显式标记为 `audit_status=missing`
- 因子研究输出必须给出 `research_status`：
  - `keep`：值得进入下一阶段研究或组合实验
  - `observe`：继续观察
  - `reject`：当前阶段不建议继续投入
- 每个研究结果都必须给出 `role_assignment`：
  - `production_core_factor`
  - `research_factor`
  - `observation_label`
  - `execution_rule_candidate`
  - `archived_reject`
- 观察层标签默认只服务于阅读、解释、复盘
- 只有标签组合研究出现稳定模式，才允许讨论把观察层信息升级成执行管理规则
- 当前系统默认处于 `stable_observation_cycle`
- 在冻结期里：
  - 主分层冻结
  - 执行管理层冻结
  - 观察层标签冻结
  - 只允许 `bugfix / monitoring / quality_dashboard / log_report_structure`
- 以后任何新增研究都必须显式给出：
  - `new_research_gate_passed`
  - `why_not`
- 如果不能改善候选池质量，或者不能让动作更清楚，就不开题
