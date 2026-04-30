---
name: fast-ablation
description: 用于在当前量化仓库中做小切片快实验，参考 vectorbt 的快迭代思路，但优先复用现有脚本和数据切片。用户要快速比较基线与一个变体、先拿真实输出再决定是否上全量时使用。
---

# fast-ablation

只做一件事：小切片、快实验、快出真结果。

## 什么时候用

- 用户要先验证一个新因子/新标签/新权重
- 用户不想一上来跑全量主库
- 用户要对比基线和一个变体

## 先看什么

1. `references/experiment_matrix.md`
2. 现成脚本：
   - `scripts/run_ablation_suite.py`
   - `scripts/run_quick_topn_notrade_grid.py`
   - `scripts/run_candidate_rerank_validation.py`

## 工作规则

1. 一次只改一个变量
2. 先锁数据切片
3. 先跑基线，再跑变体
4. 不要一次堆多个改动
5. 结果先看排序和净增量，不先看炫目的 accuracy

## 推荐默认切片

- `recent_slice_20240101_20260330_premium_dynamic.csv`

## 推荐输出

| 方案 | 数据区间 | 改动点 | top5 | top10 | 年化 | Sharpe | 回撤 | 结论 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |

## 什么时候结束

- 能证明有增量，再上正式验证
- 没增量就停，不继续烧算力

