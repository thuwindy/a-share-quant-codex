# 小切片快实验矩阵

| 场景 | 推荐脚本 | 说明 |
| --- | --- | --- |
| 因子增删 | `scripts/run_ablation_suite.py` | 看主策略收益和稳定性 |
| topN/持有期微调 | `scripts/run_quick_topn_notrade_grid.py` | 快速扫描参数 |
| 候选池精排 | `scripts/run_candidate_rerank_validation.py` | 看前5/前10真实改善 |
| ML 可行性 | `scripts/train_ml_upprob_model.py` | 先看排序能力，不看总 accuracy |

## 默认顺序

1. 基线
2. 只改一个变量
3. 看 top bucket
4. 看系统级净增量
5. 再决定是否上全量

