---
name: qlib-alpha
description: 用于在当前 A 股量化仓库中引入 Qlib 风格 alpha 思路，只抽取对日 K 短线选股有用的因子族，优先做低侵入映射、候选因子清单、落地顺序与最小实验设计。用户要增厚主策略 alpha、把 Qlib Alpha158 或 Alpha360 思路映射到现有因子系统时使用。
---

# qlib-alpha

只做一件事：把 Qlib 的高价值 alpha 思路，压缩成适合当前仓库的小步落地方案。

## 什么时候用

- 用户要补 alpha，不想重构框架
- 用户要从 Qlib 借思路，不想整包接入
- 用户要给主策略增加比动量更厚的解释层

## 先看什么

1. 当前主策略配置：`configs/research_production_default.json`
2. 当前主流程：`src/ashare_quant/pipeline.py`
3. 现有技术因子：`src/ashare_quant/factors/technical.py`
4. 参考清单：`references/factor_shortlist.md`

## 工作规则

1. 每次最多引入 1 到 3 个新因子
2. 新因子先做 prototype，不直接塞进生产主配置
3. 先走 factor-check，再考虑并入主分
4. 不整包搬 Qlib，不引入庞大依赖，不改主框架

## 推荐流程

1. 先确认用户想补的是哪类 alpha：
   - 资金确认
   - 行业扩散
   - 拥挤度/筹码
   - 波动压缩/突破
2. 从 `references/factor_shortlist.md` 里选最多 3 个候选
3. 先在独立脚本或新模块里实现
4. 对因子做：
   - 截面标准化
   - 行业/市值中性化
5. 交给 `factor-check` skill 做验证
6. 只有通过后，才改研究配置

## 实现落点

- 新因子优先落在：
  - `src/ashare_quant/factors/technical.py`
  - 或新建 `src/ashare_quant/factors/qlib_prototypes.py`
- 不要直接改回测、风控、执行

## 输出要求

必须给出表格：

| 因子 | 经济含义 | 和现有动量是否重复 | 实现成本 | 是否建议试验 |
| --- | --- | --- | --- | --- |

然后再给：

| 下一步 | 文件 | 动作 |
| --- | --- | --- |

