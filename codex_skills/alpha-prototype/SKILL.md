---
name: alpha-prototype
description: 用于从 ML4Trading 和 Alpha101 里抽取少量高价值 alpha 原型，在当前 A 股量化仓库中做低侵入试验。用户要找新的 alpha 原型但不想整包接入外部库时使用。
---

# alpha-prototype

只做一件事：从外部 alpha 思路里，挑少量能打的原型，快速做成可实验因子。

## 什么时候用

- 用户要“找几个像样的 alpha”
- 用户不想整包搬 Alpha101
- 用户要给主策略补新思路

## 先看什么

1. `references/prototype_shortlist.md`
2. 当前特征实现：
   - `src/ashare_quant/factors/technical.py`
3. 当前主策略配置：
   - `configs/research_production_default.json`

## 工作规则

1. 每次最多试 3 个 prototype
2. 先做独立列，不要直接替换旧因子
3. 先 factor-check，再决定是否进主分
4. 不抄整套 Alpha101 公式库

## 推荐流程

1. 从短名单选 1 到 3 个原型
2. 用现有 OHLCV 列实现
3. 做标准化和中性化
4. 走 `factor-check`
5. 通过后再做融合实验

## 输出要求

| 原型 | 来源 | 经济含义 | 是否适合 A 股日K短线 | 实现难度 |
| --- | --- | --- | --- | --- |

然后给：

| 文件 | 动作 |
| --- | --- |

