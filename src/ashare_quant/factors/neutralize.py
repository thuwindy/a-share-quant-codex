from __future__ import annotations

import pandas as pd

from ashare_quant.factors.factor_preprocess import FactorPreprocessConfig, preprocess_factor_cross_section


def neutralize_by_size_and_industry(
    df: pd.DataFrame,
    factor_cols: list[str],
    preprocess_config: FactorPreprocessConfig | None = None,
) -> pd.DataFrame:
    """Winsorize, residualize, z-score, then rank factors by date.

    Output columns:
    - `{factor}_resid`: residual after size + industry neutralization
    - `{factor}_z`: z-score of the residual on the same date
    - `{factor}_neu`: centered rank percentile in [-0.5, 0.5], used by default downstream
    """

    out = df.copy()
    cfg = preprocess_config or FactorPreprocessConfig()

    for date, idx in out.groupby("date").groups.items():
        sl = out.loc[idx]
        for factor in factor_cols:
            residual, zscore, ranked = preprocess_factor_cross_section(
                sl[factor],
                sl=sl,
                config=cfg,
            )
            out.loc[idx, f"{factor}_resid"] = residual
            out.loc[idx, f"{factor}_z"] = zscore
            out.loc[idx, f"{factor}_neu"] = ranked
    return out
