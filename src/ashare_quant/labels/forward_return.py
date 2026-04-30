from __future__ import annotations

import pandas as pd

from ashare_quant.labels.label_builder import ExecutionSpec, add_label_columns


def add_forward_return_label(
    df: pd.DataFrame,
    horizon: int = 5,
    signal_time: str = "close",
    execution_price: str = "close",
    execution_lag: int = 1,
) -> pd.DataFrame:
    """Backward-compatible wrapper around the new label builder.

    Default alignment uses scheme A:
    - signal generated on t close
    - execution on t+1 close
    - hold for `horizon` trading days after execution
    - raw label = close[t+1+horizon] / close[t+1] - 1
    """

    spec = ExecutionSpec(
        signal_time=signal_time,
        execution_price=execution_price,
        execution_lag=execution_lag,
        holding_window=horizon,
    )
    return add_label_columns(df, horizons=[horizon], spec=spec)
