import pandas as pd


PROFILE_WEIGHTS = {
    "Conservative": {
        "risk_score":     -0.50,
        "revenue_growth":  0.15,
        "profit_margin":   0.20,
        "market_cap":      0.15,
    },
    "Moderate": {
        "risk_score":     -0.30,
        "revenue_growth":  0.30,
        "profit_margin":   0.20,
        "market_cap":      0.20,
    },
    "Moderate-Aggressive": {
        "risk_score":     -0.20,
        "revenue_growth":  0.45,
        "profit_margin":   0.20,
        "market_cap":      0.15,
    },
    "Aggressive": {
        "risk_score":     -0.10,
        "revenue_growth":  0.60,
        "profit_margin":   0.20,
        "market_cap":      0.10,
    },
}


HORIZON_RISK_TOLERANCE = {
    "1 year":  -0.10,
    "3 years":  0.00,
    "5 years":  0.05,
    "10 years": 0.10,
}


def _min_max(series):
    series = pd.to_numeric(series, errors="coerce")

    min_v = series.min()
    max_v = series.max()

    if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
        return series.fillna(0.5) * 0 + 0.5

    return ((series - min_v) / (max_v - min_v)).fillna(0.5).clip(0, 1)


def calculate_fit_scores(df, profile, horizon):
    df = df.copy()

    weights = PROFILE_WEIGHTS.get(profile, PROFILE_WEIGHTS["Moderate"])
    horizon_adj = HORIZON_RISK_TOLERANCE.get(horizon, 0.0)

    risk_weight = weights["risk_score"] + horizon_adj

    norm_risk = _min_max(df.get("risk_score"))
    norm_growth = _min_max(df.get("revenue_growth"))
    norm_margin = _min_max(df.get("profit_margin"))
    norm_cap = _min_max(df.get("market_cap"))

    raw = (
        norm_risk * risk_weight +
        norm_growth * weights["revenue_growth"] +
        norm_margin * weights["profit_margin"] +
        norm_cap * weights["market_cap"]
    )

    raw_min = raw.min()
    raw_max = raw.max()

    if raw_max == raw_min:
        df["fit_score"] = 50.0
    else:
        df["fit_score"] = ((raw - raw_min) / (raw_max - raw_min) * 100).round(1)

    df["fit_explanation"] = df.apply(_summarize_metrics, axis=1)

    return df.sort_values("fit_score", ascending=False).reset_index(drop=True)


def _summarize_metrics(row):
    parts = []

    risk_score = row.get("risk_score")
    risk_level = row.get("risk_level")
    if pd.notna(risk_score):
        parts.append(f"Risk {risk_score}/10 ({risk_level})")

    growth = row.get("revenue_growth")
    if pd.notna(growth):
        parts.append(f"Revenue growth {growth * 100:.1f}%")

    margin = row.get("profit_margin")
    if pd.notna(margin):
        parts.append(f"Profit margin {margin * 100:.1f}%")

    cap = row.get("market_cap")
    if pd.notna(cap):
        parts.append(f"Market cap ${cap / 1e9:.1f}B")

    return " · ".join(parts) if parts else "Insufficient public data."


def get_weight_table(profile, horizon):
    weights = PROFILE_WEIGHTS.get(profile, PROFILE_WEIGHTS["Moderate"]).copy()
    weights["risk_score"] = weights["risk_score"] + HORIZON_RISK_TOLERANCE.get(horizon, 0.0)

    return pd.DataFrame([
        {"metric": "Risk score (lower is better)", "weight": round(weights["risk_score"], 2)},
        {"metric": "Revenue growth",               "weight": round(weights["revenue_growth"], 2)},
        {"metric": "Profit margin",                "weight": round(weights["profit_margin"], 2)},
        {"metric": "Market cap (stability)",       "weight": round(weights["market_cap"], 2)},
    ])
