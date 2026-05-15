def normalize(value, low, high):
    if value is None:
        return 0.5

    try:
        value = float(value)
    except Exception:
        return 0.5

    if value <= low:
        return 0
    if value >= high:
        return 1

    return (value - low) / (high - low)


def get_profile_weights(profile):
    if profile == "Conservative":
        return {
            "volatility": 0.55,
            "pe": 0.25,
            "debt": 0.20
        }

    if profile == "Aggressive":
        return {
            "volatility": 0.30,
            "pe": 0.50,
            "debt": 0.20
        }

    if profile == "Moderate-Aggressive":
        return {
            "volatility": 0.35,
            "pe": 0.45,
            "debt": 0.20
        }

    return {
        "volatility": 0.45,
        "pe": 0.35,
        "debt": 0.20
    }


def get_horizon_adjustment(horizon):
    if horizon == "1 year":
        return {
            "volatility_multiplier": 1.25,
            "pe_multiplier": 1.10,
            "debt_multiplier": 1.00
        }

    if horizon == "3 years":
        return {
            "volatility_multiplier": 1.10,
            "pe_multiplier": 1.00,
            "debt_multiplier": 1.00
        }

    if horizon == "5 years":
        return {
            "volatility_multiplier": 0.95,
            "pe_multiplier": 0.95,
            "debt_multiplier": 1.00
        }

    if horizon == "10 years":
        return {
            "volatility_multiplier": 0.85,
            "pe_multiplier": 0.90,
            "debt_multiplier": 1.00
        }

    return {
        "volatility_multiplier": 1.00,
        "pe_multiplier": 1.00,
        "debt_multiplier": 1.00
    }


def calculate_risk_score(row, profile="Moderate", horizon="5 years"):
    volatility_score = normalize(row.get("volatility"), 0.15, 0.70)
    pe_score = normalize(row.get("pe_ratio"), 15, 80)
    debt_score = normalize(row.get("debt_to_equity"), 20, 200)

    weights = get_profile_weights(profile)
    horizon_adjustment = get_horizon_adjustment(horizon)

    adjusted_volatility = volatility_score * horizon_adjustment["volatility_multiplier"]
    adjusted_pe = pe_score * horizon_adjustment["pe_multiplier"]
    adjusted_debt = debt_score * horizon_adjustment["debt_multiplier"]

    raw_score = (
        adjusted_volatility * weights["volatility"] +
        adjusted_pe * weights["pe"] +
        adjusted_debt * weights["debt"]
    )

    risk_score = min(raw_score * 10, 10)

    return round(risk_score, 1)


def classify_risk(score):
    if score < 4:
        return "Low"
    if score < 7:
        return "Medium"
    return "High"


def explain_risk(row, score, profile="Moderate", horizon="5 years"):
    reasons = []

    if row.get("volatility") and row.get("volatility") > 0.4:
        if horizon in ["1 year", "3 years"]:
            reasons.append("high volatility is more relevant for shorter investment horizons")
        else:
            reasons.append("high historical volatility")

    if row.get("pe_ratio") and row.get("pe_ratio") > 40:
        if profile in ["Conservative", "Moderate"]:
            reasons.append("high valuation may be less suitable for this risk profile")
        else:
            reasons.append("high valuation reflects strong growth expectations")

    if row.get("debt_to_equity") and row.get("debt_to_equity") > 100:
        reasons.append("high debt-to-equity ratio")

    if horizon == "1 year":
        reasons.append("short investment horizon increases sensitivity to market movements")

    if horizon == "10 years":
        reasons.append("longer horizon reduces short-term volatility impact but does not remove business risk")

    if not reasons:
        reasons.append("risk appears moderate based on available public metrics")

    return f"Risk score {score}/10 due to " + ", ".join(reasons) + "."


def add_risk_analysis(df, profile="Moderate", horizon="5 years"):
    df = df.copy()

    scores = []
    labels = []
    explanations = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()

        score = calculate_risk_score(
            row_dict,
            profile=profile,
            horizon=horizon
        )

        label = classify_risk(score)

        explanation = explain_risk(
            row_dict,
            score,
            profile=profile,
            horizon=horizon
        )

        scores.append(score)
        labels.append(label)
        explanations.append(explanation)

    df["risk_score"] = scores
    df["risk_level"] = labels
    df["risk_explanation"] = explanations

    return df