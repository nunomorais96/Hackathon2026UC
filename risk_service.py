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


def calculate_risk_score(row):
    volatility_score = normalize(row.get("volatility"), 0.15, 0.70)
    pe_score = normalize(row.get("pe_ratio"), 15, 80)
    debt_score = normalize(row.get("debt_to_equity"), 20, 200)

    raw_score = (
        volatility_score * 0.45 +
        pe_score * 0.35 +
        debt_score * 0.20
    )

    return round(raw_score * 10, 1)


def classify_risk(score):
    if score < 4:
        return "Low"
    if score < 7:
        return "Medium"
    return "High"


def explain_risk(row, score):
    reasons = []

    if row.get("volatility") and row.get("volatility") > 0.4:
        reasons.append("high historical volatility")

    if row.get("pe_ratio") and row.get("pe_ratio") > 40:
        reasons.append("high valuation based on P/E ratio")

    if row.get("debt_to_equity") and row.get("debt_to_equity") > 100:
        reasons.append("high debt-to-equity ratio")

    if not reasons:
        reasons.append("risk appears moderate based on available public metrics")

    return f"Risk score {score}/10 due to " + ", ".join(reasons) + "."


def add_risk_analysis(df):
    df = df.copy()

    scores = []
    labels = []
    explanations = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        score = calculate_risk_score(row_dict)
        label = classify_risk(score)
        explanation = explain_risk(row_dict, score)

        scores.append(score)
        labels.append(label)
        explanations.append(explanation)

    df["risk_score"] = scores
    df["risk_level"] = labels
    df["risk_explanation"] = explanations

    return df