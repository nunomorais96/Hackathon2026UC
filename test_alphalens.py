import math

import pandas as pd
import pytest

from finance_service import histories_to_long_df, normalize_histories
from fit_service import (
    HORIZON_RISK_TOLERANCE,
    PROFILE_WEIGHTS,
    _min_max,
    calculate_fit_scores,
    get_weight_table,
)
from risk_service import (
    add_risk_analysis,
    calculate_risk_score,
    classify_risk,
    explain_risk,
    normalize,
)


# ---------- fixtures ----------

@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {
            "ticker": "AAA",
            "company_name": "Alpha Corp",
            "sector": "Tech",
            "current_price": 100.0,
            "market_cap": 1_000_000_000_000,
            "pe_ratio": 25,
            "forward_pe": 22,
            "profit_margin": 0.30,
            "revenue_growth": 0.05,
            "debt_to_equity": 30,
            "volatility": 0.20,
        },
        {
            "ticker": "BBB",
            "company_name": "Beta Inc",
            "sector": "Tech",
            "current_price": 50.0,
            "market_cap": 50_000_000_000,
            "pe_ratio": 60,
            "forward_pe": 55,
            "profit_margin": 0.10,
            "revenue_growth": 0.40,
            "debt_to_equity": 150,
            "volatility": 0.55,
        },
        {
            "ticker": "CCC",
            "company_name": "Gamma Ltd",
            "sector": "Energy",
            "current_price": 30.0,
            "market_cap": 200_000_000_000,
            "pe_ratio": 15,
            "forward_pe": 14,
            "profit_margin": 0.18,
            "revenue_growth": 0.10,
            "debt_to_equity": 60,
            "volatility": 0.30,
        },
    ])


@pytest.fixture
def sample_df_with_risk(sample_df):
    return add_risk_analysis(sample_df)


# ---------- risk_service ----------

class TestNormalize:
    def test_none_returns_midpoint(self):
        assert normalize(None, 0, 10) == 0.5

    def test_non_numeric_returns_midpoint(self):
        assert normalize("not a number", 0, 10) == 0.5

    def test_below_low_returns_zero(self):
        assert normalize(-5, 0, 10) == 0

    def test_above_high_returns_one(self):
        assert normalize(15, 0, 10) == 1

    def test_midpoint(self):
        assert normalize(5, 0, 10) == 0.5

    def test_at_low_boundary(self):
        assert normalize(0, 0, 10) == 0

    def test_at_high_boundary(self):
        assert normalize(10, 0, 10) == 1


class TestClassifyRisk:
    @pytest.mark.parametrize("score,expected", [
        (0, "Low"),
        (3.9, "Low"),
        (4, "Medium"),
        (6.9, "Medium"),
        (7, "High"),
        (10, "High"),
    ])
    def test_thresholds(self, score, expected):
        assert classify_risk(score) == expected


class TestCalculateRiskScore:
    def test_low_risk_inputs_score_low(self):
        row = {"volatility": 0.10, "pe_ratio": 10, "debt_to_equity": 10}
        assert calculate_risk_score(row) < 4

    def test_high_risk_inputs_score_high(self):
        row = {"volatility": 0.80, "pe_ratio": 100, "debt_to_equity": 250}
        assert calculate_risk_score(row) >= 7

    def test_missing_inputs_use_midpoint(self):
        row = {"volatility": None, "pe_ratio": None, "debt_to_equity": None}
        # All three normalize to 0.5 → raw = 0.5 → score 5.0
        assert calculate_risk_score(row) == 5.0


class TestExplainRisk:
    def test_includes_volatility_reason_when_high(self):
        row = {"volatility": 0.5, "pe_ratio": 10, "debt_to_equity": 10}
        assert "volatility" in explain_risk(row, 5.0)

    def test_includes_pe_reason_when_high(self):
        row = {"volatility": 0.1, "pe_ratio": 60, "debt_to_equity": 10}
        assert "P/E" in explain_risk(row, 5.0)

    def test_includes_debt_reason_when_high(self):
        row = {"volatility": 0.1, "pe_ratio": 10, "debt_to_equity": 150}
        assert "debt" in explain_risk(row, 5.0).lower()

    def test_fallback_when_all_moderate(self):
        row = {"volatility": 0.1, "pe_ratio": 10, "debt_to_equity": 10}
        assert "moderate" in explain_risk(row, 3.0).lower()


class TestAddRiskAnalysis:
    def test_adds_required_columns(self, sample_df):
        result = add_risk_analysis(sample_df)
        assert "risk_score" in result.columns
        assert "risk_level" in result.columns
        assert "risk_explanation" in result.columns

    def test_does_not_mutate_input(self, sample_df):
        before_cols = set(sample_df.columns)
        add_risk_analysis(sample_df)
        assert set(sample_df.columns) == before_cols

    def test_aaa_lower_risk_than_bbb(self, sample_df_with_risk):
        # AAA has lower volatility, lower P/E, lower debt → should score lower
        aaa = sample_df_with_risk[sample_df_with_risk["ticker"] == "AAA"].iloc[0]
        bbb = sample_df_with_risk[sample_df_with_risk["ticker"] == "BBB"].iloc[0]
        assert aaa["risk_score"] < bbb["risk_score"]


# ---------- fit_service ----------

class TestMinMax:
    def test_normal_range(self):
        result = _min_max(pd.Series([0.0, 5.0, 10.0]))
        assert result.tolist() == [0.0, 0.5, 1.0]

    def test_constant_series_returns_midpoint(self):
        result = _min_max(pd.Series([5.0, 5.0, 5.0]))
        assert result.tolist() == [0.5, 0.5, 0.5]

    def test_all_nan_returns_midpoint(self):
        result = _min_max(pd.Series([None, None, None]))
        assert result.tolist() == [0.5, 0.5, 0.5]

    def test_nan_filled_with_midpoint(self):
        result = _min_max(pd.Series([0.0, None, 10.0]))
        assert result.iloc[0] == 0.0
        assert result.iloc[1] == 0.5
        assert result.iloc[2] == 1.0


class TestCalculateFitScores:
    def test_returns_sorted_descending(self, sample_df_with_risk):
        result = calculate_fit_scores(sample_df_with_risk, "Moderate", "5 years")
        scores = result["fit_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_adds_required_columns(self, sample_df_with_risk):
        result = calculate_fit_scores(sample_df_with_risk, "Moderate", "5 years")
        assert "fit_score" in result.columns
        assert "fit_explanation" in result.columns

    def test_scores_in_zero_to_hundred(self, sample_df_with_risk):
        result = calculate_fit_scores(sample_df_with_risk, "Moderate", "5 years")
        assert result["fit_score"].min() >= 0
        assert result["fit_score"].max() <= 100

    def test_aggressive_prefers_high_growth(self, sample_df_with_risk):
        # BBB has the highest revenue_growth (0.40) — under Aggressive it should rank highest.
        result = calculate_fit_scores(sample_df_with_risk, "Aggressive", "10 years")
        assert result.iloc[0]["ticker"] == "BBB"

    def test_conservative_prefers_low_risk(self, sample_df_with_risk):
        # AAA has the lowest risk metrics — under Conservative it should rank highest.
        result = calculate_fit_scores(sample_df_with_risk, "Conservative", "1 year")
        assert result.iloc[0]["ticker"] == "AAA"

    def test_horizon_softens_risk_penalty(self, sample_df_with_risk):
        # For the same Moderate profile, BBB (high-risk/high-growth) should score
        # higher with a longer horizon because the risk penalty is softer.
        short = calculate_fit_scores(sample_df_with_risk, "Moderate", "1 year")
        long = calculate_fit_scores(sample_df_with_risk, "Moderate", "10 years")

        bbb_short = short[short["ticker"] == "BBB"]["fit_score"].iloc[0]
        bbb_long = long[long["ticker"] == "BBB"]["fit_score"].iloc[0]

        assert bbb_long >= bbb_short

    def test_does_not_mutate_input(self, sample_df_with_risk):
        before_cols = set(sample_df_with_risk.columns)
        calculate_fit_scores(sample_df_with_risk, "Moderate", "5 years")
        assert set(sample_df_with_risk.columns) == before_cols

    def test_unknown_profile_falls_back_to_moderate(self, sample_df_with_risk):
        result = calculate_fit_scores(sample_df_with_risk, "NotARealProfile", "5 years")
        assert len(result) == len(sample_df_with_risk)

    def test_single_ticker(self):
        df = pd.DataFrame([{
            "ticker": "AAA", "risk_score": 5.0, "risk_level": "Medium",
            "revenue_growth": 0.1, "profit_margin": 0.2,
            "market_cap": 1e11,
        }])
        result = calculate_fit_scores(df, "Moderate", "5 years")
        # Single ticker → all normalized to 0.5 → score should be 50
        assert result.iloc[0]["fit_score"] == 50.0


class TestProfileWeights:
    def test_all_profiles_present(self):
        assert set(PROFILE_WEIGHTS.keys()) == {
            "Conservative", "Moderate", "Moderate-Aggressive", "Aggressive"
        }

    def test_risk_weight_is_negative_for_all_profiles(self):
        for weights in PROFILE_WEIGHTS.values():
            assert weights["risk_score"] < 0

    def test_growth_weight_increases_with_aggressiveness(self):
        order = ["Conservative", "Moderate", "Moderate-Aggressive", "Aggressive"]
        growths = [PROFILE_WEIGHTS[p]["revenue_growth"] for p in order]
        assert growths == sorted(growths)

    def test_horizon_tolerance_monotonic(self):
        order = ["1 year", "3 years", "5 years", "10 years"]
        values = [HORIZON_RISK_TOLERANCE[h] for h in order]
        assert values == sorted(values)


class TestGetWeightTable:
    def test_returns_dataframe_with_expected_columns(self):
        table = get_weight_table("Moderate", "5 years")
        assert list(table.columns) == ["metric", "weight"]
        assert len(table) == 4

    def test_horizon_adjustment_applied(self):
        short = get_weight_table("Moderate", "1 year")
        long = get_weight_table("Moderate", "10 years")
        short_risk = short[short["metric"].str.contains("Risk")]["weight"].iloc[0]
        long_risk = long[long["metric"].str.contains("Risk")]["weight"].iloc[0]
        # Long horizon → less negative risk weight
        assert long_risk > short_risk


# ---------- finance_service helpers ----------

class TestHistoriesToLongDf:
    def test_empty_dict(self):
        result = histories_to_long_df({})
        assert result.empty
        assert set(result.columns) == {"date", "close", "ticker"}

    def test_multi_ticker_concat(self):
        h1 = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "close": [100.0, 110.0],
            "ticker": ["AAA", "AAA"],
        })
        h2 = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "close": [50.0, 55.0],
            "ticker": ["BBB", "BBB"],
        })
        result = histories_to_long_df({"AAA": h1, "BBB": h2})
        assert len(result) == 4
        assert set(result["ticker"].unique()) == {"AAA", "BBB"}


class TestNormalizeHistories:
    def test_first_row_normalized_to_hundred(self):
        h = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "close": [50.0, 75.0, 100.0],
            "ticker": ["AAA", "AAA", "AAA"],
        })
        normalized = normalize_histories({"AAA": h})
        result = normalized["AAA"]
        assert result["close"].iloc[0] == 100.0
        assert result["close"].iloc[1] == 150.0
        assert result["close"].iloc[2] == 200.0

    def test_skips_zero_base(self):
        h = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "close": [0.0, 10.0],
            "ticker": ["AAA", "AAA"],
        })
        normalized = normalize_histories({"AAA": h})
        assert "AAA" not in normalized

    def test_skips_nan_base(self):
        h = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "close": [math.nan, 10.0],
            "ticker": ["AAA", "AAA"],
        })
        normalized = normalize_histories({"AAA": h})
        assert "AAA" not in normalized

    def test_empty_dict(self):
        assert normalize_histories({}) == {}

    def test_does_not_mutate_input(self):
        h = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "close": [50.0, 75.0],
            "ticker": ["AAA", "AAA"],
        })
        original_first = h["close"].iloc[0]
        normalize_histories({"AAA": h})
        assert h["close"].iloc[0] == original_first
