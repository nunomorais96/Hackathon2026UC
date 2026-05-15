"""End-to-end smoke test for the AlphaLens pipeline.

Runs the full flow against real yfinance + Groq and asserts the outputs are
shape-correct and free of obvious guardrail violations. Use:

    python smoke_test.py                 # default tickers (AAPL, MSFT)
    python smoke_test.py NVDA AMD        # custom tickers

Exit code is 0 on success, 1 on the first failure. Requires GROQ_API_KEY in
the environment (.env is loaded automatically) for the LLM-dependent checks.
"""

import os
import re
import sys

from dotenv import load_dotenv

from agents import (
    financial_agent,
    improvement_agent,
    report_agent,
    risk_agent,
    sentiment_agent,
)
from finance_service import get_price_history, get_stock_data
from fit_service import calculate_fit_scores
from pdf_service import generate_pdf_report
from risk_service import add_risk_analysis
from search_service import resolve_companies_to_tickers


BUY_SELL_PATTERNS = [
    r"\bi recommend (?:buying|selling)\b",
    r"\byou should (?:buy|sell)\b",
    r"\bwe (?:recommend|advise) (?:buying|selling)\b",
    r"\bbuy (?:this|the) stock\b",
    r"\bsell (?:this|the) stock\b",
    r"\bstrong buy\b",
    r"\bstrong sell\b",
]


PROFILE = "Moderate-Aggressive"
HORIZON = "5 years"


class SmokeTestFailure(AssertionError):
    pass


def step(label):
    print(f"\n--- {label} ---")


def check(condition, message):
    if condition:
        print(f"  ✅ {message}")
    else:
        print(f"  ❌ {message}")
        raise SmokeTestFailure(message)


def find_buy_sell_phrases(text):
    text_lower = text.lower()
    return [
        pattern
        for pattern in BUY_SELL_PATTERNS
        if re.search(pattern, text_lower)
    ]


def run(companies):
    load_dotenv()

    has_llm_key = bool(os.getenv("GROQ_API_KEY"))
    if not has_llm_key:
        print("⚠️  GROQ_API_KEY not set — LLM-dependent checks will be skipped.")

    step(f"Resolving {len(companies)} companies")
    resolved = resolve_companies_to_tickers(companies)
    tickers = [item["ticker"] for item in resolved if item["ticker"]]
    check(len(tickers) > 0, f"At least one ticker resolved (got {tickers})")

    step("Fetching stock data")
    df = get_stock_data(tickers)
    check(len(df) == len(tickers), f"One row per ticker (expected {len(tickers)}, got {len(df)})")
    check("ticker" in df.columns, "DataFrame has ticker column")
    check("volatility" in df.columns, "DataFrame has volatility column")

    step("Fetching price history")
    histories = get_price_history(tickers)
    check(len(histories) > 0, f"At least one price history fetched ({len(histories)} of {len(tickers)})")

    step("Calculating risk analysis")
    df = add_risk_analysis(df)
    check("risk_score" in df.columns, "risk_score column added")
    check("risk_level" in df.columns, "risk_level column added")
    check(
        df["risk_score"].between(0, 10).all(),
        "All risk scores are in [0, 10]"
    )
    check(
        df["risk_level"].isin(["Low", "Medium", "High"]).all(),
        "All risk levels are Low/Medium/High"
    )

    step("Calculating fit scores")
    fit_df = calculate_fit_scores(df, PROFILE, HORIZON)
    check(
        fit_df["fit_score"].tolist() == sorted(fit_df["fit_score"].tolist(), reverse=True),
        "Fit scores are sorted in descending order"
    )
    check(
        fit_df["fit_score"].between(0, 100).all(),
        "All fit scores are in [0, 100]"
    )
    check("fit_explanation" in fit_df.columns, "fit_explanation column added")

    if not has_llm_key:
        print("\n⚠️  Skipping remaining LLM-dependent checks.")
        return

    step("Running financial agent")
    financial_summary = financial_agent(df)
    check(isinstance(financial_summary, str) and len(financial_summary) > 50,
          f"Financial agent returned non-trivial output ({len(financial_summary)} chars)")

    step("Running sentiment agent")
    sentiment_summary = sentiment_agent(tickers)
    check(isinstance(sentiment_summary, str) and len(sentiment_summary) > 50,
          f"Sentiment agent returned non-trivial output ({len(sentiment_summary)} chars)")

    step("Running risk agent")
    risk_summary = risk_agent(df)
    check(isinstance(risk_summary, str) and len(risk_summary) > 50,
          f"Risk agent returned non-trivial output ({len(risk_summary)} chars)")

    step("Running report agent")
    final_report = report_agent(
        df=df,
        financial_summary=financial_summary,
        sentiment_summary=sentiment_summary,
        risk_summary=risk_summary,
        profile=PROFILE,
        horizon=HORIZON,
    )
    check(isinstance(final_report, str) and len(final_report) > 200,
          f"Final report has substantive content ({len(final_report)} chars)")

    step("Checking final report for buy/sell phrases")
    matches = find_buy_sell_phrases(final_report)
    check(
        len(matches) == 0,
        f"No buy/sell recommendation phrases found"
        + (f" (matched: {matches})" if matches else "")
    )

    step("Running Improvement Agent")
    improved = improvement_agent(
        final_report=final_report,
        df=df,
        profile=PROFILE,
        horizon=HORIZON,
    )
    check(isinstance(improved, str) and len(improved) > 200,
          f"Improvement agent returned substantive output ({len(improved)} chars)")

    improved_matches = find_buy_sell_phrases(improved)
    check(
        len(improved_matches) == 0,
        f"Improved brief contains no buy/sell phrases"
        + (f" (matched: {improved_matches})" if improved_matches else "")
    )

    required_sections = [
        "executive summary", "company comparison", "main risks",
        "portfolio fit", "questions"
    ]
    improved_lower = improved.lower()
    missing = [s for s in required_sections if s not in improved_lower]
    check(
        len(missing) == 0,
        f"Improved brief contains all required sections"
        + (f" (missing: {missing})" if missing else "")
    )

    check(
        PROFILE.lower() in improved_lower and HORIZON.lower() in improved_lower,
        f"Improved brief references profile ({PROFILE}) and horizon ({HORIZON})"
    )

    step("Generating PDF from improved brief")
    pdf_path = "alphalens_report.pdf"
    generate_pdf_report(
        filename=pdf_path,
        df=df,
        financial_summary=financial_summary,
        sentiment_summary=sentiment_summary,
        risk_summary=risk_summary,
        final_report=improved,
    )
    check(os.path.exists(pdf_path), f"PDF was written to {pdf_path}")
    check(os.path.getsize(pdf_path) > 1000, f"PDF is non-trivially sized ({os.path.getsize(pdf_path)} bytes)")


def main():
    companies = sys.argv[1:] or ["AAPL", "MSFT"]
    print(f"Running smoke test with companies: {companies}")

    try:
        run(companies)
    except SmokeTestFailure as exc:
        print(f"\n💥 Smoke test FAILED: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"\n💥 Smoke test crashed: {type(exc).__name__}: {exc}")
        sys.exit(1)

    print("\n🎉 All smoke checks passed.")


if __name__ == "__main__":
    main()
