import os

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from agents import (
    financial_agent,
    improvement_agent,
    report_agent,
    risk_agent,
    sentiment_agent,
    test_agent,
)
from finance_service import (
    get_price_history,
    get_stock_data,
    histories_to_long_df,
    normalize_histories,
)
from fit_service import PROFILE_WEIGHTS, calculate_fit_scores, get_weight_table
from pdf_service import generate_pdf_report
from risk_service import add_risk_analysis
from search_service import resolve_companies_to_tickers

load_dotenv()

st.set_page_config(
    page_title="AlphaLens",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AlphaLens")
st.subheader("Multi-Agent Investment Research Copilot")

st.warning(
    "Educational prototype only. This tool does not provide financial advice "
    "or buy/sell recommendations."
)

PROFILE_OPTIONS = list(PROFILE_WEIGHTS.keys())
HORIZON_OPTIONS = ["1 year", "3 years", "5 years", "10 years"]

SESSION_KEYS = [
    "df", "tickers", "histories", "resolved",
    "financial_summary", "sentiment_summary", "risk_summary",
    "final_report", "test_review", "improved_report",
    "run_profile", "run_horizon", "pdf_path",
]


def clear_session():
    for key in SESSION_KEYS:
        st.session_state.pop(key, None)


with st.sidebar:
    st.header("Input")

    companies_input = st.text_input(
        "Companies or stock tickers",
        value="Nvidia, AMD, Palantir, Amazon"
    )

    risk_profile = st.selectbox(
        "Risk profile",
        PROFILE_OPTIONS,
        index=2
    )

    horizon = st.selectbox(
        "Investment horizon",
        HORIZON_OPTIONS,
        index=2
    )

    run_button = st.button("Generate Investment Brief", type="primary")

    if "df" in st.session_state:
        st.divider()
        if st.button("Clear results"):
            clear_session()
            st.rerun()


if run_button:
    companies = [
        company.strip()
        for company in companies_input.split(",")
        if company.strip()
    ]

    if not companies:
        st.error("Please enter at least one company or ticker.")
        st.stop()

    with st.spinner("Search Agent is resolving company names into stock tickers..."):
        resolved_companies = resolve_companies_to_tickers(companies)

    tickers = [
        item["ticker"]
        for item in resolved_companies
        if item["ticker"]
    ]

    if not tickers:
        st.error("No valid tickers found. Try using known company names or stock symbols.")
        st.stop()

    with st.spinner("Financial Data Agent is collecting public market data..."):
        df = get_stock_data(tickers)

    with st.spinner("Fetching 1-year price history..."):
        histories = get_price_history(tickers)

    with st.spinner("Risk Agent is calculating risk scores..."):
        df = add_risk_analysis(df)

    with st.spinner("Financial Agent is analyzing metrics..."):
        financial_summary = financial_agent(df)

    with st.spinner("News & Sentiment Agent is preparing sentiment framework..."):
        sentiment_summary = sentiment_agent(tickers)

    with st.spinner("Risk Agent is explaining risk..."):
        risk_summary = risk_agent(df)

    with st.spinner("Report Agent is generating final research brief..."):
        final_report = report_agent(
            df=df,
            financial_summary=financial_summary,
            sentiment_summary=sentiment_summary,
            risk_summary=risk_summary,
            profile=risk_profile,
            horizon=horizon
        )

    with st.spinner("Test Agent is auditing the final brief..."):
        run_fit_df = calculate_fit_scores(df, risk_profile, horizon)
        test_review = test_agent(
            final_report=final_report,
            profile=risk_profile,
            horizon=horizon,
            fit_df=run_fit_df
        )

    with st.spinner("Improvement Agent is polishing the brief using QA feedback..."):
        improved_report = improvement_agent(
            final_report=final_report,
            test_review=test_review,
            df=df,
            profile=risk_profile,
            horizon=horizon
        )

    pdf_path = "alphalens_report.pdf"
    generate_pdf_report(
        filename=pdf_path,
        df=df,
        financial_summary=financial_summary,
        sentiment_summary=sentiment_summary,
        risk_summary=risk_summary,
        final_report=improved_report
    )

    st.session_state.df = df
    st.session_state.tickers = tickers
    st.session_state.histories = histories
    st.session_state.resolved = resolved_companies
    st.session_state.financial_summary = financial_summary
    st.session_state.sentiment_summary = sentiment_summary
    st.session_state.risk_summary = risk_summary
    st.session_state.final_report = final_report
    st.session_state.test_review = test_review
    st.session_state.improved_report = improved_report
    st.session_state.run_profile = risk_profile
    st.session_state.run_horizon = horizon
    st.session_state.pdf_path = pdf_path


if "df" not in st.session_state:
    st.info("Enter company names or tickers in the sidebar and click Generate Investment Brief.")
    st.stop()


df = st.session_state.df
tickers = st.session_state.tickers
histories = st.session_state.histories
resolved_companies = st.session_state.resolved
financial_summary = st.session_state.financial_summary
sentiment_summary = st.session_state.sentiment_summary
risk_summary = st.session_state.risk_summary
final_report = st.session_state.final_report
test_review = st.session_state.test_review
improved_report = st.session_state.improved_report
run_profile = st.session_state.run_profile
run_horizon = st.session_state.run_horizon
pdf_path = st.session_state.pdf_path

fit_df = calculate_fit_scores(df, risk_profile, horizon)


overview_tab, charts_tab, risk_tab, fit_tab, agents_tab, qa_tab, brief_tab = st.tabs([
    "📊 Overview",
    "📈 Price Charts",
    "⚠️ Risk Analysis",
    "🎯 Best Fit",
    "🤖 Agent Reasoning",
    "🧪 QA Agent",
    "📄 Final Brief",
])


with overview_tab:
    st.subheader("Resolved tickers")

    st.dataframe(
        pd.DataFrame(resolved_companies),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Snapshot")

    metric_cols = st.columns(min(len(df), 4) or 1)

    for idx, row in df.iterrows():
        col = metric_cols[idx % len(metric_cols)]

        with col:
            price = row.get("current_price")
            growth = row.get("revenue_growth")

            price_label = f"${price:,.2f}" if pd.notna(price) else "N/A"
            growth_label = (
                f"{growth * 100:.1f}% rev. growth"
                if pd.notna(growth) else "Growth N/A"
            )

            st.metric(
                label=f"{row['ticker']} — {row.get('sector', 'N/A')}",
                value=price_label,
                delta=growth_label
            )

    st.subheader("Company comparison")

    display_columns = [
        "ticker", "company_name", "sector", "current_price",
        "market_cap", "pe_ratio", "revenue_growth", "profit_margin",
        "debt_to_equity", "volatility", "risk_score", "risk_level",
    ]

    available_columns = [col for col in display_columns if col in df.columns]

    st.dataframe(
        df[available_columns],
        use_container_width=True,
        hide_index=True
    )


with charts_tab:
    if not histories:
        st.warning("No price history available for the selected tickers.")
    else:
        view_mode = st.radio(
            "Chart mode",
            ["Normalized (start = 100)", "Absolute price"],
            horizontal=True,
            help="Normalized mode makes relative performance easier to compare across companies."
        )

        if view_mode.startswith("Normalized"):
            chart_data = histories_to_long_df(normalize_histories(histories))
            y_title = "Normalized close (start = 100)"
        else:
            chart_data = histories_to_long_df(histories)
            y_title = "Close price (USD)"

        if chart_data.empty:
            st.warning("No price data to chart.")
        else:
            price_fig = px.line(
                chart_data,
                x="date",
                y="close",
                color="ticker",
                title="1-year price evolution",
                labels={"close": y_title, "date": "Date"}
            )
            price_fig.update_layout(hovermode="x unified")
            st.plotly_chart(price_fig, use_container_width=True)

        st.subheader("Focus on a single ticker")

        focus_ticker = st.selectbox(
            "Ticker",
            sorted(histories.keys()),
            key="focus_ticker"
        )

        focus_frame = histories.get(focus_ticker)

        if focus_frame is not None and not focus_frame.empty:
            focus_fig = px.area(
                focus_frame,
                x="date",
                y="close",
                title=f"{focus_ticker} — 1-year close price",
                labels={"close": "Close price (USD)", "date": "Date"}
            )
            st.plotly_chart(focus_fig, use_container_width=True)


with risk_tab:
    st.subheader("Risk score by ticker")

    risk_fig = px.bar(
        df,
        x="ticker",
        y="risk_score",
        color="risk_level",
        text="risk_score",
        title="Composite risk score (0–10)",
        category_orders={"risk_level": ["Low", "Medium", "High"]},
        color_discrete_map={"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e74c3c"}
    )
    st.plotly_chart(risk_fig, use_container_width=True)

    st.caption(
        "Composite of volatility (45%), P/E ratio (35%), and debt-to-equity (20%). "
        "See risk_service.py for the exact formula."
    )

    st.subheader("Risk vs growth")

    scatter_df = df.dropna(subset=["risk_score", "revenue_growth"]).copy()

    if not scatter_df.empty:
        scatter_df["market_cap_size"] = scatter_df["market_cap"].fillna(scatter_df["market_cap"].median())

        scatter_fig = px.scatter(
            scatter_df,
            x="revenue_growth",
            y="risk_score",
            size="market_cap_size",
            color="sector",
            hover_name="ticker",
            hover_data={"company_name": True, "market_cap_size": False, "market_cap": ":,.0f"},
            title="Where each company sits on the risk/growth map",
            labels={"revenue_growth": "Revenue growth", "risk_score": "Risk score (0–10)"}
        )
        scatter_fig.update_layout(xaxis_tickformat=".0%")
        st.plotly_chart(scatter_fig, use_container_width=True)
    else:
        st.info("Not enough data to plot risk vs growth.")

    st.subheader("Sector exposure (by market cap)")

    sector_df = df.dropna(subset=["market_cap"]).copy()

    if not sector_df.empty:
        treemap_fig = px.treemap(
            sector_df,
            path=["sector", "ticker"],
            values="market_cap",
            color="risk_score",
            color_continuous_scale="RdYlGn_r",
            title="Sector and company weight by market cap, colored by risk"
        )
        st.plotly_chart(treemap_fig, use_container_width=True)
    else:
        st.info("Market cap data unavailable for sector treemap.")

    st.subheader("Per-company risk explanation")

    for _, row in df.iterrows():
        st.markdown(f"**{row['ticker']}** — {row.get('risk_explanation', 'N/A')}")


with fit_tab:
    st.subheader(f"Best fit for a {risk_profile} profile over {horizon}")

    st.caption(
        "This ranking is an **explainable composite score**, not a buy or sell recommendation. "
        "It re-computes live as you change the risk profile or horizon in the sidebar."
    )

    if risk_profile != run_profile or horizon != run_horizon:
        st.info(
            f"Ranking reflects the current selection ({risk_profile}, {horizon}). "
            f"Agent reasoning and Final Brief still reflect the run-time selection "
            f"({run_profile}, {run_horizon}). Click **Generate Investment Brief** to refresh those."
        )

    top_row = fit_df.iloc[0]

    st.success(
        f"🏆 Highest fit score: **{top_row['ticker']}** "
        f"({top_row.get('company_name', '')}) — score {top_row['fit_score']}/100"
    )
    st.write(top_row["fit_explanation"])

    fit_chart = px.bar(
        fit_df,
        x="fit_score",
        y="ticker",
        orientation="h",
        color="fit_score",
        color_continuous_scale="Viridis",
        title=f"Fit score ranking — {risk_profile} / {horizon}",
        text="fit_score",
        labels={"fit_score": "Fit score (0–100)", "ticker": "Ticker"}
    )
    fit_chart.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(fit_chart, use_container_width=True)

    ranking_columns = [
        "ticker", "company_name", "fit_score", "risk_score", "risk_level",
        "revenue_growth", "profit_margin", "market_cap", "fit_explanation"
    ]
    available_ranking_columns = [col for col in ranking_columns if col in fit_df.columns]

    st.dataframe(
        fit_df[available_ranking_columns],
        use_container_width=True,
        hide_index=True
    )

    with st.expander("How is the fit score calculated?"):
        st.markdown(
            "Each metric is min-max normalized across the selected companies, then "
            "combined using profile-specific weights. The risk weight is also adjusted "
            "by the investment horizon — longer horizons tolerate more risk."
        )
        st.markdown(f"**Weights for {risk_profile} / {horizon}:**")
        st.dataframe(
            get_weight_table(risk_profile, horizon),
            use_container_width=True,
            hide_index=True
        )
        st.caption(
            "Negative weight on risk score means lower-risk companies score higher. "
            "Weights live in fit_service.PROFILE_WEIGHTS and are editable."
        )


with agents_tab:
    st.caption(
        f"Agent outputs from the last run "
        f"(profile: {run_profile}, horizon: {run_horizon})."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Financial Agent")
        st.write(financial_summary)

    with col2:
        st.subheader("News & Sentiment Agent")
        st.write(sentiment_summary)

    with col3:
        st.subheader("Risk Agent")
        st.write(risk_summary)


with qa_tab:
    st.subheader("QA / Test Agent review")

    st.caption(
        f"Automated audit of the final brief generated for "
        f"**{run_profile}** / **{run_horizon}**. The Test Agent checks for "
        f"buy/sell language, profile alignment, required sections, internal "
        f"consistency, and educational framing."
    )

    verdict_line = next(
        (line for line in test_review.splitlines() if line.upper().startswith("VERDICT:")),
        None
    )

    if verdict_line:
        if "PASS" in verdict_line.upper():
            st.success(verdict_line)
        elif "FAIL" in verdict_line.upper():
            st.error(verdict_line)
        else:
            st.info(verdict_line)

    st.markdown(test_review)


with brief_tab:
    st.caption(
        f"Final brief generated for profile **{run_profile}** over **{run_horizon}**. "
        f"Shown below is the **improved version** produced by the Improvement Agent "
        f"using the QA Agent's feedback. The PDF download contains this same version."
    )

    view = st.radio(
        "View",
        ["✨ Improved (used in PDF)", "📝 Original (pre-improvement)"],
        horizontal=True,
        key="brief_view"
    )

    if view.startswith("✨"):
        st.markdown(improved_report)
    else:
        st.markdown(final_report)

    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📥 Download PDF Report (improved version)",
                data=pdf_file,
                file_name="alphalens_report.pdf",
                mime="application/pdf"
            )
