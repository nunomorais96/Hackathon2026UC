import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

from search_service import resolve_companies_to_tickers
from finance_service import get_stock_data
from risk_service import add_risk_analysis
from agents import financial_agent, sentiment_agent, risk_agent, report_agent
from pdf_service import generate_pdf_report

load_dotenv()

st.set_page_config(
    page_title="AlphaLens",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AlphaLens")
st.subheader("Multi-Agent Investment Research Copilot")

st.warning(
    "Educational prototype only. This tool does not provide financial advice or buy/sell recommendations."
)

with st.sidebar:
    st.header("Input")

    companies_input = st.text_input(
        "Companies or stock tickers",
        value="Nvidia, AMD, Palantir, Amazon"
    )

    risk_profile = st.selectbox(
        "Risk profile",
        ["Conservative", "Moderate", "Moderate-Aggressive", "Aggressive"],
        index=2
    )

    horizon = st.selectbox(
        "Investment horizon",
        ["1 year", "3 years", "5 years", "10 years"],
        index=2
    )

    run_button = st.button("Generate Investment Brief")

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

    st.header("0. Search Agent — Resolved Companies")

    st.dataframe(
        resolved_companies,
        use_container_width=True,
        hide_index=True
    )

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

    with st.spinner("Risk Agent is calculating risk scores..."):
        df = add_risk_analysis(
            df,
            profile=risk_profile,
            horizon=horizon
        )

    st.success("Agents completed the first analysis.")

    st.header("1. Company Comparison")

    metric_cols = st.columns(len(df))

    for col, (_, row) in zip(metric_cols, df.iterrows()):
        with col:
            price = row.get("current_price")
            price_text = f"${price:.2f}" if price is not None else "N/A"

            st.metric(
                label=row.get("ticker", "N/A"),
                value=price_text,
                delta=row.get("company_name", "")
            )

            st.caption(f"Sector: {row.get('sector', 'N/A')}")
            st.caption(f"P/E: {row.get('pe_ratio', 'N/A')}")
            st.caption(f"Risk: {row.get('risk_score', 'N/A')}/10")

    st.subheader("Market Cap Comparison")

    fig_market_cap = px.bar(
        df,
        x="ticker",
        y="market_cap",
        text="market_cap",
        title="Market Cap by Company"
    )

    fig_market_cap.update_traces(
        texttemplate="%{y:.2s}",
        textposition="outside"
    )

    fig_market_cap.update_layout(
        yaxis_title="Market Cap",
        xaxis_title="Company"
    )

    st.plotly_chart(fig_market_cap, use_container_width=True)

    st.subheader("Key Financial Metrics")

    metrics_columns = [
        "ticker",
        "pe_ratio",
        "revenue_growth",
        "profit_margin",
        "volatility",
        "debt_to_equity",
    ]

    available_metrics_columns = [
        col for col in metrics_columns
        if col in df.columns
    ]

    metrics_df = df[available_metrics_columns].copy()

    percentage_columns = [
        "revenue_growth",
        "profit_margin",
        "volatility"
    ]

    for col in percentage_columns:
        if col in metrics_df.columns:
            metrics_df[col] = metrics_df[col] * 100

    metrics_df = metrics_df.rename(columns={
        "ticker": "Ticker",
        "pe_ratio": "P/E Ratio",
        "revenue_growth": "Revenue Growth (%)",
        "profit_margin": "Profit Margin (%)",
        "volatility": "Volatility (%)",
        "debt_to_equity": "Debt to Equity (%)",
    })

    metrics_long = metrics_df.melt(
        id_vars="Ticker",
        var_name="Metric",
        value_name="Value"
    )

    metrics_long["Value"] = metrics_long["Value"].round(2)

    fig_metrics = px.bar(
        metrics_long,
        x="Ticker",
        y="Value",
        color="Metric",
        barmode="group",
        title="Financial Metrics by Company"
    )

    fig_metrics.update_traces(
        textposition="outside"
    )

    fig_metrics.update_layout(
        yaxis_title="Metric Value",
        xaxis_title="Company",
        legend_title="Metric"
    )

    st.plotly_chart(fig_metrics, use_container_width=True)

    with st.expander("View raw company data"):
        display_columns = [
            "ticker",
            "company_name",
            "sector",
            "current_price",
            "market_cap",
            "pe_ratio",
            "revenue_growth",
            "profit_margin",
            "debt_to_equity",
            "volatility",
            "risk_score",
            "risk_level",
        ]

        available_columns = [
            col for col in display_columns
            if col in df.columns
        ]

        st.dataframe(
            df[available_columns],
            use_container_width=True,
            hide_index=True
        )

    st.header("2. Risk Score")

    risk_cols = st.columns(len(df))

    for col, (_, row) in zip(risk_cols, df.iterrows()):
        with col:
            st.metric(
                label=f"{row.get('ticker', 'N/A')} Risk",
                value=f"{row.get('risk_score', 'N/A')}/10",
                delta=row.get("risk_level", "N/A")
            )
            st.caption(row.get("risk_explanation", ""))

    fig_risk = px.bar(
        df.sort_values("risk_score", ascending=False),
        x="ticker",
        y="risk_score",
        color="risk_level",
        title=f"Risk Score by Company — {risk_profile}, {horizon}",
        text="risk_score",
        range_y=[0, 10]
    )

    fig_risk.update_traces(
        textposition="outside"
    )

    fig_risk.update_layout(
        yaxis_title="Risk Score",
        xaxis_title="Company"
    )

    st.plotly_chart(fig_risk, use_container_width=True)

    st.header("3. Final Investment Brief")

    with st.spinner("AI Agents are generating the final investment research brief..."):
        financial_summary = financial_agent(df)
        sentiment_summary = sentiment_agent(tickers)
        risk_summary = risk_agent(df)

        final_report = report_agent(
            df=df,
            financial_summary=financial_summary,
            sentiment_summary=sentiment_summary,
            risk_summary=risk_summary,
            profile=risk_profile,
            horizon=horizon
        )

    st.markdown(final_report)

    pdf_filename = "alphalens_report.pdf"

    generate_pdf_report(
        filename=pdf_filename,
        df=df,
        financial_summary=financial_summary,
        sentiment_summary=sentiment_summary,
        risk_summary=risk_summary,
        final_report=final_report
    )

    with open(pdf_filename, "rb") as pdf_file:
        st.download_button(
            label="Download PDF Report",
            data=pdf_file,
            file_name="alphalens_report.pdf",
            mime="application/pdf"
        )

else:
    st.info("Enter company names or tickers in the sidebar and click Generate Investment Brief.")