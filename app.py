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
        use_container_width=True
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
        df = add_risk_analysis(df)

    st.success("Agents completed the first analysis.")

    st.header("1. Company Comparison")

    display_columns = [
        "ticker",
        "company_name",
        "sector",
        "current_price",
        "market_cap",
        "pe_ratio",
        "revenue_growth",
        "volatility",
        "risk_score",
        "risk_level",
    ]

    available_columns = [col for col in display_columns if col in df.columns]

    st.dataframe(
        df[available_columns],
        use_container_width=True
    )

    st.header("2. Risk Score")

    fig = px.bar(
        df,
        x="ticker",
        y="risk_score",
        color="risk_level",
        title="Risk Score by Company",
        text="risk_score"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.header("3. Agent Reasoning")

    with st.spinner("Financial Agent is analyzing metrics..."):
        financial_summary = financial_agent(df)

    with st.spinner("News & Sentiment Agent is preparing sentiment framework..."):
        sentiment_summary = sentiment_agent(tickers)

    with st.spinner("Risk Agent is explaining risk..."):
        risk_summary = risk_agent(df)

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

    st.header("4. Final Investment Brief")

    with st.spinner("Report Agent is generating final research brief..."):
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