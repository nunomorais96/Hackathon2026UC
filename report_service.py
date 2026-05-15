def generate_markdown_report(df, financial_summary, sentiment_summary, risk_summary, final_report):
    markdown = "# AlphaLens Investment Research Brief\n\n"

    markdown += "## Company Comparison\n\n"
    markdown += df[
        [
            "ticker",
            "company_name",
            "sector",
            "current_price",
            "pe_ratio",
            "revenue_growth",
            "volatility",
            "risk_score",
            "risk_level",
        ]
    ].to_markdown(index=False)

    markdown += "\n\n## Financial Agent Summary\n\n"
    markdown += financial_summary

    markdown += "\n\n## News & Sentiment Agent Summary\n\n"
    markdown += sentiment_summary

    markdown += "\n\n## Risk Agent Summary\n\n"
    markdown += risk_summary

    markdown += "\n\n## Final Investment Brief\n\n"
    markdown += final_report

    markdown += "\n\n---\n"
    markdown += "Disclaimer: This is educational research only and not financial advice."

    return markdown