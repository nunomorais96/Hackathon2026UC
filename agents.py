import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None


def call_llm(prompt):
    if client is None:
        return """
LLM not configured.

Create a .env file with:
GROQ_API_KEY=your_api_key_here
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial research assistant. "
                    "Do not give financial advice. "
                    "Do not provide buy or sell recommendations. "
                    "Provide educational, explainable analysis only."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


def financial_agent(df):
    prompt = f"""
You are the Financial Data Agent.

Analyze these public financial metrics:

{df.to_string(index=False)}

Return:
1. Key strengths
2. Key weaknesses
3. Important observations

Do not provide investment advice.
"""
    return call_llm(prompt)


def sentiment_agent(tickers):
    prompt = f"""
You are the News and Sentiment Agent.

Analyze possible market sentiment drivers for these tickers:
{", ".join(tickers)}

Do not invent specific news headlines.

Return:
1. Positive catalysts
2. Negative catalysts
3. Market risks
4. What investors should verify in recent public news

Do not provide buy/sell recommendations.
"""
    return call_llm(prompt)


def risk_agent(df):
    prompt = f"""
You are the Risk Analysis Agent.

Analyze this risk data:

{df[["ticker", "risk_score", "risk_level", "risk_explanation"]].to_string(index=False)}

Return:
1. Which companies are riskier
2. Why they are riskier
3. Portfolio concentration risks
4. Risk mitigation questions

Do not provide investment advice.
"""
    return call_llm(prompt)


def test_agent(
    final_report,
    profile,
    horizon,
    fit_df
):
    top_pick = fit_df.iloc[0] if not fit_df.empty else None
    top_summary = (
        f"{top_pick['ticker']} (fit score {top_pick['fit_score']}/100)"
        if top_pick is not None else "N/A"
    )

    prompt = f"""
You are the QA / Test Agent for AlphaLens.

Your job is to audit the final investment brief below and verify it meets the
project's hard rules. You do NOT rewrite the brief. You output a structured review.

Investor context:
- Risk profile: {profile}
- Investment horizon: {horizon}
- Highest-fit ticker by composite score: {top_summary}

Final brief to audit:
---
{final_report}
---

Audit checklist:
1. Buy/sell recommendations — flag ANY phrase that recommends buying or selling
   a specific stock. The brief must be educational only.
2. Profile alignment — does the brief reference the investor's risk profile
   ({profile}) and horizon ({horizon})?
3. Required sections — Executive Summary, Company Comparison, Main Risks,
   Portfolio Fit, Questions the investor should answer. Flag any missing.
4. Internal consistency — does any claim contradict the company comparison
   data or risk explanations?
5. Educational framing — is the tone explanatory rather than advisory?

Return your review in this exact format:

VERDICT: PASS or FAIL
(FAIL if any buy/sell recommendation is present, otherwise PASS even with warnings)

ISSUES:
- (list each issue with the checklist number, or "None")

WARNINGS:
- (softer concerns that did not fail the audit, or "None")

SUGGESTIONS:
- (concrete improvements the Report Agent could make next time, or "None")
"""
    return call_llm(prompt)


def improvement_agent(
    final_report,
    test_review,
    df,
    profile,
    horizon
):
    prompt = f"""
You are the Report Improvement Agent.

Your job: produce a polished, improved version of the investment brief below,
addressing every issue and suggestion raised by the QA Test Agent while
preserving the project's hard rules.

Hard rules (NEVER violate):
- No buy or sell recommendations
- Educational framing only
- Do not invent financial data — only use what's in the company comparison below

Investor context:
- Risk profile: {profile}
- Investment horizon: {horizon}

Company comparison data (the source of truth):
{df.to_string(index=False)}

Original brief to improve:
---
{final_report}
---

QA review (issues / warnings / suggestions to address):
---
{test_review}
---

Produce the improved brief with these characteristics:
1. Address every ISSUE flagged by QA (especially any buy/sell language)
2. Incorporate the SUGGESTIONS
3. Ensure all required sections are present and clearly titled:
   - Executive Summary
   - Company Comparison
   - Main Risks
   - Portfolio Fit
   - Questions the investor should answer
4. Explicitly reference the investor's profile ({profile}) and horizon ({horizon})
5. Tighten language; remove filler; keep claims grounded in the data above
6. Use markdown headings (##) and bullet points for readability
7. End with a one-line educational disclaimer

Output ONLY the improved markdown brief. Do NOT include a changelog, preamble,
or commentary about what you changed.
"""
    return call_llm(prompt)


def report_agent(
    df,
    financial_summary,
    sentiment_summary,
    risk_summary,
    profile,
    horizon
):
    prompt = f"""
You are the Report Generator Agent.

Create a final investment research brief.

Investor profile:
- Risk profile: {profile}
- Investment horizon: {horizon}

Company data:
{df.to_string(index=False)}

Financial Agent output:
{financial_summary}

News and Sentiment Agent output:
{sentiment_summary}

Risk Agent output:
{risk_summary}

Generate:
1. Executive Summary
2. Company Comparison
3. Main Risks
4. Portfolio Fit
5. Questions the investor should answer before investing

Important:
- Educational only
- No buy/sell recommendation
- Be clear and structured
"""
    return call_llm(prompt)