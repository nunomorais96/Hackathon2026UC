import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = None

if api_key:
    client = OpenAI(api_key=api_key)


def call_llm(prompt):

    if client is None:
        return """
LLM not configured.

To enable AI responses:
1. Create a .env file
2. Add:
OPENAI_API_KEY=your_api_key
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial research assistant. "
                    "Do not provide financial advice. "
                    "Provide explainable educational analysis only."
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

Analyze the following public financial metrics and summarize:
- strengths
- weaknesses
- observations

Data:
{df.to_string(index=False)}

Do not provide investment advice.
"""

    return call_llm(prompt)


def sentiment_agent(tickers):

    prompt = f"""
You are the News and Sentiment Agent.

Analyze possible market sentiment drivers for:
{', '.join(tickers)}

Provide:
- positive catalysts
- negative catalysts
- market risks
- things investors should verify

Do not invent fake news headlines.
"""

    return call_llm(prompt)


def risk_agent(df):

    prompt = f"""
You are the Risk Analysis Agent.

Analyze the following risk data:
{df[['ticker', 'risk_score', 'risk_level', 'risk_explanation']].to_string(index=False)}

Explain:
- which companies are riskier
- why
- portfolio concentration risks

Do not provide buy/sell advice.
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

Generate a final investment research brief.

Investor profile:
- Risk profile: {profile}
- Horizon: {horizon}

Company data:
{df.to_string(index=False)}

Financial analysis:
{financial_summary}

Sentiment analysis:
{sentiment_summary}

Risk analysis:
{risk_summary}

Return:
1. Executive Summary
2. Company Comparison
3. Main Risks
4. Portfolio Fit
5. Key Questions Before Investing

Educational only.
No buy/sell recommendations.
"""

    return call_llm(prompt)