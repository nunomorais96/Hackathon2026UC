import yfinance as yf
import pandas as pd


def get_stock_data(tickers):
    results = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1y")

            if hist.empty:
                volatility = None
            else:
                hist["daily_return"] = hist["Close"].pct_change()
                volatility = hist["daily_return"].std() * (252 ** 0.5)

            results.append({
                "ticker": ticker.upper(),
                "company_name": info.get("longName", ticker.upper()),
                "sector": info.get("sector", "N/A"),
                "current_price": info.get("currentPrice"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "profit_margin": info.get("profitMargins"),
                "revenue_growth": info.get("revenueGrowth"),
                "debt_to_equity": info.get("debtToEquity"),
                "volatility": volatility,
            })

        except Exception as e:
            results.append({
                "ticker": ticker.upper(),
                "company_name": "Error",
                "sector": "N/A",
                "current_price": None,
                "market_cap": None,
                "pe_ratio": None,
                "forward_pe": None,
                "profit_margin": None,
                "revenue_growth": None,
                "debt_to_equity": None,
                "volatility": None,
                "error": str(e),
            })

    return pd.DataFrame(results)