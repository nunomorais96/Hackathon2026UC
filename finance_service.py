import yfinance as yf
import pandas as pd


def get_price_history(tickers, period="1y"):
    histories = {}

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)

            if hist.empty:
                continue

            frame = hist[["Close"]].copy()
            frame = frame.reset_index()
            frame = frame.rename(columns={"Date": "date", "Close": "close"})
            frame["ticker"] = ticker.upper()

            histories[ticker.upper()] = frame

        except Exception:
            continue

    return histories


def histories_to_long_df(histories):
    if not histories:
        return pd.DataFrame(columns=["date", "close", "ticker"])

    return pd.concat(histories.values(), ignore_index=True)


def normalize_histories(histories):
    normalized = {}

    for ticker, frame in histories.items():
        if frame.empty:
            continue

        base = frame["close"].iloc[0]

        if base == 0 or pd.isna(base):
            continue

        norm_frame = frame.copy()
        norm_frame["close"] = (norm_frame["close"] / base) * 100
        normalized[ticker] = norm_frame

    return normalized


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