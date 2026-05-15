import yfinance as yf


def search_companies(query, max_results=5):
    """
    Search companies/stocks using Yahoo Finance via yfinance.
    Returns a list of possible matches.
    """
    try:
        search = yf.Search(query, max_results=max_results)
        quotes = search.quotes or []

        results = []

        for item in quotes:
            symbol = item.get("symbol")
            name = item.get("shortname") or item.get("longname") or item.get("name")
            exchange = item.get("exchange")
            quote_type = item.get("quoteType")

            if symbol and quote_type in ["EQUITY", "ETF"]:
                results.append({
                    "symbol": symbol,
                    "name": name,
                    "exchange": exchange,
                    "quote_type": quote_type
                })

        return results

    except Exception as e:
        return [{
            "symbol": None,
            "name": f"Search error: {str(e)}",
            "exchange": None,
            "quote_type": None
        }]


def resolve_company_to_ticker(company_name):
    """
    Returns the best ticker match for a company name.
    """
    results = search_companies(company_name, max_results=5)

    valid_results = [
        r for r in results
        if r.get("symbol") is not None
    ]

    if not valid_results:
        return None

    return valid_results[0]["symbol"]


def resolve_companies_to_tickers(company_names):
    """
    Converts a list of company names into ticker symbols.
    """
    resolved = []

    for company in company_names:
        ticker = resolve_company_to_ticker(company.strip())

        if ticker:
            resolved.append({
                "input": company,
                "ticker": ticker
            })
        else:
            resolved.append({
                "input": company,
                "ticker": None
            })

    return resolved