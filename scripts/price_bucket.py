import pandas as pd


# Utils Function
def get_price_bucket(price):
    """根據價格的位數與首位數字分類"""
    if price == 0 or pd.isna(price):
        return "invalid"
    price_str = str(int(price))
    return f"{len(price_str)}figures-{price_str[0]}"
