import pandas as pd
import numpy as np
import fetch_data, volatility_profile, hourly_volatility, correlation_analysis, day_in_week_volatility
import os


start_time = "2025-05-01 00:00:00"
end_time = "2025-05-09 23:59:59"
binance_price_id = "ETHUSDT"
binance_price_ids = ["ETHUSDT", "ETCUSDT", "BTCUSDT", "BCHUSDT", "SOLUSDT"]
binance_price_freq = ["1d", "4h", "1h", "30m", "15m", "5m", "1m"]
price_url = r"D:\下載\Volatility_Research\data\raw"

fetch_data.fetch_data(
    binance_price_freq, binance_price_ids, start_time, end_time, price_url
)

# volatility_profile.volatility_profile_compare_and_only_holiday(
#     binance_price_freq, binance_price_id
# )

volatility_profile.volatility_profile_multi_assets(
    binance_price_freq, binance_price_ids, start_time, end_time
)

# volatility_profile.volatility_profile_by_close_price_magnitude(
#     binance_price_freq, binance_price_id
# )

# hourly_volatility.volatility_hourly_intraday(binance_price_freq, binance_price_id)

# day_in_week_volatility.volatility_hourly_intraday_by_weekday(
#     binance_price_freq, binance_price_id
# )

# hourly_volatility.volatility_hourly_by_price_level_include_holiday(
#     binance_price_freq, binance_price_id
# )

# correlation_analysis.price_correlation(
#     token1="ETHUSDT",
#     token2="ETCUSDT",
#     timeframes=binance_price_freq,
#     start_time=start_time,
#     end_time=end_time,
#     data_dir=price_url,
# )

# correlation_analysis.volatility_correlation(
#     token1="ETHUSDT",
#     token2="ETCUSDT",
#     timeframes=binance_price_freq,
#     start_time=start_time,
#     end_time=end_time,
#     data_dir=price_url,
# )
