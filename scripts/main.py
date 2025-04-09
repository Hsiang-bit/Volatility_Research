import pandas as pd
import numpy as np
import fetch_data, volatility_profile, hourly_volatility, correlation_analysis, day_in_week_volatility
import os


start_time = "2024-04-01 00:00:00"
end_time = "2025-03-31 23:59:59"
binance_price_id = "BNBUSDT"
binance_price_freq = ["1w", "1d", "4h", "1h", "30m", "15m", "5m", "1m"]
price_url = r"D:\下載\Volatility_Research\data\raw"


# volatility_profile.volatility_profile_compare_and_only_holiday(
#     binance_price_freq, binance_price_id
# )

# volatility_profile.volatility_profile_by_close_price_magnitude(
#     binance_price_freq, binance_price_id
# )

hourly_volatility.volatility_hourly_intraday(binance_price_freq, binance_price_id)

# hourly_volatility.volatility_hourly_intraday_include_holiday(
#     binance_price_freq, binance_price_id
# )

# hourly_volatility.volatility_hourly_intraday_exclude_holiday(
#     binance_price_freq, binance_price_id
# )

# hourly_volatility.volatility_hourly_intraday_only_holiday(
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

# day_in_week_volatility.volatility_hourly_intraday_by_weekday(
#     binance_price_freq, binance_price_id
# )
