import pandas as pd
import numpy as np
from fetch_data import BinancePriceData
import os

start_time = "2024-04-01 00:00:00"
end_time = "2025-03-31 23:59:59"
binance_price_id = "SOLUSDT"
binance_price_freq = ["1w", "1d", "4h", "1h", "30m", "15m", "5m", "1m"]
price_url = "D:\下載\Zone wallet\zonewallet-quant-research\projects\cex_savings_product\price_data"
