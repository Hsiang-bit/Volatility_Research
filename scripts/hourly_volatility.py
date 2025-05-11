import pandas as pd
import numpy as np
from fetch_data import BinancePriceData
import price_bucket
import os
from pandas.tseries.holiday import USFederalHolidayCalendar


def volatility_hourly_intraday(binance_price_freq, binance_price_id):
    # 設定輸出資料夾
    output_folder = (
        r"D:\下載\Volatility_Research\processed\hourly_avg_volatility_combined"
    )
    os.makedirs(output_folder, exist_ok=True)

    # 初始化美國聯邦假日（抓過去與未來幾年的區間）
    cal = USFederalHolidayCalendar()
    us_holidays = cal.holidays(start="2020-01-01", end="2030-12-31")

    all_hourly_avg_list = []

    for freq in binance_price_freq:
        # 設定輸入檔案路徑
        input_file = rf"D:\下載\Volatility_Research\data\raw\{binance_price_id}_{freq}_price_data.csv"
        df = pd.read_csv(input_file)

        # 確保必要欄位存在
        if "datetime" in df.columns and "high" in df.columns and "low" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["vol"] = df["high"] - df["low"]
            df["hour"] = df["datetime"].dt.hour

            # 篩選條件: 包含假日、排除假日、僅假日
            for holiday_filter_type in ["include", "exclude", "only"]:
                filtered_df = df.copy()

                if holiday_filter_type == "exclude":
                    filtered_df = filtered_df[
                        (filtered_df["datetime"].dt.weekday < 5)  # 排除週末
                        & (
                            ~filtered_df["datetime"].dt.normalize().isin(us_holidays)
                        )  # 排除美國假日
                    ]
                elif holiday_filter_type == "only":
                    filtered_df = filtered_df[
                        (filtered_df["datetime"].dt.weekday >= 5)  # 只保留週末
                        | (
                            filtered_df["datetime"].dt.normalize().isin(us_holidays)
                        )  # 只保留美國假日
                    ]

                # 計算每小時平均波動度
                if not filtered_df.empty:
                    hourly_avg_vol = (
                        filtered_df.groupby("hour")["vol"].mean().reset_index()
                    )
                    hourly_avg_vol["freq"] = freq
                    hourly_avg_vol["holiday_filter_type"] = (
                        holiday_filter_type  # 加入篩選條件標識
                    )
                    all_hourly_avg_list.append(hourly_avg_vol)
                else:
                    print(f"⚠️ {freq} 篩選後 {holiday_filter_type} 資料為空，已略過")
        else:
            print(f"❌ {input_file} 缺少必要欄位，無法計算 vol")

    # 合併所有資料並儲存為單一 CSV
    if all_hourly_avg_list:
        combined_df = pd.concat(all_hourly_avg_list, ignore_index=True)
        output_file = (
            rf"{output_folder}\{binance_price_id}_hourly_avg_volatility_combined.csv"
        )
        combined_df.to_csv(output_file, index=False)
        print(f"✅ 所有資料已合併並儲存至: {output_file}")
    else:
        print("⚠️ 沒有資料被成功處理，未生成合併檔案")


def volatility_hourly_by_price_level_include_holiday(
    binance_price_freq, binance_price_id
):
    output_folder = (
        r"D:\下載\Volatility_Research\processed\hourly_avg_volatility_by_price_level"
    )
    os.makedirs(output_folder, exist_ok=True)

    all_hourly_avg_list = []

    for freq in binance_price_freq:
        input_file = rf"D:\下載\Volatility_Research\data\raw\{binance_price_id}_{freq}_price_data.csv"
        df = pd.read_csv(input_file)

        if {"datetime", "high", "low", "close"}.issubset(df.columns):
            df["vol"] = df["high"] - df["low"]
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["hour"] = df["datetime"].dt.hour
            df["price_level"] = df["close"].apply(price_bucket.get_price_bucket)

            # 移除無效價格分類
            df = df[df["price_level"] != "invalid"]

            # 按 hour 和 price_level 分群後計算 vol 平均
            hourly_avg_vol = (
                df.groupby(["hour", "price_level"])["vol"]
                .mean()
                .reset_index()
                .rename(columns={"vol": "avg_vol"})
            )
            hourly_avg_vol["freq"] = freq
            all_hourly_avg_list.append(hourly_avg_vol)
        else:
            print(f"❌ {input_file} 缺少必要欄位")

    if all_hourly_avg_list:
        combined_df = pd.concat(all_hourly_avg_list, ignore_index=True)
        output_file = (
            rf"{output_folder}\{binance_price_id}_hourly_avg_vol_by_price_level.csv"
        )
        combined_df.to_csv(output_file, index=False)
        print(f"✅ 所有資料已分類並儲存至: {output_file}")
    else:
        print("⚠️ 沒有資料被成功處理")
