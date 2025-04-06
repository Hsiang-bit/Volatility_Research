import pandas as pd
import numpy as np
from fetch_data import BinancePriceData
import price_bucket
import os
from pandas.tseries.holiday import USFederalHolidayCalendar


def volatility_hourly_intraday_include_holiday(binance_price_freq, binance_price_id):
    # 設定輸出資料夾
    output_folder = r"D:\下載\Volatility_Research\processed\hourly_avg_volatility"
    os.makedirs(output_folder, exist_ok=True)

    # 用來存放所有時間週期的資料
    all_hourly_avg_list = []

    for freq in binance_price_freq:
        # 設定輸入檔案路徑
        input_file = rf"D:\下載\Volatility_Research\data\raw\{binance_price_id}_{freq}_Binance_price_data.csv"

        # 讀取 CSV 檔案
        df = pd.read_csv(input_file)

        # 確保 'datetime', 'high', 'low' 欄位存在
        if "datetime" in df.columns and "high" in df.columns and "low" in df.columns:
            df["vol"] = df["high"] - df["low"]
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["hour"] = df["datetime"].dt.hour

            # 計算每小時平均 vol
            hourly_avg_vol = df.groupby("hour")["vol"].mean().reset_index()
            hourly_avg_vol.rename(columns={"vol": "avg_vol"}, inplace=True)
            hourly_avg_vol["freq"] = freq  # 加上 freq 欄位

            # 加入到總列表中
            all_hourly_avg_list.append(hourly_avg_vol)
        else:
            print(f"❌ {input_file} 缺少必要欄位，無法計算 vol")

    # 合併所有時間週期的資料
    if all_hourly_avg_list:
        combined_df = pd.concat(all_hourly_avg_list, ignore_index=True)

        # 輸出合併後的檔案
        merged_output_file = (
            rf"{output_folder}\{binance_price_id}merged_hourly_avg_volatility.csv"
        )
        combined_df.to_csv(merged_output_file, index=False)
        print(f"✅ 所有資料已合併並儲存至: {merged_output_file}")
    else:
        print("⚠️ 沒有資料被成功處理，未生成合併檔案")


def volatility_hourly_intraday_exclude_holiday(binance_price_freq, binance_price_id):
    output_folder = (
        r"D:\下載\Volatility_Research\processed\hourly_avg_volatility_exclude_holiday"
    )
    os.makedirs(output_folder, exist_ok=True)

    cal = USFederalHolidayCalendar()
    us_holidays = cal.holidays(start="2020-01-01", end="2030-12-31")

    # 儲存所有 freq 的結果
    all_hourly_vol_list = []

    for freq in binance_price_freq:
        input_file = rf"D:\下載\Volatility_Research\data\raw\{binance_price_id}_{freq}_Binance_price_data.csv"
        df = pd.read_csv(input_file)

        if "datetime" in df.columns and "high" in df.columns and "low" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["vol"] = df["high"] - df["low"]

            # 移除週末與美國國定假日
            df = df[
                (df["datetime"].dt.weekday < 5)
                & (~df["datetime"].dt.normalize().isin(us_holidays))
            ]

            if not df.empty:
                df["hour"] = df["datetime"].dt.hour
                hourly_avg_vol = df.groupby("hour")["vol"].mean().reset_index()
                hourly_avg_vol["freq"] = freq
                all_hourly_vol_list.append(hourly_avg_vol)
            else:
                print(f"⚠️ {freq} 篩選後資料為空，已略過")
        else:
            print(f"❌ {input_file} 缺少必要欄位")

    # 合併成單一 DataFrame 並輸出
    if all_hourly_vol_list:
        combined_df = pd.concat(all_hourly_vol_list, ignore_index=True)
        output_file = rf"{output_folder}\{binance_price_id}_hourly_avg_volatility_exclude_us_holiday.csv"
        combined_df.to_csv(output_file, index=False)
        print(f"✅ 已儲存彙整後的每小時平均波動度資料: {output_file}")
    else:
        print("⚠️ 沒有可用資料，未產生 CSV")


def volatility_hourly_intraday_only_holiday(binance_price_freq, binance_price_id):
    output_folder = (
        r"D:\下載\Volatility_Research\processed\hourly_avg_volatility_only_holiday"
    )
    os.makedirs(output_folder, exist_ok=True)

    cal = USFederalHolidayCalendar()
    us_holidays = cal.holidays(start="2020-01-01", end="2030-12-31")

    all_holiday_hourly_vol = []

    for freq in binance_price_freq:
        input_file = rf"D:\下載\Volatility_Research\data\raw\{binance_price_id}_{freq}_Binance_price_data.csv"
        df = pd.read_csv(input_file)

        if "datetime" in df.columns and "high" in df.columns and "low" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["vol"] = df["high"] - df["low"]

            # 篩選週末或美國假日（注意：週末是 weekday >= 5）
            df = df[
                (df["datetime"].dt.weekday >= 5)
                | (df["datetime"].dt.normalize().isin(us_holidays))
            ]

            if not df.empty:
                df["hour"] = df["datetime"].dt.hour
                hourly_avg_vol = df.groupby("hour")["vol"].mean().reset_index()
                hourly_avg_vol["freq"] = freq
                all_holiday_hourly_vol.append(hourly_avg_vol)
            else:
                print(f"⚠️ {freq} 沒有符合條件的假日資料")
        else:
            print(f"❌ {input_file} 缺少必要欄位")

    # 合併結果並輸出
    if all_holiday_hourly_vol:
        holiday_df = pd.concat(all_holiday_hourly_vol, ignore_index=True)
        output_file = rf"{output_folder}\{binance_price_id}_hourly_avg_volatility_only_us_holiday.csv"
        holiday_df.to_csv(output_file, index=False)
        print(f"✅ 已儲存僅假日的每小時波動度統計: {output_file}")
    else:
        print("⚠️ 沒有任何假日資料，未產生輸出")


def volatility_hourly_by_price_level_include_holiday(
    binance_price_freq, binance_price_id
):
    output_folder = (
        r"D:\下載\Volatility_Research\processed\hourly_avg_volatility_by_price_level"
    )
    os.makedirs(output_folder, exist_ok=True)

    all_hourly_avg_list = []

    for freq in binance_price_freq:
        input_file = rf"D:\下載\Volatility_Research\data\raw\{binance_price_id}_{freq}_Binance_price_data.csv"
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
