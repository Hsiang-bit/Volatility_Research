import pandas as pd
import numpy as np
from fetch_data import BinancePriceData
import os
from pandas.tseries.holiday import USFederalHolidayCalendar
import price_bucket


# Volatility Profile
def volatiluty_profile_indclude_holiday(binance_price_freq, binance_price_id):
    # 設定輸出資料夾
    output_folder = r"D:\下載\Volatility_Research\processed\statistics_data"

    # 確保資料夾存在
    os.makedirs(output_folder, exist_ok=True)

    # 建立一個空的 DataFrame 來存放所有統計數據
    all_stats_list = []

    for freq in binance_price_freq:
        # 設定輸入檔案路徑
        input_file = rf"D:\下載\Volatility_Research\data\raw\{binance_price_id}_{freq}_Binance_price_data.csv"

        # 讀取 CSV 檔案
        df = pd.read_csv(input_file)

        # 確保 'high' 和 'low' 欄位存在
        if "high" in df.columns and "low" in df.columns:
            df["vol"] = df["high"] - df["low"]

            # 計算統計數據
            stats = df["vol"].describe()

            # 轉換成 DataFrame，並加上 'freq' 欄位
            stats_df = stats.to_frame(name="volatility")
            stats_df["freq"] = freq  # 新增頻率欄位
            stats_df.reset_index(inplace=True)  # 讓統計指標變成欄位名稱

            # 加入到列表
            all_stats_list.append(stats_df)
        else:
            print(f"❌ {input_file} 缺少 'high' 或 'low' 欄位，無法計算 vol")

    # 合併所有數據
    if all_stats_list:
        all_stats_df = pd.concat(all_stats_list, ignore_index=True)

        # 設定輸出檔案路徑
        output_file = rf"{output_folder}\{binance_price_id}_volatility_stats.csv"

        # 儲存統計數據為 CSV
        all_stats_df.to_csv(output_file, index=False)
        print(f"✅ 已儲存合併後的統計數據: {output_file}")
    else:
        print("⚠️ 沒有可用的統計數據，未生成 CSV 檔案")


def volatility_profile_exclude_us_holiday(binance_price_freq, binance_price_id):
    # 設定輸出資料夾
    output_folder = (
        r"D:\下載\Volatility_Research\processed\statistics_data_exclude_holidays"
    )
    os.makedirs(output_folder, exist_ok=True)

    # 初始化美國聯邦假日（抓過去與未來幾年的區間）
    cal = USFederalHolidayCalendar()
    holiday_range = cal.holidays(start="2020-01-01", end="2030-12-31")

    all_stats_list = []

    for freq in binance_price_freq:
        input_file = rf"D:\下載\Volatility_Research\data\raw\{binance_price_id}_{freq}_Binance_price_data.csv"
        df = pd.read_csv(input_file)

        if "datetime" in df.columns and "high" in df.columns and "low" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["vol"] = df["high"] - df["low"]

            # 移除週末與美國假日
            df = df[
                (df["datetime"].dt.weekday < 5)  # 週一到週五
                & (~df["datetime"].dt.normalize().isin(holiday_range))  # 非假日
            ]

            if not df.empty:
                stats = df["vol"].describe()
                stats_df = stats.to_frame(name="volatility")
                stats_df["freq"] = freq
                stats_df.reset_index(inplace=True)
                all_stats_list.append(stats_df)
            else:
                print(f"⚠️ 資料 {freq} 在排除假日後為空")
        else:
            print(f"❌ {input_file} 缺少必要欄位")

    if all_stats_list:
        all_stats_df = pd.concat(all_stats_list, ignore_index=True)
        output_file = rf"{output_folder}\{binance_price_id}_volatility_stats_exclude_us_holiday.csv"
        all_stats_df.to_csv(output_file, index=False)
        print(f"✅ 已儲存統計（排除美國國定假日）: {output_file}")
    else:
        print("⚠️ 無可用資料，未產生輸出")


def volatility_profile_only_us_holiday(binance_price_freq, binance_price_id):
    # 設定輸出資料夾
    output_folder = (
        r"D:\下載\Volatility_Research\processed\statistics_data_only_holidays"
    )
    os.makedirs(output_folder, exist_ok=True)

    # 初始化美國聯邦假日（抓過去與未來幾年的區間）
    cal = USFederalHolidayCalendar()
    holiday_range = cal.holidays(start="2020-01-01", end="2030-12-31")

    all_stats_list = []

    for freq in binance_price_freq:
        input_file = rf"D:\下載\Volatility_Research\data\raw\{binance_price_id}_{freq}_Binance_price_data.csv"
        df = pd.read_csv(input_file)

        if "datetime" in df.columns and "high" in df.columns and "low" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["vol"] = df["high"] - df["low"]

            # 只保留週末與美國國定假日
            df = df[
                (df["datetime"].dt.weekday >= 5)  # 週六週日
                | (df["datetime"].dt.normalize().isin(holiday_range))  # 國定假日
            ]

            if not df.empty:
                stats = df["vol"].describe()
                stats_df = stats.to_frame(name="volatility")
                stats_df["freq"] = freq
                stats_df.reset_index(inplace=True)
                all_stats_list.append(stats_df)
            else:
                print(f"⚠️ 資料 {freq} 在僅保留假日後為空")
        else:
            print(f"❌ {input_file} 缺少必要欄位")

    if all_stats_list:
        all_stats_df = pd.concat(all_stats_list, ignore_index=True)
        output_file = (
            rf"{output_folder}\{binance_price_id}_volatility_stats_only_us_holiday.csv"
        )
        all_stats_df.to_csv(output_file, index=False)
        print(f"✅ 已儲存統計（僅假日）: {output_file}")
    else:
        print("⚠️ 無可用資料，未產生輸出")


def volatility_profile_by_close_price_magnitude(binance_price_freq, binance_price_id):
    output_folder = r"D:\下載\Volatility_Research\processed\statistics_data_by_close_price_magnitude"
    os.makedirs(output_folder, exist_ok=True)

    all_stats_list = []

    for freq in binance_price_freq:
        input_file = rf"D:\下載\Volatility_Research\data\raw\{binance_price_id}_{freq}_Binance_price_data.csv"
        df = pd.read_csv(input_file)

        if {"close", "high", "low"}.issubset(df.columns):
            df["vol"] = df["high"] - df["low"]
            df["price_level"] = df["close"].apply(price_bucket.get_price_bucket)

            # 針對每個分類做統計
            for level in df["price_level"].unique():
                if level == "invalid":
                    continue
                sub_df = df[df["price_level"] == level]
                stats = sub_df["vol"].describe()
                stats_df = stats.to_frame(name="volatility")
                stats_df["freq"] = freq
                stats_df["price_level"] = level
                stats_df.reset_index(inplace=True)
                all_stats_list.append(stats_df)
        else:
            print(f"❌ {input_file} 缺少 'close', 'high', 或 'low' 欄位，無法計算 vol")

    if all_stats_list:
        all_stats_df = pd.concat(all_stats_list, ignore_index=True)
        output_file = (
            rf"{output_folder}\{binance_price_id}_volatility_by_close_price_level.csv"
        )
        all_stats_df.to_csv(output_file, index=False)
        print(f"✅ 已儲存: {output_file}")
    else:
        print("⚠️ 無可用資料，未產生統計結果")
