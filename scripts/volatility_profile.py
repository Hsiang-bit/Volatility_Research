import pandas as pd
import numpy as np
from fetch_data import BinancePriceData
import os
from pandas.tseries.holiday import USFederalHolidayCalendar
import price_bucket
import matplotlib.pyplot as plt
import seaborn as sns


# Volatility Profile
def volatility_profile_by_close_price_magnitude(binance_price_freq, binance_price_id):
    output_folder = r"D:\下載\Volatility_Research\processed\statistics_data_by_close_price_magnitude"
    os.makedirs(output_folder, exist_ok=True)

    all_stats_list = []

    for freq in binance_price_freq:
        input_file = rf"D:\下載\Volatility_Research\data\raw\{binance_price_id}_{freq}_Binance_price_data.csv"

        try:
            df = pd.read_csv(input_file)

            # 檢查必要欄位是否存在
            if {"close", "high", "low"}.issubset(df.columns):
                df["vol"] = df["high"] - df["low"]  # 計算波動度
                df["price_level"] = df["close"].apply(
                    price_bucket.get_price_bucket
                )  # 使用價格分桶

                # 針對每個價格區間進行統計
                for level in df["price_level"].unique():
                    if level == "invalid":
                        continue  # 跳過無效區間
                    sub_df = df[df["price_level"] == level]
                    stats = sub_df["vol"].describe()

                    # 將統計結果轉換為 DataFrame 格式
                    stats_df = stats.to_frame(name="volatility")
                    stats_df["freq"] = freq
                    stats_df["price_level"] = level
                    stats_df.reset_index(inplace=True)

                    # 添加到統計結果列表
                    all_stats_list.append(stats_df)
            else:
                print(
                    f"❌ {input_file} 缺少 'close', 'high', 或 'low' 欄位，無法計算 vol"
                )

        except Exception as e:
            print(f"❌ 處理 {input_file} 時出錯: {e}")

    # 儲存統計結果至 CSV
    if all_stats_list:
        all_stats_df = pd.concat(all_stats_list, ignore_index=True)
        output_file = (
            rf"{output_folder}\{binance_price_id}_volatility_by_close_price_level.csv"
        )
        all_stats_df.to_csv(output_file, index=False)
        print(f"✅ 已儲存: {output_file}")
    else:
        print("⚠️ 無可用資料，未產生統計結果")


def volatility_profile_compare_and_only_holiday(binance_price_freq, binance_price_id):
    output_folder = r"D:\下載\Volatility_Research\processed\statistics_data_compare"
    os.makedirs(output_folder, exist_ok=True)

    cal = USFederalHolidayCalendar()
    holiday_range = cal.holidays(start="2020-01-01", end="2030-12-31")

    combined_stats = []

    for freq in binance_price_freq:
        input_file = rf"D:\下載\Volatility_Research\data\raw\{binance_price_id}_{freq}_Binance_price_data.csv"

        try:
            df = pd.read_csv(input_file)

            if (
                "datetime" in df.columns
                and "high" in df.columns
                and "low" in df.columns
            ):
                df["datetime"] = pd.to_datetime(df["datetime"])
                df["vol"] = df["high"] - df["low"]

                # 含假日統計
                stats_with = df["vol"].describe()
                for stat, value in stats_with.items():
                    combined_stats.append(
                        {
                            "statistic": stat,
                            "freq": freq,
                            "include_holiday": True,
                            "value": value,
                        }
                    )

                # 排除週末與美國假日
                df_filtered = df[
                    (df["datetime"].dt.weekday < 5)
                    & (~df["datetime"].dt.normalize().isin(holiday_range))
                ]

                if not df_filtered.empty:
                    stats_without = df_filtered["vol"].describe()
                    for stat, value in stats_without.items():
                        combined_stats.append(
                            {
                                "statistic": stat,
                                "freq": freq,
                                "include_holiday": False,
                                "value": value,
                            }
                        )
                else:
                    print(f"⚠️ 資料 {freq} 在排除假日後為空")

                # 只保留週末與美國國定假日
                df_only_holidays = df[
                    (df["datetime"].dt.weekday >= 5)  # 週六週日
                    | (df["datetime"].dt.normalize().isin(holiday_range))  # 國定假日
                ]

                if not df_only_holidays.empty:
                    stats_only_holidays = df_only_holidays["vol"].describe()
                    for stat, value in stats_only_holidays.items():
                        combined_stats.append(
                            {
                                "statistic": stat,
                                "freq": freq,
                                "include_holiday": "Only Holidays",
                                "value": value,
                            }
                        )
                else:
                    print(f"⚠️ 資料 {freq} 在僅保留假日後為空")
            else:
                print(f"❌ {input_file} 缺少必要欄位")

        except Exception as e:
            print(f"❌ 處理 {input_file} 時出錯: {e}")

    # 轉成 DataFrame 並輸出 CSV
    if combined_stats:
        stats_df = pd.DataFrame(combined_stats)
        stats_df = stats_df[["statistic", "freq", "include_holiday", "value"]]

        output_file = (
            rf"{output_folder}\{binance_price_id}_volatility_stats_combined.csv"
        )
        stats_df.to_csv(output_file, index=False)
        print(f"✅ 已輸出統計資料到：{output_file}")

    else:
        print("⚠️ 沒有可用統計數據，未輸出任何檔案")
