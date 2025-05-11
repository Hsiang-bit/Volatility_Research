import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def volatility_hourly_intraday_by_weekday(binance_price_freq, binance_price_id):
    binance_price_freq = [freq for freq in binance_price_freq if freq != "1w"]
    # 設定輸出資料夾
    output_folder = (
        r"D:\下載\Volatility_Research\processed\hourly_avg_volatility_by_weekday"
    )
    os.makedirs(output_folder, exist_ok=True)

    all_stats_list = []

    for freq in binance_price_freq:
        # 設定輸入檔案路徑
        input_file = rf"D:\下載\Volatility_Research\data\raw\{binance_price_id}_{freq}_price_data.csv"

        try:
            df = pd.read_csv(input_file)

            if {"datetime", "high", "low"}.issubset(df.columns):
                df["datetime"] = pd.to_datetime(df["datetime"])
                df["vol"] = df["high"] - df["low"]
                df["hour"] = df["datetime"].dt.hour
                df["weekday"] = df["datetime"].dt.dayofweek  # 0=Monday, ..., 6=Sunday

                # 計算每週幾、每小時的平均波動
                stats = (
                    df.groupby(["weekday", "hour"])["vol"]
                    .mean()
                    .reset_index()
                    .rename(columns={"vol": "avg_vol"})
                )
                stats["freq"] = freq  # 加上頻率標籤
                all_stats_list.append(stats)
            else:
                print(f"❌ {input_file} 缺少必要欄位")
        except Exception as e:
            print(f"❌ 錯誤發生在 {freq}: {e}")

    # 合併所有時間週期的資料
    if all_stats_list:
        combined_df = pd.concat(all_stats_list, ignore_index=True)

        # 儲存成 CSV
        output_file = (
            rf"{output_folder}\{binance_price_id}_weekday_hourly_volatility.csv"
        )
        combined_df.to_csv(output_file, index=False)
        print(f"✅ 結果已儲存至: {output_file}")
    else:
        print("⚠️ 沒有可用的結果")

    # 將 weekday 轉為文字以利圖表閱讀
    weekday_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    combined_df["weekday_name"] = combined_df["weekday"].map(weekday_map)

    # 每個 freq 畫一張 heatmap
    for freq in combined_df["freq"].unique():
        df_freq = combined_df[combined_df["freq"] == freq]

        # 建立 pivot table
        heatmap_data = df_freq.pivot(
            index="weekday_name", columns="hour", values="avg_vol"
        )
        heatmap_data = heatmap_data.reindex(
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        )

        plt.figure(figsize=(30, 8))  # 放大圖片
        sns.heatmap(
            heatmap_data,
            cmap="YlOrRd",
            annot=True,
            fmt=".4f",  # 小數點兩位
            linewidths=0.5,
            annot_kws={"size": 10},
        )

        plt.title(
            f"{binance_price_id} - Avg Volatility by Weekday & Hour ({freq})",
            fontsize=16,
        )
        plt.xlabel("Hour", fontsize=12)
        plt.ylabel("Weekday", fontsize=12)

        # 儲存圖片
        heatmap_path = os.path.join(
            output_folder,
            f"{binance_price_id}_{freq}_weekday_hourly_volatility_heatmap.png",
        )
        plt.tight_layout()
        plt.savefig(heatmap_path)
        plt.close()

        print(f"📊 Heatmap 儲存為：{heatmap_path}")
