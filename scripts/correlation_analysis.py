import pandas as pd
import numpy as np
import os


# Correlation Analysis
def price_correlation(
    token1,
    token2,
    timeframes,
    start_time,
    end_time,
    data_dir,
    export_csv=True,
    output_dir="D:\下載\Volatility_Research\processed\correlation_reports",
):
    """
    計算兩個 Token 在多個 timeframe 下的報酬率 (ret) 相關係數
    """
    correlation_results = {}

    for tf in timeframes:
        file1 = os.path.join(data_dir, f"{token1}_{tf}_Binance_price_data.csv")
        file2 = os.path.join(data_dir, f"{token2}_{tf}_Binance_price_data.csv")

        try:
            df1 = pd.read_csv(file1, parse_dates=["datetime"])
            df2 = pd.read_csv(file2, parse_dates=["datetime"])

            # 篩選指定時間區間
            df1 = df1[
                (df1["datetime"] >= pd.to_datetime(start_time))
                & (df1["datetime"] <= pd.to_datetime(end_time))
            ]
            df2 = df2[
                (df2["datetime"] >= pd.to_datetime(start_time))
                & (df2["datetime"] <= pd.to_datetime(end_time))
            ]

            # 計算報酬率
            df1["ret"] = df1["close"].pct_change()
            df2["ret"] = df2["close"].pct_change()

            # 合併資料（對齊 datetime）
            merged = pd.merge(
                df1[["datetime", "ret"]],
                df2[["datetime", "ret"]],
                on="datetime",
                suffixes=(f"_{token1}", f"_{token2}"),
            )

            # 排除 NA
            merged = merged.dropna()

            if merged.empty:
                correlation_results[tf] = None
                print(f"⚠️ {tf}: 合併後資料為空")
                continue

            # 計算報酬率相關係數
            corr = merged[f"ret_{token1}"].corr(merged[f"ret_{token2}"])
            correlation_results[tf] = round(corr * 100, 2)

        except Exception as e:
            print(f"❌ 無法處理 {tf} 的資料: {e}")
            correlation_results[tf] = None

    # 輸出成 CSV
    if export_csv:
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(
            output_dir,
            f"price_correlation_{token1}_{token2}.csv",
        )

        result_df = pd.DataFrame(
            list(correlation_results.items()), columns=["timeframe", "correlation (%)"]
        )
        result_df.to_csv(output_file, index=False)
        print(f"✅ 相關係數報告已輸出至：{output_file}")

    return correlation_results


def volatility_correlation(
    token1,
    token2,
    timeframes,
    start_time,
    end_time,
    data_dir,
    export_csv=True,
    output_dir="D:\下載\Volatility_Research\processed\correlation_reports",
):
    """
    計算兩個 Token 在多個 timeframe 下的波動度相關係數
    """
    correlation_results = {}

    for tf in timeframes:
        # 組成檔案路徑
        file1 = os.path.join(data_dir, f"{token1}_{tf}_Binance_price_data.csv")
        file2 = os.path.join(data_dir, f"{token2}_{tf}_Binance_price_data.csv")

        try:
            # 讀取資料
            df1 = pd.read_csv(file1, parse_dates=["datetime"])
            df2 = pd.read_csv(file2, parse_dates=["datetime"])

            # 計算波動度
            df1["vol"] = df1["high"] - df1["low"]
            df2["vol"] = df2["high"] - df2["low"]

            # 篩選指定期間
            df1 = df1[
                (df1["datetime"] >= pd.to_datetime(start_time))
                & (df1["datetime"] <= pd.to_datetime(end_time))
            ]
            df2 = df2[
                (df2["datetime"] >= pd.to_datetime(start_time))
                & (df2["datetime"] <= pd.to_datetime(end_time))
            ]

            # 確保時間對齊
            merged = pd.merge(
                df1[["datetime", "vol"]],
                df2[["datetime", "vol"]],
                on="datetime",
                suffixes=(f"_{token1}", f"_{token2}"),
            )

            if merged.empty:
                correlation_results[tf] = None
                print(f"⚠️ {tf}: 合併後資料為空")
                continue

            # 計算皮爾森相關係數
            corr = merged[f"vol_{token1}"].corr(merged[f"vol_{token2}"])
            correlation_results[tf] = round(corr * 100, 2)

        except Exception as e:
            print(f"❌ 無法處理 {tf} 的資料: {e}")
            correlation_results[tf] = None

    # 輸出成 CSV
    if export_csv:
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(
            output_dir,
            f"volatility_correlation_{token1}_{token2}.csv",
        )

        result_df = pd.DataFrame(
            list(correlation_results.items()), columns=["timeframe", "correlation (%)"]
        )
        result_df.to_csv(output_file, index=False)
        print(f"✅ 相關係數報告已輸出至：{output_file}")

    return correlation_results
