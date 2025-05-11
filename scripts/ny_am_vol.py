import pandas as pd
import matplotlib.pyplot as plt
import os
from pandas.tseries.holiday import USFederalHolidayCalendar


def plot_multiscale_intraday_volatility(binance_price_id):
    input_folder = r"D:\下載\Volatility_Research\data\raw"
    freq_list = ["15m", "5m", "1m"]  # 不含 1h
    cal = USFederalHolidayCalendar()
    us_holidays = cal.holidays(start="2020-01-01", end="2030-12-31")

    fig, axs = plt.subplots(len(freq_list), 1, figsize=(18, 15), sharex=False)
    fig.subplots_adjust(hspace=0.6)

    for i, freq in enumerate(freq_list):
        file_path = os.path.join(
            input_folder, f"{binance_price_id}_{freq}_price_data.csv"
        )
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["vol"] = df["high"] - df["low"]
        df["date"] = df["datetime"].dt.normalize()

        # Filter: Weekdays, exclude US holidays, time between 12:00–16:00
        df = df[
            (df["datetime"].dt.weekday < 5)
            & (~df["date"].isin(us_holidays))
            & (df["datetime"].dt.hour >= 12)
            & (df["datetime"].dt.hour < 16)
        ]

        if df.empty:
            print(f"⚠️ No data for {freq} after filtering.")
            continue

        df["time_str"] = df["datetime"].dt.strftime("%H:%M")
        avg_vol = df.groupby("time_str")["vol"].mean().reset_index()

        axs[i].plot(
            avg_vol["time_str"],
            avg_vol["vol"],
            label=f"{freq} Avg Volatility",
            color="tab:blue",
        )
        axs[i].set_title(
            f"{freq} Average Volatility (Weekdays 12:00–16:00, excl. US holidays)",
            fontsize=14,
        )
        axs[i].set_xlabel("Time of Day", fontsize=12)
        axs[i].set_ylabel("Avg Volatility", fontsize=12)

        if freq in ["1m", "5m"]:
            axs[i].set_xticks(avg_vol["time_str"][::10])  # 顯示每10分鐘的時間點

        axs[i].tick_params(axis="x", rotation=90)
        axs[i].grid(True)

    plt.suptitle(f"{binance_price_id} Intraday Volatility (15m / 5m / 1m)", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


plot_multiscale_intraday_volatility("ETHUSDT")
