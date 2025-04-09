import pandas as pd
import numpy as np
import os
from pandas.tseries.holiday import USFederalHolidayCalendar


class BinancePriceData:
    """
    datetime	| datetime64[ns, UTC] |	開盤時間，UTC+0 時區的 datetime 格式，作為 DataFrame 的索引。\n
    open	    | float |	                開盤價，數據型態為浮點數。\n
    high	    | float |	                最高價，數據型態為浮點數。\n
    low	        | float |	                最低價，數據型態為浮點數。\n
    close	    | float |	                收盤價，數據型態為浮點數。\n
    volume	    | float |	                交易量，數據型態為浮點數。\n
    close time	| datetime64[ns, UTC] |	收盤時間，UTC+0 時區的 datetime 格式。
    """

    def __init__(
        self,
        price_id: str,
        freq: str,
        start_date: str,
        end_date: str,
        local_folder: str,
        base_url="https://data-api.binance.vision",
    ):
        self.symbol = price_id
        self.freq = TimeFrameConvert.convert_interval(freq)
        self.spot_client = Client(base_url=base_url)

        # 解析 start_date 和 end_date 為 datetime 對象
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S") + timedelta(
            days=1
        )

        # 設置本地檔案路徑
        self.local_folder = local_folder
        os.makedirs(self.local_folder, exist_ok=True)  # 自動建立資料夾（如果不存在）
        self.local_file = os.path.join(
            self.local_folder, f"{self.symbol}_{self.freq}_Binance_price_data.csv"
        )

    def get_klines(self, freq, start_date, end_date):
        klines = []
        while start_date < end_date:
            temp_end = min(
                start_date + 500 * 60 * 60 * 1000, end_date + 24 * 60 * 60 * 1000
            )
            temp_klines = self.spot_client.klines(
                self.symbol, freq, startTime=start_date, endTime=temp_end
            )
            if not temp_klines:
                break
            klines.extend(temp_klines)
            start_date = temp_klines[-1][
                6
            ]  # 更新下一輪請求的開始時間為上次返回的最後一根 Kline 的結束時間
            start_date += 1  # 避免重疊，+1 毫秒
        return klines

    def klines_to_dataframe(self, klines):
        df = pd.DataFrame(
            klines,
            columns=[
                "Open time",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Close time",
                "Quote asset volume",
                "Number of trades",
                "Taker buy base asset volume",
                "Taker buy quote asset volume",
                "Ignore",
            ],
        )
        df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
        df["Close time"] = pd.to_datetime(df["Close time"], unit="ms")

        return df

    def get_data(self):
        # 將 start_date 和 end_date 轉換為毫秒
        start_time = int(self.start_date.timestamp() * 1000)
        end_time = int(self.end_date.timestamp() * 1000)

        # 檢查本地檔案是否存在，若存在則讀取，否則重新抓取資料
        if os.path.exists(self.local_file):
            df_local = pd.read_csv(
                self.local_file, index_col="datetime", parse_dates=True
            )

            # 檢查時間範圍是否符合要求
            if (
                not df_local.empty
                and df_local.index.min() <= self.start_date + timedelta(days=1)
                and df_local.index.max() >= self.end_date - timedelta(days=1)
            ):
                print(f"從本地檔案讀取資料: {self.local_file}")
                df_local = df_local.loc[self.start_date : self.end_date]
                return df_local
            else:
                print(f"本地檔案資料不完整，刪除檔案並重新抓取: {self.local_file}")
                os.remove(self.local_file)

        # 如果檔案不存在或資料不完整，重新抓取資料
        print("正在從 API 抓取資料...")
        klines = self.get_klines(self.freq, start_time, end_time)
        df = self.klines_to_dataframe(klines)

        # 精確過濾掉超出時間範圍的數據
        df = df[
            (df["Open time"] >= self.start_date) & (df["Close time"] <= self.end_date)
        ]

        # 重命名列
        df.rename(
            columns={
                "Open time": "datetime",
                "Close time": "close time",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            },
            inplace=True,
        )

        # 把 datetime 改成 index 並升序排列
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)

        # 保存資料到本地檔案
        df.to_csv(self.local_file, index=True)
        print(f"資料已儲存至 {self.local_file}")

        return df


# Data Download
def fetch_data(binance_price_freq, binance_price_id, start_time, end_time, price_url):
    for freq in binance_price_freq:
        binance_price = BinancePriceData(
            binance_price_id, freq, start_time, end_time, price_url
        ).get_data()
