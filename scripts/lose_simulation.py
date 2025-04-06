import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 模擬參數
n_trials = 10_000_000
success_rate = 0.5

# 蒙地卡羅模擬：0 表失敗，1 表成功
results = np.random.choice([0, 1], size=n_trials, p=[1 - success_rate, success_rate])

# 統計連續失敗次數
streaks = []
current_streak = 0

for result in results:
    if result == 0:
        current_streak += 1
    elif current_streak > 0:
        streaks.append(current_streak)
        current_streak = 0

# 若最後一段為失敗，記錄
if current_streak > 0:
    streaks.append(current_streak)

# 建立 DataFrame 並統計每種連敗次數出現次數
streak_series = pd.Series(streaks)
distribution = streak_series.value_counts().sort_index()
distribution_df = distribution.reset_index()
distribution_df.columns = ["failure_streak", "count"]

# 儲存為 CSV
distribution_df.to_csv(
    r"D:\下載\Volatility_Research\processed\failure_streak_distribution\failure_streak_distribution.csv",
    index=False,
)
print("✅ 模擬結果已儲存為 failure_streak_distribution.csv")

# 畫出分布圖
plt.figure(figsize=(12, 6))
plt.bar(
    distribution_df["failure_streak"],
    distribution_df["count"],
    color="skyblue",
    edgecolor="black",
)
plt.xlabel("Consecutive Failures")
plt.ylabel("Count")
plt.title("Distribution of Consecutive Failures in 1,000,000 Trials (50% Success Rate)")
plt.xticks(distribution_df["failure_streak"])
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("failure_streak_distribution.png")
plt.show()

# ✨ 統計分析：連敗分位數
max_streak = streak_series.max()
mean_streak = streak_series.mean()
median_streak = streak_series.median()
p95 = streak_series.quantile(0.95)
p99 = streak_series.quantile(0.99)
p999 = streak_series.quantile(0.999)

# 顯示統計資訊
stats_summary = f"""
🔍 Consecutive Failure Streak Statistics (out of {n_trials} trials):
------------------------------------------------------
📈 Max Streak      : {max_streak}
📊 Mean Streak     : {mean_streak:.2f}
🔸 Median Streak   : {median_streak}
🔹 95th Percentile : {p95:.2f}
🔹 99th Percentile : {p99:.2f}
🔹 99.9th Percentile: {p999:.2f}
"""

print(stats_summary)
