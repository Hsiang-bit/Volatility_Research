# import numpy as np
# import matplotlib.pyplot as plt

# # 模擬參數
# initial_balance = 1000
# initial_bet_percent = 0.08
# min_bet_percent = 0.02
# max_bet_percent = 0.15 / 2
# reset_bet_percent = 0.08
# reset_threshold_gain = 0.45  # +45%獲利
# reset_threshold_loss = -0.25  # -25%虧損
# fixed_keep_percent = 0.25  # 每次保留25%
# win_rate = 0.5
# reward_risk_ratio = 3  # 3:1
# num_simulations = 10000
# num_trades = 16 * 3

# # 用來記錄每次模擬的資金曲線
# all_balances = []

# # 開始模擬
# for sim in range(num_simulations):
#     balance = initial_balance
#     current_bet_percent = initial_bet_percent
#     peak_balance = initial_balance
#     saved_balance = 0
#     balances = [balance + saved_balance]

#     for trade in range(num_trades):
#         bet_amount = balance * current_bet_percent

#         # 隨機判斷勝負
#         if np.random.rand() < win_rate:
#             # 勝利
#             gain = bet_amount * reward_risk_ratio
#             balance += gain
#             if balance + saved_balance >= initial_balance:  # 如果整體是獲利狀態
#                 current_bet_percent = min(current_bet_percent + 0.02, max_bet_percent)
#             else:
#                 current_bet_percent = min(reset_bet_percent, current_bet_percent)
#         else:
#             # 失敗
#             loss = bet_amount
#             balance -= loss
#             current_bet_percent = max(current_bet_percent - 0.02, min_bet_percent)

#         # 判斷是否需要保留資金或重置下注
#         total_balance = balance + saved_balance
#         gain_from_start = (total_balance - initial_balance) / initial_balance

#         if gain_from_start >= reset_threshold_gain:
#             # 獲利45%以上，保留25%
#             saved_amount = total_balance * fixed_keep_percent
#             saved_balance += saved_amount
#             balance = total_balance - saved_amount
#             current_bet_percent = reset_bet_percent

#         elif gain_from_start <= reset_threshold_loss:
#             # 虧損25%以上，重置下注
#             current_bet_percent = reset_bet_percent

#         balances.append(balance + saved_balance)

#     all_balances.append(balances)

# # 繪圖
# plt.figure(figsize=(16, 8))
# for balances in all_balances:
#     plt.plot(balances, color="purple", alpha=0.8)

# plt.title("Simulated Account Growth (5000 simulations)", fontsize=18)
# plt.xlabel("Trades", fontsize=14)
# plt.ylabel("Total Balance (Including Saved)", fontsize=14)
# plt.grid(True)
# plt.show()

# # 繪製資金分布 Histogram
# final_balances = [balances[-1] for balances in all_balances]

# plt.figure(figsize=(12, 6))
# plt.hist(final_balances, bins=100, color="skyblue", edgecolor="black")
# plt.title("Distribution of Final Balances (5000 simulations)", fontsize=18)
# plt.xlabel("Final Balance", fontsize=14)
# plt.ylabel("Frequency", fontsize=14)
# plt.grid(True)
# plt.show()

# # 把每次模擬的最終資金抓出來
# final_balances = [balances[-1] for balances in all_balances]

# # 計算中位數
# median_final_balance = np.median(final_balances)
# mean_final_balance = np.mean(final_balances)

# # 印出結果
# print(f"Median Final Balance after 1000 trades: ${median_final_balance:.2f}")
# print(f"Mean Final Balance after 1000 trades: ${mean_final_balance:.2f}")


# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import itertools


# # 模擬函數
# def simulate_trades(
#     initial_balance, win_rate, reward_risk_ratio, initial_bet_percentage, n_trades=1000
# ):
#     balance = initial_balance
#     saved = 0
#     bet_percentage = initial_bet_percentage
#     min_bet = 0.02
#     max_bet = 0.15
#     reset_bet = 0.08
#     trades = 0
#     peak_balance = balance

#     balances = [balance + saved]

#     while trades < n_trades:
#         # 下單金額
#         bet_amount = balance * bet_percentage

#         # 判斷勝負
#         if np.random.rand() < win_rate:
#             gain = bet_amount * reward_risk_ratio
#             balance += gain
#             bet_percentage = min(bet_percentage + 0.02, max_bet)
#             # 若當前總資金小於初始資金，不讓下注超過8%
#             if balance + saved < initial_balance:
#                 bet_percentage = min(bet_percentage, reset_bet)
#         else:
#             loss = bet_amount
#             balance -= loss
#             bet_percentage = max(bet_percentage - 0.02, min_bet)

#         trades += 1
#         total_balance = balance + saved
#         balances.append(total_balance)

#         # 獲利或虧損控制
#         if total_balance >= peak_balance * 1.45:
#             saved += balance * 0.25
#             balance *= 0.75
#             bet_percentage = reset_bet
#             peak_balance = balance + saved
#         elif total_balance <= peak_balance * 0.75:
#             bet_percentage = reset_bet
#             peak_balance = balance + saved

#     return balances[-1]


# # 設定參數範圍
# win_rates = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
# reward_risk_ratios = [1, 2, 3, 4]
# initial_bet_percentages = [0.02, 0.04, 0.06, 0.08, 0.10]

# # 存放結果
# results = {}

# # 跑所有參數組合
# for win_rate, rr_ratio, init_bet in itertools.product(
#     win_rates, reward_risk_ratios, initial_bet_percentages
# ):
#     final_balances = []
#     for _ in range(500):  # 每組參數模擬 500 次
#         final_balance = simulate_trades(
#             initial_balance=1000,
#             win_rate=win_rate,
#             reward_risk_ratio=rr_ratio,
#             initial_bet_percentage=init_bet,
#             n_trades=20,
#         )
#         final_balances.append(final_balance)
#     median_balance = np.median(final_balances)
#     results[(win_rate, rr_ratio, init_bet)] = median_balance

# # 轉成 DataFrame 格式方便畫圖
# import pandas as pd

# df = pd.DataFrame(
#     [
#         {"WinRate": k[0], "RR": k[1], "InitBet%": k[2], "MedianBalance": v}
#         for k, v in results.items()
#     ]
# )

# # Pivot 成表格（橫軸: 勝率, 縱軸: 風報比 x 初始下注比例）
# pivot_table = df.pivot_table(
#     index=["RR", "InitBet%"], columns="WinRate", values="MedianBalance"
# )

# # 畫 Heatmap
# plt.figure(figsize=(12, 8))
# sns.heatmap(
#     pivot_table,
#     annot=True,
#     fmt=".1e",  # 科學記號格式 (小數點後1位)
#     cmap="YlGnBu"
# )
# plt.title("Median Final Balance after 1000 Trades - Sensitivity Analysis", fontsize=18)
# plt.xlabel("Win Rate", fontsize=14)
# plt.ylabel("Reward/Risk & Initial Bet %", fontsize=14)
# plt.show()

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import itertools

# # 模擬不同再平衡條件的交易
# def simulate_trades_with_rebalance(initial_balance, win_rate, reward_risk_ratio, initial_bet_percentage,
#                                    n_trades, rebalance_gain_threshold, rebalance_save_ratio):
#     balance = initial_balance
#     saved = 0
#     bet_percentage = initial_bet_percentage
#     min_bet = 0.02
#     max_bet = 0.15
#     reset_bet = 0.08
#     trades = 0
#     peak_balance = balance

#     balances = [balance + saved]

#     while trades < n_trades:
#         bet_amount = balance * bet_percentage

#         if np.random.rand() < win_rate:
#             gain = bet_amount * reward_risk_ratio
#             balance += gain
#             bet_percentage = min(bet_percentage + 0.02, max_bet)
#             if balance + saved < initial_balance:
#                 bet_percentage = min(bet_percentage, reset_bet)
#         else:
#             loss = bet_amount
#             balance -= loss
#             bet_percentage = max(bet_percentage - 0.02, min_bet)

#         trades += 1
#         total_balance = balance + saved
#         balances.append(total_balance)

#         # 再平衡邏輯
#         if total_balance >= peak_balance * (1 + rebalance_gain_threshold):
#             saved += balance * rebalance_save_ratio
#             balance *= (1 - rebalance_save_ratio)
#             bet_percentage = reset_bet
#             peak_balance = balance + saved
#         elif total_balance <= peak_balance * 0.75:
#             bet_percentage = reset_bet
#             peak_balance = balance + saved

#     return balances[-1]

# # 定義參數
# initial_balance = 1000
# win_rate = 0.5
# reward_risk_ratio = 3
# initial_bet_percentage = 0.08
# n_trades = 20

# rebalance_gain_thresholds = [0.2, 0.30, 0.45, 0.60]  # 賺30%、45%、60%觸發
# rebalance_save_ratios = [0.0, 0.15, 0.20, 0.25, 0.30]      # 保留20%、25%、30%

# # 跑模擬
# results = {}

# for gain_threshold, save_ratio in itertools.product(rebalance_gain_thresholds, rebalance_save_ratios):
#     final_balances = []
#     for _ in range(500):
#         final_balance = simulate_trades_with_rebalance(
#             initial_balance,
#             win_rate,
#             reward_risk_ratio,
#             initial_bet_percentage,
#             n_trades,
#             rebalance_gain_threshold=gain_threshold,
#             rebalance_save_ratio=save_ratio
#         )
#         final_balances.append(final_balance)
#     median_balance = np.median(final_balances)
#     results[(gain_threshold, save_ratio)] = median_balance

# # 整理成 DataFrame
# df_rebalance = pd.DataFrame(
#     [
#         {"GainThreshold": k[0], "SaveRatio": k[1], "MedianBalance": v}
#         for k, v in results.items()
#     ]
# )

# # Pivot table
# pivot_table_rebalance = df_rebalance.pivot_table(
#     index="SaveRatio",
#     columns="GainThreshold",
#     values="MedianBalance"
# )

# # 畫 Heatmap
# plt.figure(figsize=(10, 7))
# sns.heatmap(
#     pivot_table_rebalance,
#     annot=True,
#     fmt=".1e",  # 科學記號
#     cmap="YlGnBu"
# )
# plt.title('Median Final Balance - Different Rebalance Conditions', fontsize=18)
# plt.xlabel('Gain Threshold (%)', fontsize=14)
# plt.ylabel('Save Ratio (%)', fontsize=14)
# plt.show()


# # -------
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import itertools

# def simulate_trades_with_rebalance(initial_balance, win_rate, reward_risk_ratio, initial_bet_percentage,
#                                    n_trades, rebalance_gain_threshold, rebalance_save_ratio):
#     balance = initial_balance
#     saved = 0
#     bet_percentage = initial_bet_percentage
#     min_bet = 0.02
#     max_bet = 0.15
#     reset_bet = initial_bet_percentage
#     trades = 0
#     peak_balance = balance

#     while trades < n_trades:
#         bet_amount = balance * bet_percentage
#         if np.random.rand() < win_rate:
#             gain = bet_amount * reward_risk_ratio
#             balance += gain
#             bet_percentage = min(bet_percentage + 0.02, max_bet)
#             if balance + saved < initial_balance:
#                 bet_percentage = min(bet_percentage, reset_bet)
#         else:
#             loss = bet_amount
#             balance -= loss
#             bet_percentage = max(bet_percentage - 0.02, min_bet)

#         trades += 1
#         total_balance = balance + saved

#         if total_balance >= peak_balance * (1 + rebalance_gain_threshold):
#             saved += balance * rebalance_save_ratio
#             balance *= (1 - rebalance_save_ratio)
#             bet_percentage = reset_bet
#             peak_balance = balance + saved
#         elif total_balance <= peak_balance * 0.75:
#             bet_percentage = reset_bet
#             peak_balance = balance + saved

#     return balance + saved

# # 參數
# initial_balance = 1000
# n_trades = 60
# sim_times = 10000

# win_rates = [0.2, 0.3, 0.4, 0.5]
# reward_risk_ratios = [1, 2, 3]
# initial_bet_percentages = [0.02, 0.04, 0.06, 0.08, 0.10]
# rebalance_gain_thresholds = [0.2, 0.30, 0.4, 0.45]
# rebalance_save_ratios = [0.00, 0.15, 0.20, 0.25]

# # 收集結果
# # all_results = []
# results = []

# # for reward_risk_ratio in reward_risk_ratios:
# #     for rebalance_gain, rebalance_save in itertools.product(rebalance_gain_thresholds, rebalance_save_ratios):
# #         for win_rate, bet_percentage in itertools.product(win_rates, initial_bet_percentages):
# #             final_balances = []
# #             for _ in range(sim_times):
# #                 final_balance = simulate_trades_with_rebalance(
# #                     initial_balance,
# #                     win_rate,
# #                     reward_risk_ratio,
# #                     bet_percentage,
# #                     n_trades,
# #                     rebalance_gain,
# #                     rebalance_save
# #                 )
# #                 final_balances.append(final_balance)
# #             median_balance = np.median(final_balances)
# #             all_results.append({
# #                 "WinRate": win_rate,
# #                 "BetPercentage": bet_percentage,
# #                 "RewardRisk": reward_risk_ratio,
# #                 "RebalanceGain": rebalance_gain,
# #                 "RebalanceSave": rebalance_save,
# #                 "MedianBalance": median_balance
# #             })

# # # 整理成 DataFrame
# # df_all = pd.DataFrame(all_results)

# # # 找出 Top 10
# # top10 = df_all.sort_values(by="MedianBalance", ascending=False)#.head(20)
# # top10.to_csv("D:\下載\Volatility_Research\processed\position_szing_sim\simulation.csv")
# # # 輸出 Top 10
# # print("\n=== Top 10 Best Parameter Combinations ===\n")
# # print(top10.to_string(index=False, float_format="%.2e"))

# # # 🔥 自動畫全部熱力圖 (RewardRisk × RebalanceGain × RebalanceSave)

# # # 設定畫布大小
# # n_cols = 4  # 每排4張
# # n_rows = int(np.ceil(len(reward_risk_ratios) * len(rebalance_gain_thresholds) * len(rebalance_save_ratios) / n_cols))

# # fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))

# # # 保證 axes 是一維
# # axes = axes.flatten()

# # plot_idx = 0

# # for reward_risk_ratio in reward_risk_ratios:
# #     for rebalance_gain, rebalance_save in itertools.product(rebalance_gain_thresholds, rebalance_save_ratios):
# #         subset = df_all[
# #             (df_all["RewardRisk"] == reward_risk_ratio) &
# #             (df_all["RebalanceGain"] == rebalance_gain) &
# #             (df_all["RebalanceSave"] == rebalance_save)
# #         ]

# #         if not subset.empty:
# #             pivot_table = subset.pivot_table(
# #                 index="WinRate",
# #                 columns="BetPercentage",
# #                 values="MedianBalance"
# #             )

# #             ax = axes[plot_idx]
# #             sns.heatmap(pivot_table, annot=True, fmt=".1e", cmap="YlGnBu", ax=ax)
# #             ax.set_title(f"RR {reward_risk_ratio} | Gain {rebalance_gain*100:.0f}% | Save {rebalance_save*100:.0f}%", fontsize=10)
# #             ax.set_xlabel('Bet %', fontsize=8)
# #             ax.set_ylabel('Win Rate', fontsize=8)
# #             plot_idx += 1

# # # 移除多餘的子圖
# # for i in range(plot_idx, len(axes)):
# #     fig.delaxes(axes[i])

# # plt.tight_layout()
# # plt.show()

# # 設定參數
# win_rates = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
# reward_risk_ratios = [1, 2, 3, 4]
# initial_bet_percentages = [0.02, 0.04, 0.06, 0.08, 0.10]
# rebalance_gain_thresholds = [0.2, 0.30, 0.45, 0.60]
# rebalance_save_ratios = [0.0, 0.15, 0.20, 0.25, 0.30]

# simulations = 5000
# trades_per_simulation = 100

# # MDD計算 function
# def calculate_mdd(balance_array):
#     peak = np.maximum.accumulate(balance_array)
#     drawdown = (peak - balance_array) / peak
#     return np.max(drawdown)

# # 開始模擬
# for (win_rate, rr_ratio, init_bet, reb_gain, reb_save) in itertools.product(
#     win_rates, reward_risk_ratios, initial_bet_percentages, rebalance_gain_thresholds, rebalance_save_ratios
# ):
#     final_balances = []
#     final_mdds = []

#     for _ in range(simulations):
#         balance = 1000
#         bet_percentage = init_bet
#         peak_balance = balance
#         balances = [balance]

#         for _ in range(trades_per_simulation):
#             bet_amount = balance * bet_percentage
#             if np.random.rand() < win_rate:
#                 balance += bet_amount * rr_ratio
#                 if balance < peak_balance:
#                     bet_percentage = min(bet_percentage + 0.02, 0.08)
#                 else:
#                     bet_percentage = min(bet_percentage + 0.02, 0.15)
#             else:
#                 balance -= bet_amount
#                 bet_percentage = max(bet_percentage - 0.02, 0.02)

#             # 再平衡條件
#             if balance >= peak_balance * (1 + reb_gain):
#                 saved = balance * reb_save
#                 balance -= saved
#                 peak_balance = balance
#                 bet_percentage = init_bet

#             if balance <= peak_balance * (1 - 0.25):
#                 peak_balance = balance
#                 bet_percentage = init_bet

#             balances.append(balance)

#         final_balances.append(balance)
#         final_mdds.append(calculate_mdd(np.array(balances)))

#     # 計算中位數與 Performance Score
#     final_balance_median = np.median(final_balances)
#     final_mdd_median = np.median(final_mdds)

#     final_return_median = final_balance_median / initial_balance
#     performance_score = final_return_median / final_mdd_median if final_mdd_median != 0 else np.nan

#     results.append({
#         'win_rate': win_rate,
#         'reward_risk_ratio': rr_ratio,
#         'initial_bet_percentage': init_bet,
#         'rebalance_gain_threshold': reb_gain,
#         'rebalance_save_ratio': reb_save,
#         'final_balance_median': final_balance_median,
#         'final_return_median':final_return_median,
#         'final_mdd_median': final_mdd_median,
#         'performance_score': performance_score
#     })

# # 組成 DataFrame
# results_df = pd.DataFrame(results)

# # 輸出 Top 100 結果
# top100 = results_df.sort_values(by='performance_score', ascending=False)
# top100.to_csv("D:\下載\Volatility_Research\processed\position_szing_sim\simulation.csv")
# print("✅ Top 100 結果已輸出到 'top100_performance_score.xlsx'")

# # 畫熱力圖（範例: win_rate vs reward_risk_ratio）
# pivot = results_df.pivot_table(
#     index='win_rate',
#     columns='reward_risk_ratio',
#     values='performance_score'
# )

# plt.figure(figsize=(10, 8))
# sns.heatmap(pivot, annot=True, fmt=".1e", cmap="YlGnBu", linewidths=0.5)
# plt.title('Performance Score (Median Final Balance / Median MDD)')
# plt.xlabel('Reward-Risk Ratio')
# plt.ylabel('Win Rate')
# plt.show()

import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

# 設定參數
win_rates = [0.35, 0.4, 0.45, 0.5]
reward_risk_ratios = [2, 3]
initial_bet_percentages = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
rebalance_gain_thresholds = [0.2, 0.30, 0.45, 0.60, 0.9]
rebalance_save_ratios = [0.0, 0.15, 0.20, 0.25, 0.30]

simulations = 1000000
trades_per_simulation = 100
initial_balance = 1000


# MDD計算 function
def calculate_mdd(balance_array):
    peak = np.maximum.accumulate(balance_array)
    drawdown = (peak - balance_array) / peak
    return np.max(drawdown)


# 幾何平均報酬計算 function
def geometric_mean_return(balance_array):
    total_return = balance_array[-1] / balance_array[0]
    return total_return ** (1 / len(balance_array)) - 1


# 開始模擬
results = []

for win_rate, rr_ratio, init_bet, reb_gain, reb_save in itertools.product(
    win_rates,
    reward_risk_ratios,
    initial_bet_percentages,
    rebalance_gain_thresholds,
    rebalance_save_ratios,
):
    final_balances = []
    final_mdds = []
    final_geo_returns = []

    for _ in range(simulations):
        balance = initial_balance
        saved_balance = 0
        bet_percentage = init_bet
        peak_balance = balance
        balances = [balance]
        total_balances = [balance + saved_balance]

        for _ in range(trades_per_simulation):
            bet_amount = balance * bet_percentage
            if np.random.rand() < win_rate:
                balance += bet_amount * rr_ratio
                if balance < peak_balance:
                    bet_percentage = min(bet_percentage + 0.02, 0.08)
                else:
                    bet_percentage = min(bet_percentage + 0.02, 0.15)
            else:
                balance -= bet_amount
                bet_percentage = max(bet_percentage - 0.02, 0.02)

            # 再平衡條件
            if balance >= peak_balance * (1 + reb_gain):
                saved = balance * reb_save
                saved_balance += saved
                balance -= saved
                peak_balance = balance
                bet_percentage = init_bet

            if balance <= peak_balance * (1 - 0.25):
                peak_balance = balance
                bet_percentage = init_bet

            balances.append(balance)
            total_balances.append(balance + saved_balance)

        final_balances.append(total_balances[-1])
        final_mdds.append(calculate_mdd(np.array(total_balances)))
        final_geo_returns.append(geometric_mean_return(np.array(total_balances)))

    # 計算中位數與 Performance Score
    final_balance_median = np.median(final_balances)
    final_mdd_median = np.median(final_mdds)
    final_geo_return_median = np.median(final_geo_returns)

    performance_score = (
        final_geo_return_median / final_mdd_median if final_mdd_median != 0 else np.nan
    )

    results.append(
        {
            "win_rate": win_rate,
            "reward_risk_ratio": rr_ratio,
            "initial_bet_percentage": init_bet,
            "rebalance_gain_threshold": reb_gain,
            "rebalance_save_ratio": reb_save,
            "final_balance_median": final_balance_median,
            "final_geo_return_median": final_geo_return_median,
            "final_mdd_median": final_mdd_median,
            "performance_score": performance_score,
        }
    )

# 組成 DataFrame
results_df = pd.DataFrame(results)

# 輸出 Top 100 結果
top100 = results_df.sort_values(by="performance_score", ascending=False)
top100.to_csv(
    r"D:\下載\Volatility_Research\processed\position_szing_sim\simulation.csv",
    index=False,
)
print("✅ Top 100 結果已輸出到 'simulation.csv'")

# 畫熱力圖（範例: win_rate vs reward_risk_ratio）
pivot = results_df.pivot_table(
    index="win_rate", columns="reward_risk_ratio", values="performance_score"
)

plt.figure(figsize=(10, 8))
sns.heatmap(pivot, annot=True, fmt=".1e", cmap="YlGnBu", linewidths=0.5)
plt.title("Performance Score (Median Geometric Return / Median MDD)")
plt.xlabel("Reward-Risk Ratio")
plt.ylabel("Win Rate")
plt.show()
