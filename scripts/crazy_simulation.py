import numpy as np
import matplotlib.pyplot as plt

simulations = 10000000
max_trades = 1000
initial_balance = 1000
win_rate = 0.5
reward_risk_ratio = 3
target_balance = 2_000_000
retreat_stop_level = 5  # 第 5 關後不再退關
record_trades = 500  # 限制最多記錄多少筆交易軌跡以節省記憶體

# 下注等級依勝負改變：27%, 9%, 3%, 1%
bet_levels = [0.27, 0.09, 0.03, 0.01]
# bet_levels = [0.36, 0.12, 0.04, 0.02]

# 每一關的成長門檻（80%）
level_up_threshold = 0.8
min_bet_percentage = 0.01


class no_back_stage:
    def simulate_growth():
        all_paths = []
        trades_to_target = []

        for _ in range(simulations):
            balance = initial_balance
            base_balance = initial_balance
            level_gain = 0
            bet_index = 0
            trades = 0
            path = [balance]

            while trades < max_trades:
                bet_amount = base_balance * bet_levels[bet_index]
                trades += 1

                if np.random.rand() < win_rate:
                    gain = bet_amount * reward_risk_ratio
                    balance += gain
                    level_gain += gain
                    # 恢復下注級別（往前一階）
                    if bet_index > 0:
                        bet_index -= 1
                else:
                    balance -= bet_amount
                    # 降低下注級別（往後一階，最低為1%）
                    if bet_index < len(bet_levels) - 1:
                        bet_index += 1

                # 檢查是否過關
                if level_gain >= base_balance * level_up_threshold:
                    base_balance = balance
                    level_gain = 0
                    bet_index = 0  # 重設下注級別為最高

                path.append(balance)

                if balance >= target_balance:
                    trades_to_target.append(trades)
                    break

            all_paths.append(path)

        return all_paths, trades_to_target

    # 執行模擬
    paths, trades_required = simulate_growth()

    # 找出成長到 1000 萬的交易數
    trades_required = np.array(trades_required)

    # 畫資金成長曲線
    plt.figure(figsize=(12, 6))
    for path in paths[:10]:  # 只畫前 10 條模擬
        plt.plot(path, alpha=0.3, color="blue")
    plt.title("Equity Growing Curve (show 10 lines simulation)")
    plt.xlabel("Traded times")
    plt.ylabel("Total Equity (USD)")
    plt.grid(True)
    plt.show()

    # 畫達標所需交易次數分布
    plt.figure(figsize=(12, 6))
    plt.hist(trades_required, bins=50, color="blue", alpha=0.7)
    median_trades = np.median(trades_required)
    plt.axvline(
        median_trades,
        color="red",
        linestyle="--",
        label=f"Median: {int(median_trades)}",
    )
    plt.title("The times you need to reach 10M USD Distribution")
    plt.xlabel("Traded times")
    plt.ylabel("Simulation times")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"中位數達到 1000 萬美元所需交易次數為：{int(median_trades)}")


class back_stage:
    # 模擬函式（含退關機制）
    def simulate_advance_and_retreat():
        trade_counts_to_target = []
        final_balances = []
        balance_paths = []

        for _ in range(simulations):
            balance = initial_balance
            current_base_balance = initial_balance
            bet_level_index = 0
            trades = 0
            balance_path = [balance]

            while trades < max_trades:
                bet_pct = bet_levels[bet_level_index]
                bet_amount = current_base_balance * bet_pct

                if np.random.rand() < win_rate:
                    balance += bet_amount * reward_risk_ratio
                    if bet_level_index > 0:
                        bet_level_index -= 1
                else:
                    balance -= bet_amount
                    if bet_level_index < len(bet_levels) - 1:
                        bet_level_index += 1
                    else:
                        # 退關：在最小層級再輸一把就重置
                        bet_level_index = 0

                trades += 1
                balance_path.append(balance)

                # 升關檢查
                if balance >= current_base_balance * (1 + level_up_threshold):
                    current_base_balance = balance
                    bet_level_index = 0

                if balance >= target_balance:
                    trade_counts_to_target.append(trades)
                    break

            final_balances.append(balance)
            balance_paths.append(balance_path)

        return final_balances, balance_paths, trade_counts_to_target

    # 執行模擬
    final_balances, balance_paths, trade_counts_to_target = (
        simulate_advance_and_retreat()
    )

    # 資金成長曲線（範例只畫前100筆）
    plt.figure(figsize=(12, 6))
    for path in balance_paths[:10]:
        plt.plot(path, alpha=0.8, linewidth=0.5)
    plt.title("Balance Growth Paths (First 100 Simulations)")
    plt.xlabel("Trades")
    plt.ylabel("Balance")
    plt.grid(True)
    plt.show()

    # 達標次數分布圖
    if trade_counts_to_target:
        plt.figure(figsize=(12, 6))
        plt.hist(trade_counts_to_target, bins=50, color="skyblue", edgecolor="black")
        plt.axvline(
            np.median(trade_counts_to_target),
            color="red",
            linestyle="--",
            label=f"Median = {np.median(trade_counts_to_target):.0f} trades",
        )
        plt.title("Distribution of Trades Needed to Reach $10M")
        plt.xlabel("Number of Trades")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No simulations reached the $10M target within the trade limit.")


# class mix_stage:
#     def simulate_strategy(mix_retreat=True):
#         final_balances = []
#         reach_goal_steps = []
#         success_paths = []
#         fail_paths = []

#         for _ in range(simulations):
#             balance = initial_balance
#             base_balance = balance
#             bet_index = 0
#             level = 1
#             win_accum = 0
#             path = []

#             for trade in range(max_trades):
#                 bet_amount = base_balance * bet_levels[bet_index]

#                 if np.random.rand() < win_rate:
#                     gain = bet_amount * reward_risk_ratio
#                     balance += gain
#                     win_accum += gain / base_balance
#                     if bet_index > 0:
#                         bet_index -= 1
#                 else:
#                     balance -= bet_amount
#                     if bet_index < len(bet_levels) - 1:
#                         bet_index += 1
#                     elif mix_retreat and level < retreat_stop_level:
#                         bet_index = 0  # 退關機制

#                 if win_accum >= level_up_threshold:
#                     level += 1
#                     base_balance = balance
#                     bet_index = 0
#                     win_accum = 0

#                 if trade < record_trades:
#                     path.append(balance)

#                 if balance >= target_balance:
#                     reach_goal_steps.append(trade + 1)
#                     if trade < record_trades:
#                         success_paths.append(path)
#                     break
#             else:
#                 reach_goal_steps.append(None)
#                 if len(path) == record_trades:
#                     fail_paths.append(path)

#             final_balances.append(balance)

#         return final_balances, reach_goal_steps, success_paths, fail_paths


#     # 執行模擬（混合式退關策略）
#     final_balances, reach_goal_steps, success_paths, fail_paths = simulate_strategy(mix_retreat=True)

#     # 畫成功者的資金成長曲線（中位數）
#     if success_paths:
#         max_len_success = max(len(path) for path in success_paths)
#         # 使用固定長度來處理每條路徑
#         success_matrix = []
#         for path in success_paths:
#             padded_path = path + [np.nan] * (max_len_success - len(path))  # 填充NaN至相同長度
#             success_matrix.append(padded_path)
#         success_matrix = np.array(success_matrix)

#         median_success = np.nanmedian(success_matrix, axis=0)
#         plt.figure(figsize=(12, 6))
#         plt.plot(median_success, label='Successful Paths (Median)', color='green')
#         plt.axhline(target_balance, color='gray', linestyle='--', label=f'Target: ${target_balance/1_000_000:.0f}M')
#         plt.title('Capital Growth Curve (Successful Simulations)')
#         plt.xlabel('Trades')
#         plt.ylabel('Balance')
#         plt.grid(True)
#         plt.legend()
#         plt.show()

#     # 畫失敗者的資金成長曲線（中位數）
#     if fail_paths:
#         max_len_fail = max(len(path) for path in fail_paths)
#         # 使用固定長度來處理每條路徑
#         fail_matrix = []
#         for path in fail_paths:
#             padded_path = path + [np.nan] * (max_len_fail - len(path))  # 填充NaN至相同長度
#             fail_matrix.append(padded_path)
#         fail_matrix = np.array(fail_matrix)

#         median_fail = np.nanmedian(fail_matrix, axis=0)
#         plt.figure(figsize=(12, 6))
#         plt.plot(median_fail, label='Failed Paths (Median)', color='red')
#         plt.title('Capital Growth Curve (Failed Simulations)')
#         plt.xlabel('Trades')
#         plt.ylabel('Balance')
#         plt.grid(True)
#         plt.legend()
#         plt.show()

#     # 畫達標交易次數分布圖
#     valid_steps = [s for s in reach_goal_steps if s is not None]
#     if valid_steps:
#         plt.figure(figsize=(12, 6))
#         plt.hist(valid_steps, bins=50, color='purple', alpha=0.7, density=True)
#         median_step = np.median(valid_steps)
#         plt.axvline(median_step, color='red', linestyle='--', label=f'Median = {int(median_step)}')
#         plt.title('Distribution of Trades to Reach $10M')
#         plt.xlabel('Number of Trades')
#         plt.ylabel('Density')
#         plt.legend()
#         plt.grid(True)
#         plt.show()


no_back_stage.simulate_growth()
back_stage.simulate_advance_and_retreat()
# mix_stage.simulate_strategy()
