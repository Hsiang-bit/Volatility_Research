import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


class SignalAnalyzer:
    """
    多種訊號連續出現影響勝率的分析系統
    """

    def __init__(self, signal_columns, result_column="result", odds=2.0):
        """
        初始化分析器

        參數:
        - signal_columns: 訊號列的名稱列表
        - result_column: 結果列的名稱 (1表示勝利, 0表示失敗)
        - odds: 賠率
        """
        self.signal_columns = signal_columns
        self.result_column = result_column
        self.odds = odds
        self.significance_results = {}
        self.expected_values = {}
        self.kelly_fractions = {}

    def process_data(self, df):
        """
        處理數據，計算連續訊號和組合訊號

        參數:
        - df: 包含訊號和結果的DataFrame

        返回:
        - 處理後的DataFrame
        """
        processed_df = df.copy()

        # 計算每個訊號的連續出現次數
        for signal in self.signal_columns:
            # 創建連續計數特徵
            processed_df[f"{signal}_streak"] = (
                processed_df[signal]
                .groupby(
                    (processed_df[signal] != processed_df[signal].shift()).cumsum()
                )
                .cumcount()
                + 1
            )

            # 只保留值為1的連續計數
            processed_df[f"{signal}_streak"] = (
                processed_df[f"{signal}_streak"] * processed_df[signal]
            )

        # 創建訊號組合特徵
        for i, signal1 in enumerate(self.signal_columns):
            for j, signal2 in enumerate(self.signal_columns[i + 1 :], i + 1):
                combo_name = f"{signal1}_{signal2}_combo"
                processed_df[combo_name] = (
                    (processed_df[signal1] == 1) & (processed_df[signal2] == 1)
                ).astype(int)

        # 如果有3個或更多訊號，創建所有訊號同時出現的組合
        if len(self.signal_columns) >= 3:
            all_signals_combo = processed_df[self.signal_columns[0]] == 1
            for signal in self.signal_columns[1:]:
                all_signals_combo = all_signals_combo & (processed_df[signal] == 1)
            processed_df["all_signals_combo"] = all_signals_combo.astype(int)

        return processed_df

    def calculate_win_rates(self, df):
        """
        計算各種訊號條件下的勝率

        參數:
        - df: 處理後的DataFrame

        返回:
        - 包含各種訊號勝率的字典
        """
        signal_win_rates = {}

        # 計算基準勝率 (無訊號情況)
        no_signal_condition = df[self.signal_columns[0]] == 0
        for signal in self.signal_columns[1:]:
            no_signal_condition = no_signal_condition & (df[signal] == 0)
        base_win_rate = df[no_signal_condition][self.result_column].mean()
        signal_win_rates["base_rate"] = base_win_rate

        # 單一訊號的勝率
        for signal in self.signal_columns:
            signal_win_rates[signal] = df[df[signal] == 1][self.result_column].mean()

            # 連續出現的勝率 (分析連續1-5次的情況)
            streak_column = f"{signal}_streak"
            for streak in range(1, 6):
                key = f"{signal}_streak_{streak}"
                if (df[streak_column] == streak).any():
                    signal_win_rates[key] = df[df[streak_column] == streak][
                        self.result_column
                    ].mean()

        # 訊號組合的勝率
        combo_columns = [col for col in df.columns if "_combo" in col]
        for combo in combo_columns:
            if (df[combo] == 1).any():
                signal_win_rates[combo] = df[df[combo] == 1][self.result_column].mean()

        return signal_win_rates

    def test_significance(self, df, signal_win_rates):
        """
        檢測各訊號勝率與基準勝率的差異顯著性

        參數:
        - df: 處理後的DataFrame
        - signal_win_rates: 包含各種訊號勝率的字典

        返回:
        - 包含顯著性檢驗結果的字典
        """
        base_win_rate = signal_win_rates["base_rate"]
        significance_results = {}

        for signal, win_rate in signal_win_rates.items():
            if signal == "base_rate":
                continue

            # 解析訊號條件
            if "_streak_" in signal:
                signal_name, streak = signal.split("_streak_")
                streak = int(streak)
                condition = df[f"{signal_name}_streak"] == streak
            elif "_combo" in signal:
                condition = df[signal] == 1
            else:
                condition = df[signal] == 1

            # 計算在該訊號條件下的勝敗次數
            total = condition.sum()
            wins = ((df[condition])[self.result_column] == 1).sum()

            if total > 10:  # 確保樣本量足夠
                # 執行比例檢驗
                try:
                    z_stat, p_val = stats.proportions_ztest(
                        [wins, int(base_win_rate * total)], [total, total]
                    )
                    significance_results[signal] = {
                        "win_rate": win_rate,
                        "base_rate": base_win_rate,
                        "difference": win_rate - base_win_rate,
                        "total_samples": total,
                        "wins": wins,
                        "p_value": p_val,
                        "significant": p_val < 0.05,
                    }
                except:
                    pass  # 如果檢驗失敗(如除零)，跳過該訊號

        self.significance_results = significance_results
        return significance_results

    def calculate_expected_values(self):
        """
        計算每種訊號組合的期望值和凱利比例

        返回:
        - 包含期望值和凱利比例的字典
        """
        expected_values = {}
        kelly_fractions = {}

        for signal, stats in self.significance_results.items():
            win_rate = stats["win_rate"]
            # 計算期望值 = 獲勝機率*賠率 - 損失機率*1
            expected_values[signal] = win_rate * self.odds - (1 - win_rate)

            # 使用凱利公式計算最佳下注比例
            if expected_values[signal] > 0:  # 只在正期望值情況下下注
                b = self.odds - 1  # 淨賠率
                f = (b * win_rate - (1 - win_rate)) / b  # 凱利公式
                kelly_fractions[signal] = max(0, f)  # 確保不為負
            else:
                kelly_fractions[signal] = 0

        self.expected_values = expected_values
        self.kelly_fractions = kelly_fractions

        return expected_values, kelly_fractions

    def train_model(self, df):
        """
        訓練機器學習模型，評估訊號的預測能力

        參數:
        - df: 處理後的DataFrame

        返回:
        - 模型評估結果和特徵重要性
        """
        # 準備特徵和目標變量
        signal_features = self.signal_columns.copy()
        streak_features = [f"{signal}_streak" for signal in self.signal_columns]
        combo_features = [col for col in df.columns if "_combo" in col]

        features = signal_features + streak_features + combo_features
        features = [f for f in features if f in df.columns]  # 確保所有特徵都存在

        X = df[features]
        y = df[self.result_column]

        # 分割數據
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # 訓練模型
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 評估模型
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)

        # 獲取特徵重要性
        feature_importance = pd.DataFrame(
            {"Feature": features, "Importance": model.feature_importances_}
        ).sort_values("Importance", ascending=False)

        return {
            "model": model,
            "accuracy": accuracy,
            "report": report,
            "feature_importance": feature_importance,
        }

    def visualize_results(self, signal_win_rates=None):
        """
        視覺化分析結果

        參數:
        - signal_win_rates: 包含各種訊號勝率的字典
        """
        if signal_win_rates is None and not self.significance_results:
            print("沒有可視化的結果。請先運行分析。")
            return

        # 1. 顯著訊號勝率比較
        plt.figure(figsize=(12, 6))

        # 過濾出顯著的訊號
        sig_signals = [
            s for s, stats in self.significance_results.items() if stats["significant"]
        ]
        if sig_signals:
            # 準備數據
            sig_rates = [self.significance_results[s]["win_rate"] for s in sig_signals]
            base_rate = self.significance_results[sig_signals[0]]["base_rate"]

            # 繪製條形圖
            bars = plt.bar(sig_signals, sig_rates, color="skyblue")
            plt.axhline(
                y=base_rate,
                color="r",
                linestyle="-",
                label=f"基準勝率: {base_rate:.2f}",
            )

            # 添加數值標籤
            for bar, rate in zip(bars, sig_rates):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    rate + 0.01,
                    f"{rate:.2f}",
                    ha="center",
                    va="bottom",
                    rotation=0,
                )

            plt.title("統計顯著的訊號勝率比較")
            plt.xlabel("訊號類型")
            plt.ylabel("勝率")
            plt.xticks(rotation=45, ha="right")
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            print("沒有找到統計顯著的訊號。")

        # 2. 凱利比例與期望值
        if self.kelly_fractions:
            plt.figure(figsize=(12, 6))

            # 過濾出正凱利值的訊號
            positive_kelly = {k: v for k, v in self.kelly_fractions.items() if v > 0}
            if positive_kelly:
                # 準備數據
                signals = list(positive_kelly.keys())
                kelly_values = list(positive_kelly.values())
                ev_values = [self.expected_values[s] for s in signals]

                # 創建雙坐標軸
                fig, ax1 = plt.subplots(figsize=(12, 6))
                ax2 = ax1.twinx()

                # 繪製凱利比例條形圖
                bars = ax1.bar(
                    signals, kelly_values, color="skyblue", alpha=0.7, label="凱利比例"
                )
                ax1.set_ylabel("凱利比例", color="blue")
                ax1.tick_params(axis="y", labelcolor="blue")

                # 添加凱利比例標籤
                for bar, value in zip(bars, kelly_values):
                    ax1.text(
                        bar.get_x() + bar.get_width() / 2,
                        value + 0.01,
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        color="blue",
                        rotation=0,
                    )

                # 繪製期望值線圖
                ax2.plot(signals, ev_values, "r-o", label="期望值")
                ax2.set_ylabel("期望值", color="red")
                ax2.tick_params(axis="y", labelcolor="red")

                # 添加期望值標籤
                for i, value in enumerate(ev_values):
                    ax2.text(
                        i,
                        value + 0.02,
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        color="red",
                    )

                # 圖表設置
                plt.title("訊號的凱利比例與期望值")
                plt.xlabel("訊號類型")
                plt.xticks(rotation=45, ha="right")

                # 添加圖例
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

                plt.tight_layout()
                plt.show()
            else:
                print("沒有找到正凱利值的訊號。")

    def recommend_betting_strategy(self):
        """
        推薦下注策略

        返回:
        - 包含下注建議的字典
        """
        if not self.kelly_fractions:
            return "請先進行分析以計算凱利比例。"

        # 過濾出正凱利值的訊號，並按凱利值排序
        positive_kelly = {k: v for k, v in self.kelly_fractions.items() if v > 0}
        sorted_signals = sorted(
            positive_kelly.items(), key=lambda x: x[1], reverse=True
        )

        recommendations = []

        for signal, kelly in sorted_signals:
            stats = self.significance_results[signal]

            # 計算保守下注策略（使用1/4凱利）
            conservative_bet = kelly / 4

            recommendations.append(
                {
                    "signal": signal,
                    "win_rate": stats["win_rate"],
                    "p_value": stats["p_value"],
                    "expected_value": self.expected_values[signal],
                    "kelly_fraction": kelly,
                    "conservative_bet_fraction": conservative_bet,
                    "sample_size": stats["total_samples"],
                }
            )

        return recommendations

    def run_full_analysis(self, df):
        """
        運行完整分析流程

        參數:
        - df: 原始數據DataFrame

        返回:
        - 分析結果的字典
        """
        # 1. 處理數據
        processed_df = self.process_data(df)

        # 2. 計算各訊號勝率
        signal_win_rates = self.calculate_win_rates(processed_df)

        # 3. 進行顯著性檢驗
        self.test_significance(processed_df, signal_win_rates)

        # 4. 計算期望值和凱利比例
        self.calculate_expected_values()

        # 5. 訓練預測模型
        model_results = self.train_model(processed_df)

        # 6. 生成下注建議
        betting_recommendations = self.recommend_betting_strategy()

        # 7. 視覺化結果
        self.visualize_results(signal_win_rates)

        return {
            "processed_data": processed_df,
            "win_rates": signal_win_rates,
            "significance_results": self.significance_results,
            "expected_values": self.expected_values,
            "kelly_fractions": self.kelly_fractions,
            "model_results": model_results,
            "betting_recommendations": betting_recommendations,
        }


# 示例用法
if __name__ == "__main__":
    # 創建示例數據
    np.random.seed(42)
    n_samples = 1000

    # 生成三種訊號，其中訊號A在連續出現時更可能導致勝利
    data = {
        "timestamp": pd.date_range(start="2023-01-01", periods=n_samples),
        "signal_A": np.random.binomial(1, 0.3, n_samples),
        "signal_B": np.random.binomial(1, 0.25, n_samples),
        "signal_C": np.random.binomial(1, 0.2, n_samples),
    }

    # 創建DataFrame
    df = pd.DataFrame(data)

    # 人為設置連續訊號A更有可能導致勝利
    df["signal_A_streak"] = (
        df["signal_A"]
        .groupby((df["signal_A"] != df["signal_A"].shift()).cumsum())
        .cumcount()
        + 1
    )
    df["signal_A_streak"] = df["signal_A_streak"] * df["signal_A"]

    # 創建結果列
    base_win_rate = 0.48  # 基礎勝率
    df["result"] = np.random.binomial(1, base_win_rate, n_samples)

    # 讓連續的訊號A提高勝率
    df.loc[df["signal_A_streak"] == 2, "result"] = np.random.binomial(
        1, base_win_rate + 0.1, len(df[df["signal_A_streak"] == 2])
    )
    df.loc[df["signal_A_streak"] >= 3, "result"] = np.random.binomial(
        1, base_win_rate + 0.2, len(df[df["signal_A_streak"] >= 3])
    )

    # 讓訊號B和C同時出現時提高勝率
    df.loc[(df["signal_B"] == 1) & (df["signal_C"] == 1), "result"] = (
        np.random.binomial(
            1,
            base_win_rate + 0.15,
            len(df[(df["signal_B"] == 1) & (df["signal_C"] == 1)]),
        )
    )

    # 初始化分析器
    analyzer = SignalAnalyzer(
        signal_columns=["signal_A", "signal_B", "signal_C"], odds=2.0
    )

    # 運行完整分析
    results = analyzer.run_full_analysis(df)

    # 輸出下注建議
    print("\n下注策略建議:")
    recommendations = results["betting_recommendations"]
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. 訊號: {rec['signal']}")
        print(f"   勝率: {rec['win_rate']:.2f}, p值: {rec['p_value']:.4f}")
        print(
            f"   期望值: {rec['expected_value']:.2f}, 凱利比例: {rec['kelly_fraction']:.2f}"
        )
        print(f"   建議下注資金比例: {rec['conservative_bet_fraction']:.2f} (保守策略)")
        print(f"   樣本量: {rec['sample_size']}")
        print()
