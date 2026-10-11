"""FXトレードの「期待値」と「破産確率」を並べて見るためのシミュレーター。

R = 1回のトレードで賭けるリスク額(損切り幅 × ロット)。損益はすべてRの倍数で表す。
  勝ち:  +reward - cost
  負け:  -1       - cost      (cost = スプレッド等の往復コスト。勝敗に関係なく払う)

使い方:  python3 sim_expectancy.py
"""
import numpy as np

N_TRADES = 500
N_PATHS = 20_000
RUIN_LEVEL = 0.5  # 資金が初期の50%を割ったら「実質破綻」とみなす
rng = np.random.default_rng(42)

# (名前, 勝率, 平均利益R, コストR, 1回のリスク=資金の何%)
SCENARIOS = [
    ("A ランダム売買(勝率50% R:R=1:1)",       0.50, 1.0, 0.10, 0.01),
    ("B コツコツドカン(勝率70% R:R=0.4:1)",   0.70, 0.4, 0.10, 0.01),
    ("C 損小利大(勝率40% R:R=2:1)",           0.40, 2.0, 0.10, 0.01),
    ("D Cと同じ手法・リスク5%",                0.40, 2.0, 0.10, 0.05),
    ("E Cと同じ手法・リスク0.5%",              0.40, 2.0, 0.10, 0.005),
    ("F Cの優位がコスト2倍で消える(cost=0.4R)", 0.40, 2.0, 0.40, 0.01),
]


def simulate(p, reward, cost, risk_frac):
    wins = rng.random((N_PATHS, N_TRADES)) < p
    r = np.where(wins, reward - cost, -1.0 - cost)       # R単位の損益
    equity = np.cumprod(1.0 + risk_frac * r, axis=1)     # 複利で資金推移
    return equity


def main():
    header = f"{'シナリオ':<38}{'期待値/回':>9}{'中央値':>9}{'平均':>9}{'DD50%到達':>11}{'最終<1':>8}"
    print(f"{N_TRADES}回トレード × {N_PATHS}通りの未来 (初期資金 = 1.00)\n")
    print(header)
    print("-" * len(header))
    for name, p, reward, cost, f in SCENARIOS:
        ev = p * reward - (1 - p) * 1.0 - cost
        eq = simulate(p, reward, cost, f)
        ruined = (eq.min(axis=1) < RUIN_LEVEL).mean()
        final = eq[:, -1]
        print(f"{name:<38}{ev:>+8.2f}R{np.median(final):>9.2f}{final.mean():>9.2f}"
              f"{ruined:>10.1%}{(final < 1).mean():>9.1%}")


if __name__ == "__main__":
    main()
