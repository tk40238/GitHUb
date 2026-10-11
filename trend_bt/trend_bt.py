#!/usr/bin/env python3
"""
FX trend-following empirical test on public daily data.

Data   : data/daily.csv = GitHub 'datasets/exchange-rates' mirror of FRED/Federal Reserve H.10
         daily noon-NY buying rates (all columns turned out to be FOREIGN-CCY PER USD, README notwithstanding).
         Downloaded from https://raw.githubusercontent.com/datasets/exchange-rates/main/data/daily.csv
Limits : noon snapshots (no highs/lows, not tradable closes), spot only (no carry/swap), results are not a forecast.

Run    : python3 trend_bt.py            (needs numpy, pandas; writes results.json next to this file)

Conventions
  price p_i  = USD per 1 unit of foreign currency (so +1 position = long that currency / short USD)
  signal s_i in {-1,0,+1} (or continuous) decided on close t, held over t -> t+1 (lag=1), lag=2 = extra day of delay
  size_i     = min(TV/sigma_i, MAXLEV), sigma_i = EWMA(span 60) std of daily log returns * sqrt(252)   [inverse-vol, equal risk]
  portfolio  = equal-weight average of the pair strategies; then ONE constant ex-post leverage k per rule so that the
               full-window net vol (at 1x cost) = 10% (does not change Sharpe; makes CAGR/MaxDD comparable).
               The same k is used for the 2x/4x cost reruns.
  costs      = c * |change in notional position| per pair, c = 1bp (1x), 2bp (2x), 4bp (4x) per side
  trades     = position episodes per pair (same sign, flat/reversal ends it), net of costs, assigned by entry date.
"""
import os, sys, json, math
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "daily.csv")
ANN = 252
TV = 0.10          # per-pair vol target
PORT_TV = 0.10     # portfolio vol after ex-post leverage normalisation
MAXLEV = 4.0
VOL_SPAN = 60
COST_BP = {"1x": 1e-4, "2x": 2e-4, "4x": 4e-4}

COLS = {"Euro": "EUR", "United Kingdom": "GBP", "Australia": "AUD", "New Zealand": "NZD", "Japan": "JPY",
        "Switzerland": "CHF", "Canada": "CAD", "Norway": "NOK", "Sweden": "SEK"}
MAIN = ["Euro", "United Kingdom", "Australia", "New Zealand", "Japan", "Switzerland", "Canada", "Norway"]
LONG = ["United Kingdom", "Australia", "New Zealand", "Japan", "Switzerland", "Canada", "Norway", "Sweden"]


# ----------------------------------------------------------------------------- data
def load_prices(cols, start, ffill_limit=5):
    d = pd.read_csv(DATA, parse_dates=["Date"])
    d.columns = ["Date", "Country", "Rate"]
    w = d.pivot(index="Date", columns="Country", values="Rate").sort_index()
    w = w[cols].loc[start:].dropna(how="all").ffill(limit=ffill_limit).dropna()
    p = 1.0 / w                      # USD per FC
    p.columns = [COLS[c] for c in cols]
    return p


# ----------------------------------------------------------------------------- signals
def sig_tsmom(p, L):
    return np.sign(np.log(p) - np.log(p.shift(L)))


def sig_tsmom_multi(p, Ls=(21, 63, 252)):
    return sum(sig_tsmom(p, L) for L in Ls) / len(Ls)


def sig_tsmom_cont(p, L=252):
    """continuous version: 12m log return divided by its own 1-year vol scale, clipped to [-1,1] (Baltas-Kosowski-style)"""
    r = np.log(p).diff()
    z = (np.log(p) - np.log(p.shift(L))) / (r.ewm(span=VOL_SPAN, min_periods=VOL_SPAN).std() * math.sqrt(L))
    return z.clip(-1, 1)


def sig_sma(p, f, s):
    return np.sign(p.rolling(f).mean() - p.rolling(s).mean())


def _donchian_1(x, N, M):
    s = pd.Series(x)
    hiN = s.rolling(N).max().shift(1).values; loN = s.rolling(N).min().shift(1).values
    hiM = s.rolling(M).max().shift(1).values; loM = s.rolling(M).min().shift(1).values
    out = np.zeros(len(x)); st = 0
    for t in range(len(x)):
        v = x[t]
        if st == 0:
            if v > hiN[t]: st = 1
            elif v < loN[t]: st = -1
        elif st == 1:
            if v < loN[t]: st = -1
            elif v < loM[t]: st = 0
        else:
            if v > hiN[t]: st = 1
            elif v > hiM[t]: st = 0
        out[t] = st
    return out


def sig_donchian(p, N, M):
    return pd.DataFrame({c: _donchian_1(p[c].values, N, M) for c in p.columns}, index=p.index)


def _chandelier_1(x, N, k, atr_n):
    s = pd.Series(x)
    hiN = s.rolling(N).max().shift(1).values; loN = s.rolling(N).min().shift(1).values
    atr = s.diff().abs().rolling(atr_n).mean().values       # close-to-close proxy (no H/L in data)
    hhc = s.rolling(atr_n).max().values; llc = s.rolling(atr_n).min().values
    out = np.zeros(len(x)); st = 0; stop = np.nan
    for t in range(len(x)):
        v = x[t]
        if np.isnan(atr[t]):
            out[t] = 0; continue
        if st == 1:
            stop = max(stop, hhc[t] - k * atr[t])            # ratchet: never loosened
            if v < stop: st = 0
        elif st == -1:
            stop = min(stop, llc[t] + k * atr[t])
            if v > stop: st = 0
        if st == 0:
            if v > hiN[t]: st = 1; stop = v - k * atr[t]
            elif v < loN[t]: st = -1; stop = v + k * atr[t]
        out[t] = st
    return out


def sig_chandelier(p, N=20, k=3.0, atr_n=22):
    return pd.DataFrame({c: _chandelier_1(p[c].values, N, k, atr_n) for c in p.columns}, index=p.index)


def sig_er_tsmom(p, L, n, theta):
    er = (p - p.shift(n)).abs() / p.diff().abs().rolling(n).sum()
    return sig_tsmom(p, L).where(er > theta, 0.0)


def sig_csmom(p, L, top=3, rebal=21):
    ret = np.log(p) - np.log(p.shift(L))
    sig = pd.DataFrame(np.nan, index=p.index, columns=p.columns)
    for i in range(L, len(p), rebal):
        r = ret.iloc[i]
        if r.isna().any(): continue
        rk = r.rank(method="first")
        n = len(r)
        s = pd.Series(0.0, index=p.columns)
        s[rk > n - top] = 1.0
        s[rk <= top] = -1.0
        sig.iloc[i] = s
    return sig.ffill().fillna(0.0)


def sig_buyhold(p):
    return pd.DataFrame(1.0, index=p.index, columns=p.columns)


# ----------------------------------------------------------------------------- engine
class Run:
    def __init__(self, sig, r, sigma, lag=1, sizing="vol"):
        size = (TV / sigma).clip(upper=MAXLEV) if sizing == "vol" else pd.DataFrame(1.0, index=r.index, columns=r.columns)
        size = size.fillna(0.0)
        pos = sig.fillna(0.0) * size
        self.P = pos.shift(lag).fillna(0.0)
        self.r = r
        self.gross = self.P * r
        self.turn = self.P.diff().abs().fillna(0.0)

    def port(self, c):
        return (self.gross - c * self.turn).mean(axis=1)

    def k(self, mask):
        v = self.port(COST_BP["1x"])[mask].std() * math.sqrt(ANN)
        return PORT_TV / v if v > 1e-12 else 1.0


def stats(net):
    n = len(net)
    if n < 30: return dict(cagr=np.nan, sharpe=np.nan, maxdd=np.nan, vol=np.nan, n=n)
    eq = (1 + net).cumprod()
    cagr = eq.iloc[-1] ** (ANN / n) - 1
    sd = net.std()
    sharpe = net.mean() / sd * math.sqrt(ANN) if sd > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    return dict(cagr=float(cagr), sharpe=float(sharpe), maxdd=float(dd), vol=float(sd * math.sqrt(ANN)), n=n)


def episodes(run, c):
    """list of (entry_index, pair, net_pnl) for each position episode; pnl in unit of the pair strategy (10% vol target)."""
    out = []
    P = run.P.values; R = run.r.fillna(0.0).values
    for j in range(P.shape[1]):
        p = P[:, j]; rr = R[:, j]
        sp = 0; start = None; pnl = 0.0
        for t in range(len(p)):
            s = int(np.sign(p[t])); g = p[t] * rr[t]
            if s == sp:
                if s != 0: pnl += g - c * abs(p[t] - p[t - 1])
            else:
                if sp != 0:
                    pnl -= c * abs(p[t - 1]); out.append((start, j, pnl))
                if s != 0:
                    start = t; pnl = g - c * abs(p[t])
                sp = s
        if sp != 0: out.append((start, j, pnl))
    return out


def trade_stats(eps, idx, mask_dates):
    sel = [e[2] for e in eps if mask_dates[e[0]]]
    if not sel: return dict(trades=0, win=np.nan, payoff=np.nan)
    a = np.array(sel); w = a[a > 0]; l = a[a <= 0]
    payoff = (w.mean() / abs(l.mean())) if len(w) and len(l) and l.mean() != 0 else np.nan
    return dict(trades=int(len(a)), win=float((a > 0).mean()), payoff=float(payoff))


def boot_sharpe_ci(net, block=21, B=2000, seed=7):
    rng = np.random.default_rng(seed)
    x = net.values; n = len(x)
    nb = int(math.ceil(n / block))
    shs = np.empty(B)
    for b in range(B):
        st = rng.integers(0, n - block + 1, nb)
        s = np.concatenate([x[i:i + block] for i in st])[:n]
        shs[b] = s.mean() / s.std() * math.sqrt(ANN)
    return float(np.percentile(shs, 2.5)), float(np.percentile(shs, 97.5))


# ----------------------------------------------------------------------------- experiment
def build_universe(cols, start, eval_start=None, warm=260):
    p = load_prices(cols, start)
    r = np.log(p).diff().fillna(0.0)
    sigma = r.ewm(span=VOL_SPAN, min_periods=VOL_SPAN).std() * math.sqrt(ANN)
    ev = p.index[warm] if eval_start is None else p.index[p.index >= eval_start][0]
    return p, r, sigma, ev


def masks(idx, ev, is_end="2012-12-31", oos_start="2013-01-01"):
    return {"IS": (idx >= ev) & (idx <= is_end), "OOS": idx >= oos_start, "FULL": idx >= ev}


def evaluate(name, run, idx, ev, with_trades=True, boot=False, costs=None):
    costs = costs or COST_BP
    M = masks(idx, ev)
    k = run.k(M["FULL"])
    rows = {}
    eps = {c: episodes(run, v) for c, v in costs.items()} if with_trades else {}
    for cname, c in costs.items():
        net = run.port(c) * k
        for s, m in M.items():
            st = stats(net[m])
            if with_trades:
                st.update(trade_stats(eps[cname], idx, m.values if hasattr(m, "values") else m))
            if boot and cname == "1x" and s in ("OOS", "FULL"):
                st["sharpe_ci95"] = boot_sharpe_ci(net[m])
            rows[(cname, s)] = st
    gs = {s: stats(run.port(0.0)[m] * k)["sharpe"] for s, m in M.items()}   # gross-of-cost Sharpe
    return dict(name=name, k=float(k), rows=rows, gross_sharpe=gs)


def fmt(v, pct=False, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "  nan"
    return f"{v * 100:6.1f}%" if pct else f"{v:6.{d}f}"


def line(res, cname="1x"):
    out = []
    for s in ("IS", "OOS", "FULL"):
        r = res["rows"][(cname, s)]
        out.append(f"{s}: CAGR {fmt(r['cagr'], True)} Sh {fmt(r['sharpe'])} DD {fmt(r['maxdd'], True)}"
                   + (f" W {fmt(r.get('win'), True)} PO {fmt(r.get('payoff'))} N {r.get('trades')}" if 'trades' in r else ""))
    return "\n    ".join(out)


def sweep_sharpe(label, sig_fn, params, r, sigma, idx, ev, p):
    rows = []
    for prm in params:
        run = Run(sig_fn(prm), r, sigma)
        res = evaluate(label + str(prm), run, idx, ev, with_trades=False)
        rows.append((prm, res))
    return rows


def print_sweep(title, rows):
    print(f"\n  [{title}]  net Sharpe @1x cost  (IS / OOS / FULL)   | OOS @4x | gross FULL")
    for prm, res in rows:
        a = res["rows"]
        print(f"    {str(prm):<22} {a[('1x','IS')]['sharpe']:6.2f} {a[('1x','OOS')]['sharpe']:6.2f} {a[('1x','FULL')]['sharpe']:6.2f}"
              f"   | {a[('4x','OOS')]['sharpe']:6.2f} | {res['gross_sharpe']['FULL']:6.2f}")


def main():
    out = {}
    # ======================= primary: 8 USD pairs 1999-2026 =======================
    p, r, sigma, ev = build_universe(MAIN, "1999-01-01")
    idx = p.index
    print(f"Universe {list(p.columns)}  data {idx[0].date()}..{idx[-1].date()}  n={len(p)}  eval start {ev.date()}")
    print("Daily-return annualised vol per pair:", (r.std() * math.sqrt(ANN)).round(3).to_dict())

    base = {
        "Donchian 55/20": sig_donchian(p, 55, 20),
        "TSMOM 12m sign": sig_tsmom(p, 252),
        "SMA 50/200": sig_sma(p, 50, 200),
    }
    extras = {
        "TSMOM multi 1/3/12m": sig_tsmom_multi(p),
        "TSMOM 12m continuous (clip z)": sig_tsmom_cont(p, 252),
        "Chandelier (20d brk, 6x mean|dP|22)": sig_chandelier(p, 20, 6.0, 22),
        "TSMOM12 + ER60>0.15": sig_er_tsmom(p, 252, 60, 0.15),
        "CS momentum 12m top3/bot3 monthly": sig_csmom(p, 252, 3, 21),
        "Buy&hold long FC/short USD": sig_buyhold(p),
    }
    allsig = {**base, **extras}
    runs = {n: Run(s, r, sigma) for n, s in allsig.items()}
    results = {}
    bh_net = runs["Buy&hold long FC/short USD"].port(COST_BP["1x"])
    print("\n=== PRIMARY RESULTS (net, 1bp/side, ex-post leverage so full-window vol=10%) ===")
    for n, run in runs.items():
        res = evaluate(n, run, idx, ev, with_trades=True, boot=(n in base))
        results[n] = res
        c = run.port(COST_BP["1x"])[masks(idx, ev)["FULL"]].corr(bh_net[masks(idx, ev)["FULL"]])
        print(f"\n  {n}   (k={res['k']:.2f}; corr with always-long-FC = {c:.2f}; gross-of-cost Sharpe IS/OOS/FULL = "
              f"{res['gross_sharpe']['IS']:.2f}/{res['gross_sharpe']['OOS']:.2f}/{res['gross_sharpe']['FULL']:.2f})")
        print("    " + line(res, "1x"))
        for cn in ("2x", "4x"):
            a = res["rows"]
            print(f"    cost {cn}: Sharpe IS/OOS/FULL = {a[(cn,'IS')]['sharpe']:.2f}/{a[(cn,'OOS')]['sharpe']:.2f}/{a[(cn,'FULL')]['sharpe']:.2f}"
                  f" ; CAGR OOS {a[(cn,'OOS')]['cagr']*100:.1f}%")
        for s in ("OOS", "FULL"):
            ci = res["rows"][("1x", s)].get("sharpe_ci95")
            if ci: print(f"    {s} Sharpe 95% block-bootstrap CI (21d blocks): [{ci[0]:.2f}, {ci[1]:.2f}]")
    out["primary"] = {n: {f"{c}|{s}": v for (c, s), v in res["rows"].items()} for n, res in results.items()}
    out["primary_k"] = {n: res["k"] for n, res in results.items()}
    out["primary_gross_sharpe"] = {n: res["gross_sharpe"] for n, res in results.items()}

    # ---- per-pair breakdown (pair-level strategy, net 1x, Sharpe) ----
    print("\n=== PER-PAIR net Sharpe @1x (IS / OOS / FULL) ===")
    M = masks(idx, ev)
    pp = {}
    for n in base:
        run = runs[n]
        print(f"  {n}")
        pp[n] = {}
        for c in p.columns:
            net = run.gross[c] - COST_BP["1x"] * run.turn[c]
            sh = [stats(net[M[s]])["sharpe"] for s in ("IS", "OOS", "FULL")]
            pp[n][c] = sh
            print(f"    {c}: {sh[0]:6.2f} {sh[1]:6.2f} {sh[2]:6.2f}")
    out["per_pair_sharpe"] = pp

    # ---- robustness: extra execution lag, fixed notional (no vol scaling) ----
    print("\n=== ROBUSTNESS: execution lag 2 / fixed-notional sizing (net 1x, Sharpe IS/OOS/FULL) ===")
    rob = {}
    for n, s in base.items():
        for lab, kw in (("lag2", dict(lag=2)), ("fixed-notional", dict(sizing="fixed"))):
            res = evaluate(n + lab, Run(s, r, sigma, **kw), idx, ev, with_trades=False)
            a = res["rows"]
            rob[f"{n}|{lab}"] = [a[("1x", x)]["sharpe"] for x in ("IS", "OOS", "FULL")]
            print(f"  {n:<16} {lab:<15} {rob[f'{n}|{lab}'][0]:6.2f} {rob[f'{n}|{lab}'][1]:6.2f} {rob[f'{n}|{lab}'][2]:6.2f}")
    out["robustness"] = rob

    # ======================= parameter sweeps =======================
    print("\n=== PARAMETER SWEEPS (primary universe) ===")
    sw = {}
    don_grid = [(N, M_) for N in (20, 40, 55, 100) for M_ in (10, 20, 40) if M_ < N]
    sw["donchian"] = sweep_sharpe("don", lambda a: sig_donchian(p, *a), don_grid, r, sigma, idx, ev, p)
    print_sweep("Donchian entry/exit", sw["donchian"])
    sw["tsmom"] = sweep_sharpe("ts", lambda L: sig_tsmom(p, L), [21, 63, 126, 189, 252], r, sigma, idx, ev, p)
    print_sweep("TSMOM lookback (days)", sw["tsmom"])
    sma_grid = [(20, 50), (20, 100), (20, 200), (50, 100), (50, 200), (100, 200)]
    sw["sma"] = sweep_sharpe("sma", lambda a: sig_sma(p, *a), sma_grid, r, sigma, idx, ev, p)
    print_sweep("SMA fast/slow", sw["sma"])
    sw["chand"] = sweep_sharpe("ch", lambda k: sig_chandelier(p, 20, k, 22), [3, 4.5, 6, 9, 12, 18], r, sigma, idx, ev, p)
    print_sweep("Chandelier k (x mean|dClose|, 22d) with 20d entry", sw["chand"])
    er_grid = [(60, th) for th in (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4)] + [(252, th) for th in (0.0, 0.05, 0.1, 0.15, 0.2)]
    sw["er"] = sweep_sharpe("er", lambda a: sig_er_tsmom(p, 252, a[0], a[1]), er_grid, r, sigma, idx, ev, p)
    print_sweep("TSMOM12 gated by ER(n)>theta  (n,theta)", sw["er"])
    sw["cs"] = sweep_sharpe("cs", lambda L: sig_csmom(p, L, 3, 21), [21, 63, 126, 252], r, sigma, idx, ev, p)
    print_sweep("CS momentum lookback (top3/bottom3, monthly)", sw["cs"])
    out["sweeps"] = {k: [(str(prm), {f"{c}|{s}": v["sharpe"] for (c, s), v in res["rows"].items()}) for prm, res in rows]
                     for k, rows in sw.items()}

    # ---- calendar-year net returns (k-scaled, 1x cost) for baselines ----
    print("\n=== CALENDAR-YEAR net return (%), portfolio scaled to ~10% vol, 1x cost ===")
    yr = {}
    for n in list(base) + ["TSMOM 12m continuous (clip z)", "Buy&hold long FC/short USD"]:
        net = runs[n].port(COST_BP["1x"]) * results[n]["k"]
        y = ((1 + net[net.index >= ev]).groupby(net.index[net.index >= ev].year).prod() - 1) * 100
        yr[n] = y.round(1).to_dict()
    yrs = sorted(next(iter(yr.values())).keys())
    print("  year  " + " ".join(f"{n[:14]:>15}" for n in yr))
    for y_ in yrs:
        print(f"  {y_}  " + " ".join(f"{yr[n][y_]:15.1f}" for n in yr))
    out["calendar_year_pct"] = yr

    # ======================= supplementary: EM basket (floating-ish EM currencies) =======================
    print("\n=== SUPPLEMENT: EM basket MXN ZAR BRL KRW INR TWD THB SGD, 1999-2026 (spot only, no carry; 1bp is unrealistic for EM) ===")
    EMCOLS = {"Mexico": "MXN", "South Africa": "ZAR", "Brazil": "BRL", "South Korea": "KRW", "India": "INR",
              "Taiwan": "TWD", "Thailand": "THB", "Singapore": "SGD"}
    COLS.update(EMCOLS)
    pE, rE, sgE, evE = build_universe(list(EMCOLS), "1999-01-01")
    idxE = pE.index
    print("  pair vols:", (rE.std() * math.sqrt(ANN)).round(3).to_dict(), f"eval {evE.date()}..{idxE[-1].date()}")
    costsE = {"1x": 1e-4, "4x": 4e-4, "10bp": 10e-4, "20bp": 20e-4}
    emsig = {"Donchian 55/20": sig_donchian(pE, 55, 20), "TSMOM 12m sign": sig_tsmom(pE, 252), "SMA 50/200": sig_sma(pE, 50, 200),
             "Buy&hold long FC/short USD": sig_buyhold(pE)}
    em = {}
    for n, sg in emsig.items():
        res = evaluate(n, Run(sg, rE, sgE), idxE, evE, with_trades=False, costs=costsE)
        em[n] = {f"{c}|{s_}": v for (c, s_), v in res["rows"].items()}
        a = res["rows"]
        print(f"  {n:<28} Sharpe IS/OOS/FULL  1bp {a[('1x','IS')]['sharpe']:5.2f}/{a[('1x','OOS')]['sharpe']:5.2f}/{a[('1x','FULL')]['sharpe']:5.2f}"
              f" | 10bp {a[('10bp','IS')]['sharpe']:5.2f}/{a[('10bp','OOS')]['sharpe']:5.2f}/{a[('10bp','FULL')]['sharpe']:5.2f}"
              f" | 20bp {a[('20bp','FULL')]['sharpe']:5.2f} | MaxDD FULL {a[('1x','FULL')]['maxdd']*100:.0f}%")
    out["em_supplement"] = em
    print("  per-pair net Sharpe FULL (1bp / 10bp):")
    for n in ("Donchian 55/20", "TSMOM 12m sign", "SMA 50/200"):
        run = Run(emsig[n], rE, sgE)
        mE = masks(idxE, evE)["FULL"]
        cells = []
        for c in pE.columns:
            a1 = stats((run.gross[c] - 1e-4 * run.turn[c])[mE])["sharpe"]
            a10 = stats((run.gross[c] - 10e-4 * run.turn[c])[mE])["sharpe"]
            cells.append(f"{c} {a1:5.2f}/{a10:5.2f}")
        print(f"    {n:<16} " + "  ".join(cells))
    for k_ in EMCOLS: COLS.pop(k_)

    # ======================= long history (no EUR) =======================
    print("\n=== LONG HISTORY: 8 pairs ex-EUR, 1975-2026, net 1x Sharpe by sub-period ===")
    pL, rL, sigL, evL = build_universe(LONG, "1971-01-01", eval_start="1975-01-01", warm=260)
    idxL = pL.index
    longsig = {"Donchian 55/20": sig_donchian(pL, 55, 20), "TSMOM 12m sign": sig_tsmom(pL, 252), "SMA 50/200": sig_sma(pL, 50, 200),
               "Buy&hold long FC/short USD": sig_buyhold(pL)}
    periods = [("1975-1989", "1975-01-01", "1989-12-31"), ("1990-1999", "1990-01-01", "1999-12-31"),
               ("2000-2012", "2000-01-01", "2012-12-31"), ("2013-2026", "2013-01-01", "2026-12-31"),
               ("1975-2026", "1975-01-01", "2026-12-31")]
    lh = {}
    for n, s in longsig.items():
        run = Run(s, rL, sigL)
        full = (idxL >= evL)
        k = run.k(full)
        lh[n] = {}
        row = []
        for lab, a, b in periods:
            m = (idxL >= a) & (idxL <= b)
            st1 = stats(run.port(COST_BP["1x"])[m] * k)
            st4 = stats(run.port(COST_BP["4x"])[m] * k)
            g = stats(run.port(0.0)[m] * k)
            lh[n][lab] = dict(net1x=st1, net4x=st4, gross=g)
            row.append(f"{lab}: {st1['sharpe']:5.2f}(g{g['sharpe']:5.2f}, 4x {st4['sharpe']:5.2f})")
        print(f"  {n:<28} " + " | ".join(row))
    out["long_history"] = lh
    print(f"  (long-history universe: {list(pL.columns)}, data {idxL[0].date()}..{idxL[-1].date()}; 1bp cost is unrealistically low before ~1990)")

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o), indent=1)
    print("\nsaved results.json")


if __name__ == "__main__":
    main()
