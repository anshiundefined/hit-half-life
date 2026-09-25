"""
The half-life of a hit: are songs burning out faster in the streaming era?

Song-level panel from every Billboard Hot 100 chart since 1958:
1. Front-loading: are hits debuting higher and peaking sooner?
2. Survival: Kaplan-Meier curves + Cox proportional hazards for leaving the Top 40, by era.
3. ML: how much of a song's lifespan is predictable from its first three chart weeks, and has that changed?
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.duration.hazard_regression import PHReg
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score

from utils import DATA, HIGHLIGHT, MUTED, PALETTE, get, md_table, savefig, setup, write_results

URL = "https://raw.githubusercontent.com/utdata/rwd-billboard-data/main/data-out/hot-100-current.csv"
SRC = "Billboard Hot 100, 1958–present (compiled by utdata/rwd-billboard-data)"
ERAS = [(1958, 1990, "Radio & sales (–1990)"), (1991, 2004, "SoundScan (1991–2004)"),
        (2005, 2012, "Downloads (2005–12)"), (2013, 2100, "Streaming (2013–)")]
ERA_COL = dict(zip([e[2] for e in ERAS], [MUTED, PALETTE[0], PALETTE[2], HIGHLIGHT]))


def era_of(year: int) -> str:
    for a, b, name in ERAS:
        if a <= year <= b:
            return name
    return ERAS[-1][2]


def load() -> pd.DataFrame:
    d = pd.read_csv(io.StringIO(get(URL).text), parse_dates=["chart_week"])
    d = d.rename(columns={"current_week": "pos"})
    d["song"] = d["title"].str.strip() + " — " + d["performer"].str.strip()
    return d.sort_values(["song", "chart_week"])


def build_songs(d: pd.DataFrame) -> pd.DataFrame:
    last_chart = d["chart_week"].max()
    g = d.groupby("song")
    s = pd.DataFrame({
        "performer": g["performer"].first(),
        "debut": g["chart_week"].min(),
        "last": g["chart_week"].max(),
        "weeks": g.size(),
        "top40_weeks": g["pos"].apply(lambda x: int((x <= 40).sum())),
        "peak": g["pos"].min(),
        "debut_pos": g["pos"].first(),
    })
    first_peak = d.loc[d["pos"] == d.groupby("song")["pos"].transform("min")].groupby("song")["chart_week"].min()
    s["weeks_to_peak"] = ((first_peak - s["debut"]).dt.days // 7).astype(int)
    first3 = d.groupby("song")["pos"].apply(lambda x: list(x.iloc[:3]) + [101] * (3 - len(x.iloc[:3])))
    s[["w1", "w2", "w3"]] = pd.DataFrame(first3.tolist(), index=first3.index)
    s["year"] = s["debut"].dt.year
    s["era"] = s["year"].map(era_of)
    s["censored"] = s["last"] >= last_chart - pd.Timedelta(days=7)   # still charting
    # artist experience: number of earlier Top-40 hits by the same credited performer
    s = s.sort_values("debut")
    s["prior_hits"] = s.groupby("performer")["peak"].transform(lambda p: (p <= 40).cumsum().shift(fill_value=0))
    s.reset_index().to_csv(DATA / "songs.csv", index=False)
    return s


def kaplan_meier(t: np.ndarray, event: np.ndarray) -> pd.Series:
    times = np.sort(np.unique(t[event]))
    surv, s = [], 1.0
    for u in times:
        at_risk = (t >= u).sum()
        d = ((t == u) & event).sum()
        s *= 1 - d / at_risk
        surv.append(s)
    return pd.Series(surv, index=times)


def main() -> None:
    setup()
    d = load()
    s = build_songs(d)
    s = s[(s["year"] >= 1959) & (s["year"] < d["chart_week"].max().year)]    # full years only
    md = []

    # ---- 1. Front-loading over time -----------------------------------------------------------
    hits = s[s["peak"] <= 10]
    yr = pd.DataFrame({
        "Top-10 hits debuting in the Top 10 (%)": 100 * hits.groupby("year")["debut_pos"].apply(lambda x: (x <= 10).mean()),
        "Median weeks to peak (Top-10 hits)": hits.groupby("year")["weeks_to_peak"].median(),
        "Songs charting per year": s.groupby("year").size(),
        "Median weeks in Top 40 (Top-40 hits)": s[s["peak"] <= 40].groupby("year")["top40_weeks"].median(),
    })
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, col, c in zip(axes.flat, yr.columns, [HIGHLIGHT, PALETTE[0], PALETTE[2], PALETTE[1]]):
        ax.plot(yr.index, yr[col], color=c, lw=1.2, alpha=0.5)
        ax.plot(yr.index, yr[col].rolling(5, center=True, min_periods=3).mean(), color=c, lw=2.6)
        for a, _, _ in ERAS[1:]:
            ax.axvline(a, color="#999", ls=":", lw=1)
        ax.set_title(col, fontsize=11)
    fig.suptitle("Hits arrive faster, and there are more of them", fontweight="bold")
    f1 = savefig(fig, "01_front_loading", SRC + ". Dotted lines: SoundScan (1991), downloads (2005), streaming (2013).")

    # ---- 2. Survival in the Top 40 --------------------------------------------------------------
    t40 = s[s["peak"] <= 40].copy()
    t40["event"] = ~t40["censored"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    med = {}
    for _, _, name in ERAS:
        e = t40[t40["era"] == name]
        km = kaplan_meier(e["top40_weeks"].values, e["event"].values)
        km = pd.concat([pd.Series([1.0], index=[0]), km])
        ax.step(km.index, km.values, where="post", color=ERA_COL[name], lw=2.4, label=f"{name} (n={len(e):,})")
        med[name] = float(km.index[np.argmax(km.values <= 0.5)]) if (km.values <= 0.5).any() else np.nan
    ax.axhline(0.5, color="#999", ls="--", lw=1)
    ax.set_xlim(0, 40)
    ax.set_xlabel("Weeks spent in the Top 40")
    ax.set_ylabel("Share of Top-40 hits still in the Top 40")
    ax.set_title("Kaplan-Meier survival in the Top 40, by era")
    ax.legend()
    f2 = savefig(fig, "02_survival_by_era", SRC)

    # Cox PH: hazard of leaving the Top 40
    X = pd.get_dummies(t40["era"])[[e[2] for e in ERAS[1:]]].astype(float)
    X["Peak position (per 10 places)"] = t40["peak"] / 10
    X["Debuted in Top 10"] = (t40["debut_pos"] <= 10).astype(float)
    X["log(1 + prior Top-40 hits)"] = np.log1p(t40["prior_hits"])
    cox = PHReg(t40["top40_weeks"].values, X.values, status=t40["event"].astype(int).values, ties="efron").fit()
    ci = cox.conf_int()
    cox_tab = pd.DataFrame({"Variable": X.columns, "Hazard ratio": np.exp(cox.params),
                            "95% CI low": np.exp(ci[:, 0]), "95% CI high": np.exp(ci[:, 1]), "p": cox.pvalues})

    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(len(cox_tab))[::-1]
    ax.errorbar(cox_tab["Hazard ratio"], y, xerr=[cox_tab["Hazard ratio"] - cox_tab["95% CI low"],
                cox_tab["95% CI high"] - cox_tab["Hazard ratio"]], fmt="o", color=PALETTE[0], capsize=4)
    ax.axvline(1, color="#333", lw=1)
    ax.set_yticks(y, cox_tab["Variable"])
    ax.set_xscale("log")
    ax.set_xlabel("Hazard ratio of dropping out of the Top 40 (>1 = burns out faster; ref. era: radio & sales)")
    ax.set_title("Cox proportional hazards")
    f3 = savefig(fig, "03_cox", SRC)

    # ---- 3. Predictability from the first three weeks --------------------------------------------
    feats = ["w1", "w2", "w3", "prior_hits"]
    done = t40[t40["event"]].copy()
    done["y"] = np.log1p(done["top40_weeks"])
    r2 = []
    for _, _, name in ERAS:
        e = done[done["era"] == name]
        pred = cross_val_predict(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0),
                                 e[feats], e["y"], cv=KFold(5, shuffle=True, random_state=0))
        r2.append({"Era": name, "Songs": len(e), "Out-of-fold R²": r2_score(e["y"], pred)})
    r2 = pd.DataFrame(r2)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(r2["Era"], r2["Out-of-fold R²"], color=[ERA_COL[e] for e in r2["Era"]])
    for i, v in enumerate(r2["Out-of-fold R²"]):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center")
    ax.set_ylabel("R² (5-fold, gradient boosting)")
    ax.set_title("How much of a hit's Top-40 life is decided in its first 3 weeks?")
    ax.tick_params(axis="x", labelsize=9)
    f4 = savefig(fig, "04_predictability", SRC)

    one_wk = t40.groupby("era")["top40_weeks"].apply(lambda x: 100 * (x <= 1).mean())
    long_run = t40.groupby("era")["top40_weeks"].apply(lambda x: 100 * (x >= 20).mean())

    # ---- Write-up ---------------------------------------------------------------------------------
    first, last = ERAS[0][2], ERAS[-1][2]
    early = yr.loc[1960:1989].mean()
    late = yr.loc[2015:].mean()
    md.append("### Headline numbers\n")
    md.append(f"- Top-10 hits that **debuted** in the Top 10: **{early.iloc[0]:.0f}%** (1960–89 avg) → **{late.iloc[0]:.0f}%** (2015+ avg).")
    md.append(f"- Median weeks from debut to peak for Top-10 hits: **{early.iloc[1]:.1f} → {late.iloc[1]:.1f}**.")
    md.append(f"- Songs charting per year: **{early.iloc[2]:.0f} → {late.iloc[2]:.0f}**.")
    md.append(f"- Median Top-40 life (Kaplan-Meier): **{med[first]:.0f} weeks** in the {first} era vs **{med[last]:.0f} weeks** in the {last} era.")
    md.append(f"- **One-week wonders:** {one_wk[first]:.0f}% of Top-40 entries lasted a single week in the {first} era vs "
              f"**{one_wk[last]:.0f}%** in the {last} era, yet **{long_run[last]:.0f}%** of streaming-era Top-40 hits stay 20+ weeks "
              f"(vs {long_run[first]:.0f}%). The chart has polarised into flash debuts and long-runners.")
    st = cox_tab.set_index("Variable").loc[last]
    md.append(f"- Cox model: holding peak, debut and artist track record fixed, a streaming-era hit's hazard of leaving the Top 40 is "
              f"**{st['Hazard ratio']:.2f}×** the radio-era baseline (95% CI {st['95% CI low']:.2f}–{st['95% CI high']:.2f}).")
    md.append(f"- Share of Top-40 life predictable from the first three weeks: R² **{r2['Out-of-fold R²'].iloc[0]:.2f}** ({first}) → "
              f"**{r2['Out-of-fold R²'].iloc[-1]:.2f}** ({last}).\n")
    md.append("### Cox proportional hazards (leaving the Top 40)\n")
    md.append(md_table(cox_tab, "{:.3f}"))
    md.append("\n### Predictability by era\n")
    md.append(md_table(r2, "{:.3f}"))
    md.append("\n### Figures\n")
    for f, cap in [(f1, "Front-loading over time"), (f2, "Survival in the Top 40"), (f3, "Cox hazard ratios"),
                   (f4, "Predictability from the first 3 weeks")]:
        md.append(f"**{cap}**\n\n![{cap}]({f})\n")
    write_results("\n".join(md))
    print("done")


if __name__ == "__main__":
    main()
