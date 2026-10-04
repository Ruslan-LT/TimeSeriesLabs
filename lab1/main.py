"""
Лабораторна робота №1. Отримання та підготовка Time Series. Варіант 3 ІО-36 Літвінов Руслан.

"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st

# ----------------------------- Параметри ------------------------------------
SEED = 42
N = 1000
REAL_FILE = "Oschadbank.xlsx"
REAL_COLUMNS = ["Купівля", "Продаж", "Нбу"]
BINS = 40

# Закони похибки (випадкова складова)
EXP_SCALE = 2.0
NORM_MU, NORM_SIGMA = 0.0, 2.0

# Закони тренду (невипадкова складова)
CONST_C = 10.0
QUAD_A, QUAD_B, QUAD_C = 0.0001, 0.02, 5.0   # y = a*t^2 + b*t + c

rng = np.random.default_rng(SEED)
t = np.arange(N)


# -------------------  Модель випадкової величини --------------------------
def noise_exponential(n):
    return rng.exponential(scale=EXP_SCALE, size=n)


def noise_normal(n):
    return rng.normal(loc=NORM_MU, scale=NORM_SIGMA, size=n)


# -------------------  Модель ідеального тренду ----------------------------
def trend_constant(t):
    return np.full_like(t, CONST_C, dtype=float)


def trend_quadratic(t):
    return QUAD_A * t**2 + QUAD_B * t + QUAD_C


# -------------------  Адитивна модель: --------------------
def additive(trend, noise):
    return trend + noise


# -------------------  Статистичні характеристики --------------------------
def characteristics(x):
    x = np.asarray(x, dtype=float)
    return {
        "n": len(x),
        "mean (M)": x.mean(),
        "variance (D)": x.var(ddof=1),
        "std (sigma)": x.std(ddof=1),
        "min": x.min(),
        "max": x.max(),
    }


def print_stats(title, x):
    print(f"\n--- {title} ---")
    for k, v in characteristics(x).items():
        print(f"{k:>14}: {v:.4f}" if k != "n" else f"{k:>14}: {v}")


def hist(ax, x, title, pdf=None):
    ax.hist(x, bins=BINS, density=True, alpha=0.6, edgecolor="black")
    if pdf is not None:
        xs = np.linspace(np.min(x), np.max(x), 300)
        ax.plot(xs, pdf(xs), "r-", lw=2, label="теоретична щільність")
        ax.legend()
    ax.set_title(title)
    ax.grid(alpha=0.3)


# ------------------- моделі та вибірки ------------------------
def run_synthetic():
    e = noise_exponential(N)
    g = noise_normal(N)
    exp_pdf = lambda x: st.expon.pdf(x, scale=EXP_SCALE)
    norm_pdf = lambda x: st.norm.pdf(x, NORM_MU, NORM_SIGMA)


    print_stats("Експоненційна похибка", e)
    print_stats("Нормальна похибка", g)


    print("\n=== Верифікація випадкових величин ===")
    rows = [
        ("Експоненційна", EXP_SCALE, EXP_SCALE**2, e),
        ("Нормальна", NORM_MU, NORM_SIGMA**2, g),
    ]
    for name, m_t, d_t, x in rows:
        print(f"{name}: M теор={m_t:.4f} / вибірк={x.mean():.4f}; "
              f"D теор={d_t:.4f} / вибірк={x.var(ddof=1):.4f}")
    print("KS-тест (p>0.05 => узгоджується із законом):",
          f"exp p={st.kstest(e, st.expon(scale=EXP_SCALE).cdf).pvalue:.3f},",
          f"norm p={st.kstest(g, st.norm(NORM_MU, NORM_SIGMA).cdf).pvalue:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    hist(ax[0], e, "Експоненційна похибка", exp_pdf)
    hist(ax[1], g, "Нормальна похибка", norm_pdf)
    fig.tight_layout()
    fig.savefig("fig_noise_hist.png", dpi=150)


    trends = {"константа": trend_constant(t), "квадратичний": trend_quadratic(t)}
    noises = {"експоненційна": e, "нормальна": g}
    n_t = {"експоненційна": (EXP_SCALE, EXP_SCALE**2),
           "нормальна": (NORM_MU, NORM_SIGMA**2)}

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 7))
    print("\n=== Адитивні моделі ===")
    for i, (tn, tr) in enumerate(trends.items()):
        for j, (nn, nz) in enumerate(noises.items()):
            x = additive(tr, nz)
            print_stats(f"Тренд: {tn} + похибка: {nn}", x)


            resid = x - tr
            m_t, d_t = n_t[nn]
            print(f"  M теор = {tr.mean() + m_t:.4f}, вибірк = {x.mean():.4f}")
            print(f"  D залишків теор = {d_t:.4f}, вибірк = {resid.var(ddof=1):.4f}")

            ax = axes[i, j]
            ax.plot(t, x, lw=0.7, label="вибірка")
            ax.plot(t, tr, "r-", lw=2, label="ідеальний тренд")
            ax.set_title(f"{tn} + {nn}")
            ax.legend()
            ax.grid(alpha=0.3)
            hist(axes2[i, j], x, f"Гістограма: {tn} + {nn}")
    fig.tight_layout()
    fig.savefig("fig_additive_series.png", dpi=150)
    fig2.tight_layout()
    fig2.savefig("fig_additive_hist.png", dpi=150)


# ------------------- реальні дані -------------------------------
def to_number(s):
    return pd.to_numeric(s.astype(str).str.replace(",", ".").str.replace(" ", ""),
                         errors="coerce")


def load_real():
    if not os.path.exists(REAL_FILE):
        print(f"\n[!] Файл '{REAL_FILE}' не знайдено - реальні дані пропущено.")
        return None
    df = pd.read_excel(REAL_FILE)
    print("\nКолонки файлу:", list(df.columns))


    df["dt"] = pd.to_datetime(df["Дата"].astype(str) + " " + df["Час"].astype(str),
                              format="%d.%m.%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["dt"]).sort_values("dt")
    df = df.drop_duplicates(subset="Дата", keep="last").set_index("dt")

    num = df.drop(columns=["Дата", "Час"]).apply(to_number)
    num = num.dropna(axis=1, how="all")
    cols = REAL_COLUMNS or list(num.columns[:3])
    out = num[cols].dropna(how="all")
    print(f"Реальні дані: {len(out)} спостережень, "
          f"{out.index.min().date()} - {out.index.max().date()}")
    return out


def run_real():
    data = load_real()
    if data is None:
        return
    fig, axes = plt.subplots(len(data.columns), 2,
                             figsize=(12, 3.2 * len(data.columns)), squeeze=False)
    for k, col in enumerate(data.columns):
        x = data[col].dropna().to_numpy()
        tt = np.arange(len(x))
        print_stats(f"Реальні дані: {col}", x)

        for deg in (1, 2):
            coef = np.polyfit(tt, x, deg)
            fit = np.polyval(coef, tt)
            r2 = 1 - np.sum((x - fit) ** 2) / np.sum((x - x.mean()) ** 2)
            print(f"  тренд deg={deg}: коеф.={np.round(coef, 6)}, R2={r2:.4f}")
        coef2 = np.polyfit(tt, x, 2)
        axes[k, 0].plot(tt, x, lw=0.8, label=col)
        axes[k, 0].plot(tt, np.polyval(coef2, tt), "r-", lw=2, label="тренд (deg=2)")
        axes[k, 0].legend()
        axes[k, 0].grid(alpha=0.3)
        hist(axes[k, 1], x, f"Гістограма: {col}")
    fig.tight_layout()
    fig.savefig("fig_real.png", dpi=150)


if __name__ == "__main__":
    run_synthetic()
    run_real()
    plt.show()