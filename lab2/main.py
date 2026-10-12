"""
Лабораторна робота №2. Статистичне навчання за Big Data Time Series. Варіант 3 ІО-36 Літвінов Руслан.
Рівень II (група вимог 1 + 2):
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42
N = 1000                       # обсяг синтетичної вибірки
FORECAST_FRAC = 0.5            # горизонт прогнозу = 0.5 * N
MAX_DEG = 6                    # максимальний степінь полінома при оптимізації

# Синтетична модель (як у Лр1): квадратичний тренд + похибка
QUAD_A, QUAD_B, QUAD_C = 0.0001, 0.02, 5.0
EXP_SCALE = 2.0
NORM_MU, NORM_SIGMA = 0.0, 2.0

# Аномалії
ANOM_FRAC = 0.05               # частка аномальних вимірів
ANOM_MAG = (8.0, 15.0)         # амплітуда викидів у одиницях sigma шуму
Z_THRESHOLD = 5.0              # поріг робастного z-критерію

# Реальні дані
REAL_FILE = "Oschadbank.xlsx"
REAL_COLUMNS = ["Купівля", "Продаж", "Нбу"]
BACKTEST_TRAIN_FRAC = 2 / 3    # навчання на 2/3, перевірка прогнозу на решті

rng = np.random.default_rng(SEED)


# =========================== МНК (LSM) ======================================
def design_matrix(t, deg, t0, scale):

    tau = (np.asarray(t, dtype=float) - t0) / scale
    return np.vander(tau, deg + 1, increasing=True)


class PolyLSM:

    def __init__(self, deg):
        self.deg = deg

    def fit(self, t, y):
        t, y = np.asarray(t, float), np.asarray(y, float)
        self.t0, self.scale = t.mean(), max(np.ptp(t), 1.0)
        X = design_matrix(t, self.deg, self.t0, self.scale)
        self.coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ self.coef
        self.n = len(y)
        self.rss = float(res @ res)
        self.sigma = np.sqrt(self.rss / max(self.n - self.deg - 1, 1))
        return self

    def predict(self, t):
        return design_matrix(t, self.deg, self.t0, self.scale) @ self.coef


# ======================= Показники якості моделі ============================
def quality(model, y):
    n, k = model.n, model.deg + 1
    rss = max(model.rss, 1e-12)
    tss = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "deg": model.deg,
        "RSS": rss,
        "R2": 1 - rss / tss,
        "AIC": n * np.log(rss / n) + 2 * k,
        "BIC": n * np.log(rss / n) + k * np.log(n),   # основний критерій
    }


def cv_rmse(t, y, deg, origins=(0.5, 0.6, 0.7, 0.8), horizon_frac=0.2):
    n = len(y)
    h = max(int(horizon_frac * n), 1)
    errs = []
    for o in origins:
        k = int(o * n)
        if k + 1 >= n or k < deg + 2:
            continue
        m = PolyLSM(deg).fit(t[:k], y[:k])
        errs.append(rmse(y[k:k + h], m.predict(t[k:k + h])))
    return float(np.mean(errs)) if errs else np.inf


def select_model(t, y, max_deg=MAX_DEG, criterion="BIC", verbose=True, label=""):
    t, y = np.asarray(t, float), np.asarray(y, float)
    table, best = [], None
    for d in range(0, max_deg + 1):
        m = PolyLSM(d).fit(t, y)
        q = quality(m, y)
        q["CV_RMSE"] = cv_rmse(t, y, d)
        table.append(q)
        if best is None or q[criterion] < best[1][criterion]:
            best = (m, q)
    if verbose:
        print(f"\n--- Вибір моделі {label} (критерій: {criterion}) ---")
        print(pd.DataFrame(table).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"=> Обрано степінь {best[0].deg}")
    return best[0], table


# ======================= Метрики точності ===================================
def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def mape(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.mean(np.abs((a - b) / a)) * 100)


# ======================= Група 2: аномалії ==================================
def trend_quad(t):
    return QUAD_A * np.asarray(t) ** 2 + QUAD_B * np.asarray(t) + QUAD_C


def make_noise(kind, n):
    if kind == "нормальна":
        return rng.normal(NORM_MU, NORM_SIGMA, n), NORM_SIGMA
    return rng.exponential(EXP_SCALE, n), EXP_SCALE   # sigma = scale


def inject_anomalies(y, sigma):
    y = y.copy()
    k = int(ANOM_FRAC * len(y))
    idx = rng.choice(len(y), size=k, replace=False)
    sign = rng.choice([-1, 1], size=k)
    amp = rng.uniform(*ANOM_MAG, size=k) * sigma
    y[idx] += sign * amp
    mask = np.zeros(len(y), dtype=bool)
    mask[idx] = True
    return y, mask


def detect_anomalies(t, y, deg, z_thr=Z_THRESHOLD, max_iter=10):
    t, y = np.asarray(t, float), np.asarray(y, float)
    keep = np.ones(len(y), dtype=bool)
    for _ in range(max_iter):
        m = PolyLSM(deg).fit(t[keep], y[keep])
        r = y - m.predict(t)
        q25, q50, q75 = np.percentile(r[keep], [25, 50, 75])
        s_up = max((q75 - q50) / 0.6745, 1e-9)
        s_low = max((q50 - q25) / 0.6745, 1e-9)
        dev = r - q50
        new_keep = np.where(dev >= 0, dev / s_up, -dev / s_low) <= z_thr
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return ~keep


def detection_report(true_mask, found_mask):
    tp = int(np.sum(true_mask & found_mask))
    fp = int(np.sum(~true_mask & found_mask))
    fn = int(np.sum(true_mask & ~found_mask))
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    return {"TP": tp, "FP": fp, "FN": fn, "precision": prec, "recall": rec}


# ======================= Група 1 + 2: синтетика =============================
def run_synthetic():
    t = np.arange(N)
    h = int(FORECAST_FRAC * N)
    t_f = np.arange(N, N + h)
    ideal_all_base = trend_quad(np.arange(N + h))

    for noise_kind in ("нормальна", "експоненційна"):
        print("\n" + "=" * 70)
        print(f"СИНТЕТИКА: квадратичний тренд + {noise_kind} похибка")
        print("=" * 70)

        noise, sigma = make_noise(noise_kind, N)
        noise_mean = NORM_MU if noise_kind == "нормальна" else EXP_SCALE
        y_clean = trend_quad(t) + noise
        ideal_all = ideal_all_base + noise_mean   # еталон для верифікації
        y_anom, true_mask = inject_anomalies(y_clean, sigma)
        print(f"Внесено аномалій: {true_mask.sum()} із {N}")

        m_dirty, _ = select_model(t, y_anom, label="(з аномаліями, до очищення)")

        found = detect_anomalies(t, y_anom, deg=m_dirty.deg if m_dirty.deg >= 2 else 2)
        rep = detection_report(true_mask, found)
        print("\nЯкість виявлення аномалій:",
              {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rep.items()})
        t_c, y_c = t[~found], y_anom[~found]
        print(f"Після очищення залишилось {len(y_c)} точок")

        m_clean, _ = select_model(t_c, y_c, label="(після очищення)")

        f_dirty = m_dirty.predict(t_f)
        f_clean = m_clean.predict(t_f)
        f_ideal = ideal_all[N:]
        print("\n=== Верифікація (порівняння з еталоном) ===")
        print(f"{'':28}{'RMSE (інтервал)':>18}{'RMSE (прогноз)':>18}")
        for name, m, f in (("до очищення", m_dirty, f_dirty),
                           ("після очищення", m_clean, f_clean)):
            print(f"{name:28}{rmse(m.predict(t), ideal_all[:N]):>18.4f}"
                  f"{rmse(f, f_ideal):>18.4f}")
        print(f"Залишки після очищення: sigma = {m_clean.sigma:.4f} (теор. {sigma:.4f})")
        print(f"Еталон = ідеальний тренд + M(шуму) = +{noise_mean}")
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax[0].plot(t, y_anom, lw=0.6, label="вибірка з аномаліями")
        ax[0].scatter(t[found], y_anom[found], c="r", s=14, zorder=3,
                      label=f"виявлені аномалії ({found.sum()})")
        ax[0].plot(np.arange(N + h), ideal_all, "k--", lw=1.5, label="еталон (тренд + M шуму)")
        ax[0].set_title(f"Виявлення аномалій. Квадратичний тренд + {noise_kind} похибка")
        ax[0].legend(); ax[0].grid(alpha=0.3)

        ax[1].plot(t_c, y_c, lw=0.6, color="gray", label="очищена вибірка")
        ax[1].plot(t, m_clean.predict(t), "g-", lw=2,
                   label=f"МНК, степінь {m_clean.deg}")
        ax[1].plot(t_f, f_clean, "r-", lw=2, label=f"прогноз на {h} кроків")
        ax[1].fill_between(t_f, f_clean - 2 * m_clean.sigma, f_clean + 2 * m_clean.sigma,
                           color="r", alpha=0.15, label="±2σ")
        ax[1].plot(np.arange(N + h), ideal_all, "k--", lw=1, label="еталон")
        ax[1].axvline(N, color="k", lw=0.8)
        ax[1].set_title("Статистичне навчання на очищених даних та екстраполяція")
        ax[1].legend(); ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"fig2_synthetic_{'norm' if noise_kind=='нормальна' else 'exp'}.png",
                    dpi=150)


# ======================= Група 1: реальні дані ==============================
def to_number(s):
    return pd.to_numeric(s.astype(str).str.replace(",", ".").str.replace(" ", ""),
                         errors="coerce")


def load_real():
    if not os.path.exists(REAL_FILE):
        print(f"\n[!] Файл '{REAL_FILE}' не знайдено - реальні дані пропущено.")
        return None
    df = pd.read_excel(REAL_FILE)
    df["dt"] = pd.to_datetime(df["Дата"].astype(str) + " " + df["Час"].astype(str),
                              format="%d.%m.%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["dt"]).sort_values("dt")
    df = df.drop_duplicates(subset="Дата", keep="last").set_index("dt")
    num = df.drop(columns=["Дата", "Час"]).apply(to_number)
    return num[REAL_COLUMNS].dropna(how="all")


def run_real():
    data = load_real()
    if data is None:
        return
    print("\n" + "=" * 70)
    print(f"РЕАЛЬНІ ДАНІ: {len(data)} спостережень, "
          f"{data.index.min().date()} - {data.index.max().date()}")
    print("(аномалії для реальних даних не моделюються - за умовою завдання)")
    print("=" * 70)

    fig, axes = plt.subplots(len(data.columns), 1, figsize=(12, 4 * len(data.columns)),
                             squeeze=False)
    for k, col in enumerate(data.columns):
        y = data[col].dropna().to_numpy()
        n = len(y)
        t = np.arange(n)
        h = int(FORECAST_FRAC * n)
        t_f = np.arange(n, n + h)

        model, _ = select_model(t, y, criterion="CV_RMSE", label=f"(реальні дані: {col})")
        forecast = model.predict(t_f)

        n_tr = int(BACKTEST_TRAIN_FRAC * n)
        t_tr, y_tr = t[:n_tr], y[:n_tr]
        t_te, y_te = t[n_tr:], y[n_tr:]
        bt_model, _ = select_model(t_tr, y_tr, criterion="CV_RMSE", verbose=False)
        bt_pred = bt_model.predict(t_te)
        naive = np.full_like(y_te, y_tr[-1])
        print(f"\nБектест ({col}): навчання {n_tr} точок, перевірка {len(y_te)} точок")
        print(f"  обрана модель: степінь {bt_model.deg}")
        print(f"  RMSE моделі = {rmse(y_te, bt_pred):.4f}, MAPE = {mape(y_te, bt_pred):.3f} %")
        print(f"  RMSE naive  = {rmse(y_te, naive):.4f}, MAPE = {mape(y_te, naive):.3f} %")
        print(f"Прогноз {col}: перше значення {forecast[0]:.3f}, "
              f"останнє (через {h} кроків) {forecast[-1]:.3f}")

        ax = axes[k, 0]
        ax.plot(t, y, lw=0.8, label=col)
        ax.plot(t, model.predict(t), "g-", lw=2, label=f"МНК, степінь {model.deg}")
        ax.plot(t_f, forecast, "r-", lw=2, label=f"прогноз на {h} кроків")
        ax.fill_between(t_f, forecast - 2 * model.sigma, forecast + 2 * model.sigma,
                        color="r", alpha=0.15, label="±2σ")
        ax.plot(t_te, bt_pred, "m--", lw=1.5, label="бектест-прогноз")
        ax.axvline(n_tr, color="m", lw=0.6, ls=":")
        ax.axvline(n, color="k", lw=0.8)
        ax.set_title(f"Реальні дані: {col}")
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig2_real.png", dpi=150)


if __name__ == "__main__":
    run_synthetic()
    run_real()
    plt.show()