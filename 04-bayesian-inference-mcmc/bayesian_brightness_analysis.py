"""
Bayesian Brightness Inference – Uzak Galaksi Parlaklık Analizi
==============================================================
YZM212 Makine Öğrenmesi – 4. Laboratuvar Ödevi

Gürültülü astronomik gözlem verilerinden gerçek parlaklık (mu) ve
ölçüm belirsizliği (sigma) parametrelerini MCMC ile tahmin eder.

Çalıştırmak için:
    python bayesian_brightness_analysis.py

Çıktılar figures/ klasörüne kaydedilir.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Sunucu/headless ortamında render hatası vermemesi için
import matplotlib.pyplot as plt
import emcee
import corner

# ---------------------------------------------------------------------------
# 0. Klasör hazırlığı
# ---------------------------------------------------------------------------
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Sabitler ve sentetik veri üretimi
# ---------------------------------------------------------------------------
TRUE_MU    = 150.0   # Gerçek parlaklık (bilinmiyor gibi davranıyoruz)
TRUE_SIGMA = 10.0    # Gözlem gürültüsü / ölçüm belirsizliği
N_OBS      = 50      # Temel gözlem sayısı
SEED       = 42      # Tekrar üretilebilirlik için

np.random.seed(SEED)
data = TRUE_MU + TRUE_SIGMA * np.random.randn(N_OBS)

# ---------------------------------------------------------------------------
# 2. Bayesyen model fonksiyonları
# ---------------------------------------------------------------------------

def log_likelihood(theta, obs):
    """Gaussian log-likelihood. sigma <= 0 fiziksel değildir."""
    mu, sigma = theta
    if sigma <= 0:
        return -np.inf
    return -0.5 * np.sum(((obs - mu) / sigma)**2 + np.log(2.0 * np.pi * sigma**2))


def log_prior(theta, mu_low=0.0, mu_high=300.0, sigma_high=50.0):
    """Düz (uniform) prior: 0 < mu < 300, 0 < sigma < 50."""
    mu, sigma = theta
    if mu_low < mu < mu_high and 0.0 < sigma < sigma_high:
        return 0.0
    return -np.inf


def log_probability(theta, obs, mu_low=0.0, mu_high=300.0, sigma_high=50.0):
    """Log-posterior = log-prior + log-likelihood."""
    lp = log_prior(theta, mu_low, mu_high, sigma_high)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, obs)


# ---------------------------------------------------------------------------
# 3. MCMC yardımcı fonksiyonu
# ---------------------------------------------------------------------------

def run_mcmc(obs, initial, n_walkers=32, n_steps=2000,
             burn_in=500, thin=15,
             mu_low=0.0, mu_high=300.0, sigma_high=50.0,
             seed=SEED):
    """
    emcee.EnsembleSampler ile MCMC örneklemesi yapar.
    flat_samples, sampler ikilisini döndürür.
    """
    np.random.seed(seed)
    ndim = 2
    pos = np.array(initial) + 1e-4 * np.random.randn(n_walkers, ndim)

    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, log_probability,
        args=(obs, mu_low, mu_high, sigma_high)
    )
    sampler.run_mcmc(pos, n_steps, progress=False)

    flat_samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
    return flat_samples, sampler


def posterior_stats(flat_samples):
    """Her parametrenin median, %16, %84 değerlerini döndürür."""
    results = {}
    labels = ["mu", "sigma"]
    for i, name in enumerate(labels):
        med = np.median(flat_samples[:, i])
        lo  = np.percentile(flat_samples[:, i], 16)
        hi  = np.percentile(flat_samples[:, i], 84)
        results[name] = {"median": med, "p16": lo, "p84": hi}
    return results


# ---------------------------------------------------------------------------
# 4. Görselleştirme fonksiyonları
# ---------------------------------------------------------------------------

def plot_data(obs, true_mu, true_sigma, filename):
    """Sentetik veri histogramı + gerçek dağılım."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(obs, bins=12, density=True, alpha=0.6, color="steelblue",
            edgecolor="white", label="Gürültülü Gözlemler")

    x = np.linspace(obs.min() - 5, obs.max() + 5, 300)
    from scipy.stats import norm
    ax.plot(x, norm.pdf(x, true_mu, true_sigma),
            "r-", lw=2, label=f"Gerçek Dağılım N({true_mu}, {true_sigma}²)")

    ax.axvline(true_mu, color="red", linestyle="--", alpha=0.6, label=f"μ={true_mu}")
    ax.set_xlabel("Parlaklık (Flux Birimi)")
    ax.set_ylabel("Olasılık Yoğunluğu")
    ax.set_title("Sentetik Gözlem Verisi")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  [Kaydedildi] {filename}")


def plot_trace(sampler, burn_in, labels, filename):
    """Walker trace (zincir) grafikleri – burn-in sonrası."""
    chain = sampler.get_chain()  # (n_steps, n_walkers, ndim)
    n_steps_total = chain.shape[0]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(chain[:, :, i], alpha=0.35, lw=0.6, color="steelblue")
        ax.axvline(burn_in, color="red", linestyle="--", lw=1.5,
                   label="Burn-in sınırı")
        ax.set_ylabel(label, fontsize=12)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Adım (MCMC İterasyonu)")
    fig.suptitle("MCMC Trace Plot – Walker Zincirleri", fontsize=13)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  [Kaydedildi] {filename}")


def plot_corner(flat_samples, true_vals, labels, title, filename):
    """Corner plot – posterior dağılımı ve parametre korelasyonu."""
    fig = corner.corner(
        flat_samples,
        labels=labels,
        truths=true_vals,
        truth_color="red",
        quantiles=[0.16, 0.50, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 11},
        label_kwargs={"fontsize": 12},
        color="steelblue"
    )
    fig.suptitle(title, fontsize=13, y=1.02)
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Kaydedildi] {filename}")


def print_table(label, stats, true_mu=TRUE_MU, true_sigma=TRUE_SIGMA):
    """Sonuç tablosunu ekrana yazar."""
    print(f"\n{'─'*65}")
    print(f"  {label}")
    print(f"{'─'*65}")
    print(f"  {'Parametre':<12} {'Gerçek':>8} {'Median':>10} "
          f"{'%16':>10} {'%84':>10} {'|Hata|':>10}")
    print(f"{'─'*65}")

    mu_s  = stats["mu"]
    sig_s = stats["sigma"]

    print(f"  {'μ (Parlaklık)':<12} {true_mu:>8.2f} {mu_s['median']:>10.4f} "
          f"{mu_s['p16']:>10.4f} {mu_s['p84']:>10.4f} "
          f"{abs(mu_s['median']-true_mu):>10.4f}")
    print(f"  {'σ (Gürültü)':<12} {true_sigma:>8.2f} {sig_s['median']:>10.4f} "
          f"{sig_s['p16']:>10.4f} {sig_s['p84']:>10.4f} "
          f"{abs(sig_s['median']-true_sigma):>10.4f}")
    print(f"{'─'*65}")

    mu_ci  = mu_s['p84']  - mu_s['p16']
    sig_ci = sig_s['p84'] - sig_s['p16']
    print(f"  μ  güven aralığı genişliği (%68): {mu_ci:.4f}")
    print(f"  σ  güven aralığı genişliği (%68): {sig_ci:.4f}")


# ---------------------------------------------------------------------------
# 5. Senaryo 1 – Temel analiz (n=50, geniş prior)
# ---------------------------------------------------------------------------
print("\n" + "="*65)
print("  SENARYO 1: Temel Analiz  (n=50, geniş prior)")
print("="*65)

plot_data(data, TRUE_MU, TRUE_SIGMA,
          os.path.join(FIGURES_DIR, "01_synthetic_data.png"))

flat_base, sampler_base = run_mcmc(
    obs=data,
    initial=[140.0, 5.0],
    n_walkers=32, n_steps=2000,
    burn_in=500, thin=15
)

stats_base = posterior_stats(flat_base)
print_table("Temel Senaryo  (n=50, prior: 0<μ<300, 0<σ<50)", stats_base)

plot_trace(sampler_base, burn_in=500,
           labels=[r"$\mu$ (Parlaklık)", r"$\sigma$ (Gürültü)"],
           filename=os.path.join(FIGURES_DIR, "02_trace_base.png"))

plot_corner(flat_base,
            true_vals=[TRUE_MU, TRUE_SIGMA],
            labels=[r"$\mu$ (Parlaklık)", r"$\sigma$ (Gürültü)"],
            title="Corner Plot – Temel Senaryo (n=50)",
            filename=os.path.join(FIGURES_DIR, "03_corner_base.png"))


# ---------------------------------------------------------------------------
# 6. Senaryo 2 – Dar prior etkisi (100 < mu < 110)
# ---------------------------------------------------------------------------
print("\n" + "="*65)
print("  SENARYO 2: Dar Prior Etkisi  (100 < μ < 110)")
print("="*65)

flat_narrow, sampler_narrow = run_mcmc(
    obs=data,
    initial=[105.0, 5.0],
    n_walkers=32, n_steps=2000,
    burn_in=500, thin=15,
    mu_low=100.0, mu_high=110.0
)

stats_narrow = posterior_stats(flat_narrow)
print_table("Dar Prior Senaryosu  (100<μ<110)", stats_narrow)

plot_corner(flat_narrow,
            true_vals=[TRUE_MU, TRUE_SIGMA],
            labels=[r"$\mu$ (Parlaklık)", r"$\sigma$ (Gürültü)"],
            title="Corner Plot – Dar Prior (100 < μ < 110)",
            filename=os.path.join(FIGURES_DIR, "04_corner_narrow_prior.png"))

print("\n  [Yorum] Gerçek μ=150 değeri, prior aralığı (100-110) dışında kaldığı")
print("  için sampler bu bölgeyi hiç keşfedemez. σ tahmininin dramatik biçimde")
print("  büyümesi, modelin veri uyumsuzluğunu σ parametresi üzerinden")
print("  'telafi etmeye' çalışmasının sonucudur.")


# ---------------------------------------------------------------------------
# 7. Senaryo 3 – Az veri etkisi (n=5)
# ---------------------------------------------------------------------------
print("\n" + "="*65)
print("  SENARYO 3: Az Veri Etkisi  (n=5)")
print("="*65)

np.random.seed(SEED)
data5 = TRUE_MU + TRUE_SIGMA * np.random.randn(5)

flat_5, sampler_5 = run_mcmc(
    obs=data5,
    initial=[140.0, 5.0],
    n_walkers=32, n_steps=2000,
    burn_in=500, thin=15
)

stats_5 = posterior_stats(flat_5)
print_table("Az Veri Senaryosu  (n=5)", stats_5)

# Karşılaştırmalı posterior görselleştirmesi
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
param_names = [r"$\mu$ (Parlaklık)", r"$\sigma$ (Gürültü)"]
true_vals    = [TRUE_MU, TRUE_SIGMA]

for i, (ax, name, tv) in enumerate(zip(axes, param_names, true_vals)):
    ax.hist(flat_base[:, i], bins=40, density=True,
            alpha=0.55, color="steelblue", label="n=50")
    ax.hist(flat_5[:, i],    bins=40, density=True,
            alpha=0.55, color="tomato",    label="n=5")
    ax.axvline(tv, color="black", linestyle="--", lw=1.5,
               label=f"Gerçek = {tv}")
    ax.set_xlabel(name, fontsize=11)
    ax.set_ylabel("Olasılık Yoğunluğu")
    ax.set_title(f"Posterior: {name}")
    ax.legend(fontsize=9)

fig.suptitle("Posterior Karşılaştırması: n=50 vs n=5", fontsize=13)
fig.tight_layout()
comp_path = os.path.join(FIGURES_DIR, "05_posterior_comparison_n5_vs_n50.png")
fig.savefig(comp_path, dpi=150)
plt.close(fig)
print(f"  [Kaydedildi] {comp_path}")

# CI genişlik karşılaştırması
mu_ci_50  = stats_base["mu"]["p84"]  - stats_base["mu"]["p16"]
mu_ci_5   = stats_5["mu"]["p84"]     - stats_5["mu"]["p16"]
sig_ci_50 = stats_base["sigma"]["p84"] - stats_base["sigma"]["p16"]
sig_ci_5  = stats_5["sigma"]["p84"]    - stats_5["sigma"]["p16"]

print(f"\n  μ  CI genişliği: n=50 → {mu_ci_50:.4f}  |  n=5 → {mu_ci_5:.4f}")
print(f"  σ  CI genişliği: n=50 → {sig_ci_50:.4f}  |  n=5 → {sig_ci_5:.4f}")
print(f"\n  μ için belirsizlik {mu_ci_5/mu_ci_50:.1f}x arttı")
print(f"  σ için belirsizlik {sig_ci_5/sig_ci_50:.1f}x arttı")


# ---------------------------------------------------------------------------
# 8. Genel özet
# ---------------------------------------------------------------------------
print("\n" + "="*65)
print("  TÜM SENARYOLAR – ÖZET")
print("="*65)
print(f"  {'Senaryo':<35} {'μ median':>10} {'σ median':>10}")
print(f"  {'-'*55}")
print(f"  {'Temel (n=50, geniş prior)':<35} "
      f"{stats_base['mu']['median']:>10.4f} "
      f"{stats_base['sigma']['median']:>10.4f}")
print(f"  {'Dar prior (100<μ<110)':<35} "
      f"{stats_narrow['mu']['median']:>10.4f} "
      f"{stats_narrow['sigma']['median']:>10.4f}")
print(f"  {'Az veri (n=5, geniş prior)':<35} "
      f"{stats_5['mu']['median']:>10.4f} "
      f"{stats_5['sigma']['median']:>10.4f}")
print(f"\n  Gerçek değerler: μ={TRUE_MU}, σ={TRUE_SIGMA}")
print(f"\n  Tüm görseller '{FIGURES_DIR}/' klasörüne kaydedildi.")
print("="*65 + "\n")
