import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Gözlemlenen trafik verisi (1 dakikada geçen araç sayısı)
traffic_data = np.array([12, 15, 10, 8, 14, 11, 13, 16, 9, 12, 11, 14, 10, 15])

def negative_log_likelihood(lam, data):
    """Poisson dağılımı için negatif log-olabilirlik hesaplar.
    log(k!) terimi lambda'ya bağlı olmadığı için optimizasyonda ihmal edilir.
    """
    lam = np.atleast_1d(lam)[0]
    if lam <= 0:
        return np.inf
    n = len(data)
    nll = n * lam - np.sum(data) * np.log(lam)
    return nll

initial_guess = [1.0]
result = opt.minimize(negative_log_likelihood, initial_guess, args=(traffic_data,), bounds=[(0.001, None)])
lambda_mle = result.x[0]

print(f"Sayısal Tahmin (MLE lambda): {lambda_mle:.6f}")
print(f"Analitik Tahmin (Ortalama): {np.mean(traffic_data):.6f}")

x = np.arange(0, max(traffic_data) + 6)
pmf_values = poisson.pmf(x, lambda_mle)

plt.figure(figsize=(10, 6))
plt.hist(traffic_data, bins=np.arange(min(traffic_data)-0.5, max(traffic_data)+1.5, 1), density=True, alpha=0.6, label='Gerçek Veri Histogramı')
plt.plot(x, pmf_values, 'o-', label=f'Poisson PMF (lambda = {lambda_mle:.3f})')
plt.xlabel('1 Dakikadaki Araç Sayısı')
plt.ylabel('Olasılık')
plt.title('Trafik Verisi Histogramı ve Poisson PMF Karşılaştırması')
plt.legend()
plt.tight_layout()
plt.savefig('traffic_fit.png', dpi=200)
plt.show()

traffic_data_outlier = np.append(traffic_data, 200)
result_outlier = opt.minimize(negative_log_likelihood, initial_guess, args=(traffic_data_outlier,), bounds=[(0.001, None)])
print(f"Outlier öncesi lambda: {np.mean(traffic_data):.6f}")
print(f"Outlier sonrası lambda: {result_outlier.x[0]:.6f}")
