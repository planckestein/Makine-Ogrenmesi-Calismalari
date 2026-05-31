# 🔭 Bayesian Brightness Inference
### YZM212 Makine Öğrenmesi – 4. Laboratuvar Ödevi

> Gürültülü astronomik gözlem verilerinden bir gök cisminin gerçek parlaklığını ve ölçüm belirsizliğini **MCMC tabanlı Bayesyen çıkarım** ile tahmin etme.

---

## Problem Tanımı

Uzak bir galaksinin gözlemi sırasında, teleskop sensöründeki ısıl gürültü, atmosferik dalgalanmalar ve kozmik toz bulutları ölçümleri bozar. Bu çalışmada:

- **μ (mu):** Gök cisminin gerçek parlaklığı  
- **σ (sigma):** Gözlem gürültüsünün standart sapması  

parametreleri, 50 adet gürültülü gözlemden Bayesyen yöntemlerle tahmin edilmektedir.

---

## Kullanılan Yöntem

| Bileşen | Detay |
|---|---|
| **Model** | Gaussian Likelihood + Uniform Prior |
| **Örnekleyici** | `emcee.EnsembleSampler` (affine-invariant MCMC) |
| **Walker Sayısı** | 32 |
| **MCMC Adımı** | 2000 |
| **Burn-in** | İlk 500 adım atılıyor |
| **Thinning** | Her 15 adımda bir örnek alınıyor |

---

## Sabitler

| Değişken | Değer | Açıklama |
|---|---|---|
| `true_mu` | 150.0 | Gerçek parlaklık (Ground Truth) |
| `true_sigma` | 10.0 | Gerçek ölçüm gürültüsü |
| `n_obs` | 50 | Temel gözlem sayısı |
| `seed` | 42 | Rastgele sayı üreteci sabit değeri |
| Prior (μ) | (0, 300) | Uniform prior aralığı |
| Prior (σ) | (0, 50) | Uniform prior aralığı |

---

## Nasıl Çalıştırılır

```bash
# 1. Bağımlılıkları yükle
pip install -r requirements.txt

# 2. Analiz scriptini çalıştır
python bayesian_brightness_analysis.py

# 3. (Opsiyonel) Notebook'u başlat
jupyter notebook Bayesian_Brightness_Inference.ipynb
```

---

## Üretilen Çıktılar

```
figures/
├── 01_synthetic_data.png          # Gözlem histogramı + gerçek dağılım
├── 02_trace_base.png              # MCMC walker trace grafikleri
├── 03_corner_base.png             # Corner plot – temel senaryo
├── 04_corner_narrow_prior.png     # Corner plot – dar prior senaryosu
└── 05_posterior_comparison_n5_vs_n50.png  # n=5 vs n=50 karşılaştırması
```

---

## Temel Bulgular

### Senaryo 1 – Temel Analiz (n=50, geniş prior)

| Parametre | Gerçek Değer | Median Tahmin | %16 | %84 | Mutlak Hata |
|---|---|---|---|---|---|
| μ (Parlaklık) | 150.00 | 147.78 | 146.39 | 149.12 | 2.22 |
| σ (Gürültü) | 10.00 | 9.47 | 8.63 | 10.58 | 0.53 |

> μ için %68 güven aralığı genişliği: **2.73**, σ için: **1.95**

### Senaryo 2 – Dar Prior (100 < μ < 110)

Model, gerçek değeri (μ=150) içermeyen bir aralığa zorlandığında μ tahmini ~109'a saplanır ve σ, bu uyumsuzluğu telafi etmek için ~40'a yükselir.

### Senaryo 3 – Az Veri (n=5)

| | n=50 | n=5 | Oran |
|---|---|---|---|
| μ CI Genişliği | 2.73 | 8.49 | **3.1×** |
| σ CI Genişliği | 1.95 | 8.76 | **4.5×** |

---

## Sonuç

Bayesyen çıkarım, astronomik parametre tahmini için güçlü ve güvenilir bir yöntemdir. **Prior seçimi** ve **veri miktarı**, posterior kalitesini doğrudan belirler. Gerçekçi prior aralıklarıyla 50 gözlem, hem parlaklık hem de gürültü parametrelerini yüksek hassasiyetle kurtarmaya yeterlidir.

---

*2025–2026 Bahar Dönemi | YZM212 Makine Öğrenmesi*
