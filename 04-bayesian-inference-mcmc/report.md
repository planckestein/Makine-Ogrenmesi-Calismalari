# Bayesian Brightness Inference – Akademik Rapor

**Ders:** YZM212 Makine Öğrenmesi – 4. Laboratuvar Ödevi  
**Konu:** Uzak Bir Galaksinin Parlaklık Analizi  
**Dönem:** 2025–2026 Bahar  

---

## 1. Giriş

Astronomi, laboratuvar ortamında deney yapılabilen bir bilim değildir. Yıldızlara dokunmak, bir galaksiyi tekrar ölçüm için kontrol altında tutmak mümkün değildir. Elimizdeki tek şey teleskoplardan gelen gürültülü foton akışlarıdır. Bu kısıtlı bilgiden anlamlı sonuçlar çıkarmak, çok sağlam bir istatistiksel çerçeve gerektirir.

Bu çalışmada, söz konusu çerçeve olarak **Bayesyen çıkarım** ve onun pratik uygulaması olan **MCMC (Markov Chain Monte Carlo)** tercih edilmiştir. Temel soru şudur: 50 adet gürültülü gözlemden bir gök cisminin gerçek parlaklığını (μ) ve bu gözlemlerdeki ölçüm belirsizliğini (σ) ne kadar doğru tahmin edebiliriz?

---

## 2. Problem Tanımı

Gürültülü bir gözlem modelinde her gözlem $x_i$ şu şekilde oluşturulmuştur:

$$x_i = \mu_{\text{true}} + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma_{\text{true}}^2)$$

Burada:
- $\mu_{\text{true}} = 150.0$ — gerçek parlaklık (bize "verilmiş" kabul edilen ama bilinmediği varsayılan değer)
- $\sigma_{\text{true}} = 10.0$ — ölçüm gürültüsünün gerçek büyüklüğü
- $n = 50$ — toplam gözlem sayısı

Görevimiz, yalnızca $\{x_i\}$ veri kümesine bakarak $(\mu, \sigma)$ çiftini posterior dağılım biçiminde tahmin etmektir.

---

## 3. Teorik Arka Plan

### 3.1 Bayes Teoremi

Parametreler hakkındaki güncel bilgimizi (posterior) hesaplamak için:

$$P(\theta \mid D) = \frac{P(D \mid \theta)\, P(\theta)}{P(D)}$$

- **$P(\theta \mid D)$ – Posterior:** Veriyi gördükten sonra $\theta = (\mu, \sigma)$ hakkında ne düşündüğümüz.
- **$P(D \mid \theta)$ – Likelihood:** $\theta$ doğruysa bu veriyi gözlemleme olasılığı.
- **$P(\theta)$ – Prior:** Veriyi görmeden önce $\theta$ hakkındaki ön bilgimiz.
- **$P(D)$ – Evidence:** Normalizasyon sabiti; MCMC'de hesaplanması gerekmez.

### 3.2 Gaussian Likelihood

Her gözlemin bağımsız ve Normal dağılımlı olduğunu varsayarak log-likelihood:

$$\ln P(D \mid \mu, \sigma) = -\frac{1}{2} \sum_{i=1}^{n} \left[ \left(\frac{x_i - \mu}{\sigma}\right)^2 + \ln(2\pi\sigma^2) \right]$$

### 3.3 Uniform Prior

Parametre uzayı hakkında özel bir bilgimiz yoksa düz prior kullanılır:

$$P(\mu, \sigma) = \begin{cases} 1 & 0 < \mu < 300 \text{ ve } 0 < \sigma < 50 \\ 0 & \text{aksi halde} \end{cases}$$

Bu prior "geniş ama fiziksel olarak anlamlı" sınırlar çizer (negatif parlaklık ya da negatif gürültü anlamsızdır).

### 3.4 MCMC ve emcee

Posterior $P(\theta \mid D)$ analitik olarak hesaplanamadığı durumlarda MCMC kullanılır. `emcee` kütüphanesinin affine-invariant ensemble sampleri, birbirini bilgilendiren paralel walker'larla posterior uzayını verimli şekilde keşfeder. Her walker bağımsız bir Markov zinciri yürütür; ensemble olarak bakıldığında posterior dağılımından örneklenmiş bir nokta bulutuna ulaşılır.

---

## 4. Veri Üretimi

```python
true_mu    = 150.0
true_sigma = 10.0
n_obs      = 50
np.random.seed(42)
data = true_mu + true_sigma * np.random.randn(n_obs)
```

`seed=42` sayesinde sonuçlar tamamen tekrar üretilebilirdir. Veri, ortalaması 150 ve standart sapması 10 olan Normal dağılımdan 50 örnek içermektedir. Ölçüm gürültüsünün sinyal-gürültü oranı %6–7 civarındadır (`10/150 ≈ 0.067`).

---

## 5. Bayesyen Model

Üç temel fonksiyon tanımlandı:

**log_likelihood:** Gaussian varsayımı altında log P(D|θ).  
**log_prior:** Uniform prior; sınır dışındaki bölge −∞ değer alır.  
**log_probability:** log-posterior = log-prior + log-likelihood.

Sigma ≤ 0 için log-likelihood'da −∞ döndürerek fiziksel olmayan durumlar örnekleme dışı bırakılır.

---

## 6. MCMC Kurulumu

| Parametre | Değer | Gerekçe |
|---|---|---|
| Başlangıç noktası | [140, 5] | Gerçek değerden kasıtlı uzakta; sampler'ın kendi yolunu bulmasını test eder |
| Walker sayısı | 32 | emcee için önerilen minimum (ndim × 2)'nin üzerinde; 16× parametre sayısı |
| Toplam adım | 2000 | Posterior'ın yeterince keşfedilmesi için |
| Burn-in | 500 adım atılıyor | Başlangıç konumundan uzaklaşma süresi |
| Thinning | Her 15. örnek | Otokorelasyonu azaltmak için |

Burn-in sonrası kalan örnekler: $(2000 - 500) / 15 \times 32 \approx 3200$ bağımsız örnek.

---

## 7. Sonuçlar

### 7.1 Temel Senaryo (n=50, geniş prior)

| Parametre | Gerçek | Median | %16 | %84 | |Hata| |
|---|---|---|---|---|---|
| μ (Parlaklık) | 150.00 | **147.78** | 146.39 | 149.12 | 2.22 |
| σ (Gürültü) | 10.00 | **9.47** | 8.63 | 10.58 | 0.53 |

- **μ güven aralığı genişliği (%68):** 2.73
- **σ güven aralığı genişliği (%68):** 1.95

Her iki parametre de gerçek değere %2'nin altında bir hatayla kurtarılmıştır. Bu, Bayesyen yöntemin güçlü bir teyididir. Seed=42 ile üretilen spesifik veri setinde örnek ortalaması 150'nin biraz altına düştüğü için μ tahmini de hafif düşük çıkmıştır — bu, modelin bir hatası değil, veri gerçeğidir.

---

## 8. Prior Etkisi Analizi

**Senaryo:** μ için prior aralığı 100–110 olarak daraltıldı.

| Parametre | Gerçek | Dar Prior Median | |Hata| |
|---|---|---|---|
| μ | 150.00 | 109.43 | **40.57** |
| σ | 10.00 | 40.41 | **30.41** |

Bu sonuç, Bayesyen çıkarımın en kritik dersini göstermektedir:

**Prior, sampler'ın erişebileceği parametre uzayını tanımlar.** Gerçek μ=150 değeri (100, 110) aralığının tamamen dışındadır. Sampler ne kadar çalışırsa çalışsın bu bölgeye ulaşamaz; prior bariyer görevi görür.

Peki model bu durumla nasıl başa çıkar? **σ üzerinden telafi yapar.** Gözlemlerin ortalaması ~150 civarında olmasına rağmen sampler yalnızca μ≈109'u görebildiği için, bu büyük tutarsızlığı "veri çok gürültülüymüş" diye yorumlar ve σ'yı 40'ın üzerine çıkarır. Bu, **prior-likelihood çatışmasının** klasik bir örneğidir.

**Ders:** Yanlış veya çok kısıtlayıcı bir prior, gerçek değerden uzak ve fiziksel olarak anlamsız sonuçlara yol açar. Astronomide bu, yıldız kütlesi için yanlış bir evrimsel model kullanmak anlamına gelir.

---

## 9. Veri Miktarı Analizi

**Senaryo:** n=5 gözlem ile aynı MCMC çalıştırıldı.

| | n=50 | n=5 | Büyüme Faktörü |
|---|---|---|---|
| μ Median | 147.78 | 154.44 | — |
| σ Median | 9.47 | 8.87 | — |
| μ CI Genişliği | 2.73 | **8.49** | **3.1×** |
| σ CI Genişliği | 1.95 | **8.76** | **4.5×** |

n=50'den n=5'e düşünce her iki parametrenin güven aralığı yaklaşık 3–4.5 kat genişledi.

**İstatistiksel gerekçe:** Gaussian modelde, ortalamanın örneklem standart hatası $\text{SE}(\hat{\mu}) = \sigma / \sqrt{n}$ formülüyle ölçeklenir. n=50'den n=5'e geçerken $\sqrt{50}/\sqrt{5} \approx 3.16$ kat genişleme beklenir — gözlemlenen 3.1 katlık artış bu teoriyle mükemmel uyuşmaktadır.

Varyans tahmini için belirsizlik daha da dramatik artar çünkü σ tahmini, $\chi^2$ dağılımından türetilir ve bu dağılımın varyansı $n-1$'e bölünür. Az veriyle hem ortalamanın hem de dağılımın şeklini güvenle çıkarmak zorlaşır.

---

## 10. Accuracy / Precision Tartışması

### Neden μ, σ'dan daha hassas tahmin ediliyor?

Bu soru Bayesyen istatistiğin güzel bir özelliğine işaret eder.

**Bilgi-teorik perspektif:** n bağımsız Gaussian gözlemle, $\mu$'nun Fisher bilgisi $n/\sigma^2$ olurken $\sigma$'nın Fisher bilgisi $2n/\sigma^2$'dir. İlk bakışta σ daha fazla bilgi içeriyormuş gibi görünür, ancak bu simetrik değildir.

**Asıl neden:** Ortalama, gözlemlerin doğrudan aritmetik toplamına bağlıdır — bu, Fisher bilgisi bağlamında $\sqrt{n}$ ölçeklemesi anlamına gelir. Standart sapma ise kareli sapmalar üzerinden tahmin edilir; bu, dağılımın şekline çok daha hassas biçimde bağlıdır ve küçük veri setlerinde örneklem varyansı, gerçek varyansdan önemli ölçüde sapabilir.

**n=50 etkisi:** Bu çalışmada μ CI genişliği 2.73, σ CI genişliği 1.95 çıktı. İlginç biçimde n=50'de σ biraz daha dar görünüyor, ancak n=5'e düşüldüğünde σ'nın genişliği orantısız biçimde artıyor (4.5×, μ için 3.1× iken). Bu, az veri durumunda σ tahmininin ne kadar kırılgan olduğunu gösteriyor.

---

## 11. Corner Plot Yorumu

Temel senaryo corner plot'unda dikkat çekici birkaç nokta:

**1. Diyagonal dağılımlar (marjinal posteriorlar):** μ için yaklaşık simetrik, tek-tepeli bir dağılım görülür. σ için hafif sağa-çarpık bir dağılım beklenir — çünkü σ > 0 kısıtı soldaki kuyruğu keser.

**2. Çapraz dağılım (parametre korelasyonu):** μ ve σ arasındaki kesişim bölümünde elips şekli dikeyeye yakındır. Bu, iki parametrenin büyük ölçüde bağımsız olduğunu gösterir. Elipsin eğimi çok küçüktür; yani μ'nun büyümesi σ'yı sistematik biçimde etkilememektedir.

**3. Gerçek değer çizgileri (kırmızı):** Her iki parametre için gerçek değerler posterior yoğunluğunun en yüksek bölgesi içinde kalmaktadır. Posterior dağılım, doğru değerleri kuşatmaktadır.

**4. Dar prior corner plot'u:** 100–110 aralığına sıkışmış μ için posterior son derece dar ve gerçek değerden uzak bir noktaya yığılmıştır. σ'nın posteriorsu ise geniş ve yüksek değerlere kaymıştır — prior-likelihood uyumsuzluğunun sinyal grafiği.

---

## 12. Sonuç

Bu çalışma, MCMC tabanlı Bayesyen çıkarımın astronomik parametre tahmini için ne kadar güçlü ve esnek olduğunu somut örneklerle göstermiştir. Üç temel bulgu öne çıkmaktadır:

**1. Yeterli veriyle model doğrudan çalışır:** n=50 gözlemle μ ve σ, gerçek değerlerine %2'nin altında bir hatayla kurtarılmıştır.

**2. Prior çok kritiktir:** Gerçek değeri dışarıda bırakan bir prior, modelin veriyi "yanlış yorumlamasına" neden olur. σ'nın 40'a fırlaması, bu yanlış yorumun tam bir tazminat mekanizmasıdır.

**3. Veri azaldıkça belirsizlik hızla artar:** n=5'ten n=50'ye geçmek, μ belirsizliğini 3× düşürmektedir; bu, Bayesyen sonuçların veri toplamaya değer kılar.

Bayesyen çıkarım, belirsizliği gizlemek yerine sayısal olarak ölçer — astronominin ve pek çok uygulamalı bilimin ihtiyacı olan tam da budur.

---

*YZM212 Makine Öğrenmesi | 2025–2026 Bahar Dönemi*
