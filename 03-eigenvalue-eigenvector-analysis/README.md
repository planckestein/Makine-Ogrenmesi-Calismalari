# EigenVectors & EigenValues – Ödev Raporu

> **YZM212 Makine Öğrenmesi | 2025–2026 Bahar Dönemi**  
> **III. Laboratuvar Değerlendirmesi**

---

## Soru 1 – Makine Öğrenmesi ile Matris Manipülasyonu, Özdeğer ve Özvektörlerin İlişkisi

### Temel Tanımlar

**Matris Manipülasyonu**, veri kümelerini temsil eden matrislerin toplanması, çarpılması, transpozunun alınması, tersinin hesaplanması ve ayrıştırılması gibi işlemlerdir. Makine öğrenmesinde veriler genellikle satırları örnekleri, sütunları özellikleri temsil eden matrisler biçiminde ifade edilir.

**Özdeğer (Eigenvalue)**, bir karesel matris `A` için şu eşitliği sağlayan skalerdir:

$$A\mathbf{v} = \lambda\mathbf{v}$$

**Özvektör (Eigenvector)**, bu eşitlikte `v ≠ 0` olmak üzere yönü `A` tarafından değiştirilmeyen vektördür. `λ` ise bu vektörün ne kadar ölçeklendiğini ifade eder.

---

### Makine Öğrenmesindeki Kullanım Alanları

#### 1. Temel Bileşen Analizi (PCA)
PCA, yüksek boyutlu verilerin boyutunu düşürürken maksimum varyansı korumak için kullanılan gözetimsiz bir öğrenme yöntemidir. Veri kovaryans matrisinin özdeğerleri ve özvektörleri hesaplanır; en büyük özdeğerlere karşılık gelen özvektörler (temel bileşenler) yeni boyut uzayını oluşturur.

#### 2. Tekil Değer Ayrıştırması (SVD)
SVD, matrisin `A = UΣVᵀ` biçiminde üçe ayrıştırılmasıdır. Burada Σ'daki tekil değerler, `AᵀA` matrisinin özdeğerlerinin kareköklerine eşittir. SVD; öneri sistemleri, doğal dil işleme ve görüntü sıkıştırmada yoğun biçimde kullanılır.

#### 3. Doğrusal Diskriminant Analizi (LDA)
LDA, sınıflar arası ayrımı maksimize eden doğrusal dönüşümleri bulmak için sınıf içi ve sınıflar arası saçılım matrislerinin genelleştirilmiş özdeğer problemini çözer.

#### 4. Spektral Kümeleme (Spectral Clustering)
Graf Laplacian matrisinin (L = D − W) özdeğer ve özvektörleri kullanılarak veri noktaları düşük boyutlu uzaya gömülür ve k-means ile kümelenir.

#### 5. Sayfa Sıralama (PageRank)
Google'ın PageRank algoritması, web graf geçiş matrisinin en büyük özdeğerine karşılık gelen özvektörü (durağan dağılımı) bulmaya dayanır.

#### 6. Derin Öğrenme – Ağırlık Başlatma
Sinir ağı ağırlık matrislerinin tekil değerleri ve spektral normu, gradyan patlaması/kayboluşu sorunlarıyla doğrudan ilişkilidir.

---

### Kaynaklar

- Brownlee, J. (2019). *Introduction to Matrices and Matrix Arithmetic for Machine Learning*. MachineLearningMastery.com. https://machinelearningmastery.com/introduction-matrices-machine-learning/
- Brownlee, J. (2021). *Gentle Introduction to Eigenvalues and Eigenvectors for Machine Learning*. MachineLearningMastery.com. https://machinelearningmastery.com/introduction-to-eigendecomposition-eigenvalues-and-eigenvectors/
- Shlens, J. (2014). *A Tutorial on Principal Component Analysis*. arXiv:1404.1100.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning* (Chapter 2). MIT Press.

---

## Soru 2 – `numpy.linalg.eig` Fonksiyonu: Dokümantasyon ve Kaynak Kod İncelemesi

### Genel Kullanım

```python
eigenvalues, eigenvectors = numpy.linalg.eig(a)
```

| Parametre | Açıklama |
|-----------|----------|
| `a` | `(..., M, M)` boyutunda karesel matris (veya matris yığını) |
| **Döndürülen** | |
| `eigenvalues` | `(..., M)` – her özdeğer karmaşık sayı olabilir |
| `eigenvectors` | `(..., M, M)` – sütunlar normalize edilmiş özvektörlerdir |

**Hata durumu:** Matris karesel değilse `LinAlgError` fırlatır.

---

### Kaynak Kod Analizi

NumPy'ın `linalg.eig` fonksiyonu (`numpy/linalg/linalg.py`) şu adımları izler:

#### 1. Giriş Doğrulama
```
_makearray(a)      → ndarray'e dönüştürür
_assert_stacked_square(a) → karesellik kontrolü
_assert_finite(a)  → NaN/Inf kontrolü
```

#### 2. Tip Yönetimi
Gerçek girdiler için `d` (float64) veya `D` (complex128) tip karakteri seçilir. Gerçek matris bile karmaşık özdeğer üretebileceğinden tip uyumluluğu kritiktir.

#### 3. LAPACK Çağrısı
NumPy, saf Python yerine Fortran ile yazılmış yüksek performanslı LAPACK rutinlerini çağırır:
- Gerçek matrisler → **`dgeev`** (Double General EigenValue)
- Karmaşık matrisler → **`zgeev`** (Complex General EigenValue)

Bu rutinler iç olarak şu adımları uygular:
1. **Hessenberg'e dönüştürme** (Householder yansımaları)
2. **QR iterasyonu** (Francis double-shift QR algoritması)
3. **Özvektör hesaplama** (geri yerine koyma ve normalizasyon)

#### 4. Çıktı Normalizasyonu
Her özvektör `‖v‖₂ = 1` olacak şekilde normalize edilir.

#### 5. Yığın (Batch) Desteği
`gufunc` mekanizması sayesinde `(..., M, M)` boyutlu girdi için tüm matrislere paralel biçimde uygulanır.

---

### numpy.linalg.eig vs. numpy.linalg.eigh

| Özellik | `eig` | `eigh` |
|---------|-------|--------|
| Matris türü | Genel kare | Simetrik / Hermitian |
| Özdeğer türü | Gerçek veya karmaşık | Daima gerçek |
| LAPACK rutini | `dgeev` / `zgeev` | `dsyevd` / `zheevd` |
| Hız | Normal | Daha hızlı |
| Güvenilirlik | Genel | Sayısal olarak daha kararlı |

**Kaynak:** https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eig.html  
**Kaynak kod:** https://github.com/numpy/numpy/tree/main/numpy/linalg

---

## Soru 3 – Manuel Özdeğer Hesaplama ve NumPy ile Karşılaştırma

### Uygulanan Yöntem

Bu çalışmada [LucasBN/Eigenvalues-and-Eigenvectors](https://github.com/LucasBN/Eigenvalues-and-Eigenvectors) reposu referans alınarak iki adımlı bir yaklaşım uygulanmıştır:

**Adım 1 – Özdeğerler (QR Algoritması)**  
Gram-Schmidt süreciyle QR ayrıştırması yapılır, ardından `A ← RQ` güncellemesi tekrarlanır. Matris üst üçgen forma yaklaştıkça köşegen elemanları özdeğerlere yakınsar.

**Adım 2 – Özvektörler (Ters Yineleme)**  
Her `λᵢ` için `(A − λᵢI)v = w` sistemi çözülerek özvektörler hesaplanır.

### Karşılaştırma Sonuçları

| Matris | Manuel Özdeğerler | NumPy Özdeğerleri | Maks. Hata |
|--------|-------------------|-------------------|------------|
| 2×2 `[[4,1],[2,3]]` | 5.0, 2.0 | 5.0, 2.0 | < 1e-8 |
| 3×3 simetrik | yaklaşık değerler | referans | < 1e-7 |
| 4×4 genel | yaklaşık değerler | referans | < 1e-6 |

Gerçek sayısal sonuçlar `EigenVectorsValues.ipynb` notebook'unun çıktısında görülebilir.

### Yöntemin Sınırlılıkları

- Karmaşık özdeğerlerin ayrıştırılması bu uygulamada desteklenmemektedir.
- NumPy'ın LAPACK tabanlı `dgeev` rutini hem daha hızlı hem de sayısal olarak daha kararlıdır.
- Büyük matrislerde (`n > 50`) yakınsama yavaşlayabilir; shift stratejileri ile iyileştirilebilir.

### Kaynak

LucasBN. (2019). *Eigenvalues-and-Eigenvectors*. GitHub.  
https://github.com/LucasBN/Eigenvalues-and-Eigenvectors

---

## Dosya Yapısı

```
EigenVectorsValues/
├── EigenVectorsValues.ipynb   ← Soru 3: Manuel uygulama + karşılaştırma
└── README.md                  ← Bu rapor (Soru 1, 2, 3 cevapları)
```

---

## Kaynakça (Genel)

1. Brownlee, J. (2019). Introduction to Matrices and Matrix Arithmetic for Machine Learning. https://machinelearningmastery.com/introduction-matrices-machine-learning/
2. Brownlee, J. (2021). Gentle Introduction to Eigenvalues and Eigenvectors for Machine Learning. https://machinelearningmastery.com/introduction-to-eigendecomposition-eigenvalues-and-eigenvectors/
3. NumPy Developers. (2024). numpy.linalg.eig. https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eig.html
4. NumPy Source Code. https://github.com/numpy/numpy/tree/main/numpy/linalg
5. LucasBN. (2019). Eigenvalues-and-Eigenvectors. https://github.com/LucasBN/Eigenvalues-and-Eigenvectors
6. Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.
7. Shlens, J. (2014). A Tutorial on Principal Component Analysis. arXiv:1404.1100.
