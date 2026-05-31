# YZM212 Makine Öğrenmesi I Laboratuvar Ödevi Raporu

## 1. Bölüm: Teorik Temeller

Senaryo:
- Gizli durumlar: `S = {e, v}`
- Gözlemler: `O = {High, Low}`
- Başlangıç olasılığı: `P(e)=1.0`, `P(v)=0.0`

Geçiş olasılıkları:
- `P(e→e)=0.6`, `P(e→v)=0.4`
- `P(v→e)=0.2`, `P(v→v)=0.8`

Emisyon olasılıkları:
- `P(High|e)=0.7`, `P(Low|e)=0.3`
- `P(High|v)=0.1`, `P(Low|v)=0.9`

Gözlem dizisi:

```text
[High, Low]
```

### Adım 1: t = 1 için başlangıç
İlk gözlem `High` olduğuna göre:

- `δ1(e) = P(e) * P(High|e) = 1.0 * 0.7 = 0.7`
- `δ1(v) = P(v) * P(High|v) = 0.0 * 0.1 = 0.0`

Sonuç:

```text
δ1(e)=0.7
δ1(v)=0.0
```

### Adım 2: t = 2 için `Low` gözlemi

#### `e` durumuna gelme olasılığı

İki yol var:
- `e -> e`: `δ1(e) * P(e→e) = 0.7 * 0.6 = 0.42`
- `v -> e`: `δ1(v) * P(v→e) = 0.0 * 0.2 = 0.0`

En büyük yol:
- `max(0.42, 0.0) = 0.42`

Emisyon ile çarp:
- `δ2(e) = 0.42 * P(Low|e) = 0.42 * 0.3 = 0.126`

Geri izleme bilgisi:
- `ψ2(e) = e`

#### `v` durumuna gelme olasılığı

İki yol var:
- `e -> v`: `δ1(e) * P(e→v) = 0.7 * 0.4 = 0.28`
- `v -> v`: `δ1(v) * P(v→v) = 0.0 * 0.8 = 0.0`

En büyük yol:
- `max(0.28, 0.0) = 0.28`

Emisyon ile çarp:
- `δ2(v) = 0.28 * P(Low|v) = 0.28 * 0.9 = 0.252`

Geri izleme bilgisi:
- `ψ2(v) = e`

### Adım 3: Son durum seçimi

Son anda en yüksek olasılık:
- `δ2(e)=0.126`
- `δ2(v)=0.252`

Büyük olan:

```text
max(0.126, 0.252) = 0.252
```

Dolayısıyla son durum `v` olur.

### Adım 4: Geri izleme
- `t=2` anında en iyi durum `v`
- `ψ2(v)=e` olduğundan `t=1` anındaki durum `e`

### Nihai Sonuç
En olası fonem dizisi:

```text
e -> v
```

Yani gözlem dizisi `[High, Low]` için en olası açıklama **"EV" kelimesinin fonem akışı olan `e-v`** dizisidir.

---

## 2. Bölüm: Uygulama

Bu bölümde Python'da `hmmlearn` kullanılarak iki farklı HMM modeli tanımlanmıştır:
- `EV`
- `OKUL`

Kod dosyası:

```text
src/recognizer.py
```

### Yaklaşım
- Gözlemler iki kategoriye indirgenmiştir: `High` ve `Low`
- `EV` için 2 durumlu oyuncak HMM tanımlanmıştır
- `OKUL` için 4 durumlu oyuncak HMM tanımlanmıştır
- Yeni gelen test verisi için her iki modelin log-likelihood değeri hesaplanmıştır
- Daha yüksek skor veren model çıktı olarak seçilmiştir

### Örnek Test
Test gözlem dizisi:

```text
[High, Low]
```

Bu dizi kısa olduğu için genellikle `EV` modelinde daha anlamlı skor üretir.

---

## 3. Bölüm: Analiz ve Yorumlama

### Soru 1: Gürültü, Emisyon Olasılıklarını nasıl etkiler?
Gürültü, gözlemlerin gerçek durumu temsil etmesini bozar. Yani bir fonem normalde `High` ağırlıklı görünmesi gerekirken, gürültü yüzünden `Low` gibi ölçülebilir.

Bunun sonucu:
- Emisyon dağılımları daha bulanık hale gelir.
- Doğru durum ile doğru gözlem arasındaki bağ zayıflar.
- Modeller birbirinden daha az ayırt edilebilir hale gelir.
- Viterbi yanlış yol seçmeye daha açık olur.

Kısa konuşalım: gürültü, emisyon matrisine kum atar. Sinyal berraklığını düşürür, sınıflandırma güvenini kemirir.

### Soru 2: Neden büyük sistemlerde Viterbi yerine daha karmaşık yapılar tercih ediliyor?
Çünkü gerçek dünya oyuncak örnek değil.

Binlerce kelime, farklı konuşmacılar, aksanlar, hız farkı, arka plan gürültüsü, birleşik kelimeler ve sürekli konuşma olduğunda klasik HMM yaklaşımı tek başına yetersiz kalır.

Sebepler:
1. **Özellik temsili sınırlı**: HMM çoğu zaman elle tasarlanmış ya da basit özelliklere dayanır.
2. **Bağımlılıkları zayıf yakalar**: Uzun menzilli ilişkileri derin ağlar kadar iyi taşıyamaz.
3. **Ölçek problemi**: Kelime sayısı büyüdükçe modelleme karmaşıklığı artar.
4. **Veri bolluğundan tam yararlanamaz**: Deep learning modelleri büyük veriyle daha iyi performans verir.
5. **Uçtan uca öğrenme avantajı**: Modern sistemler ses sinyalinden metne kadar doğrudan öğrenebilir.

Bu yüzden güncel sistemlerde DNN-HMM hibritleri, RNN, LSTM, Transformer ve end-to-end ASR mimarileri tercih edilir.

---

## Genel Değerlendirme
Bu ödevde HMM mantığı küçük bir problem üzerinde gösterilmiştir. Eğitim için doğru yaklaşım bu. Ama üretim sistemine bunu çıplak halde koyarsan rakip seni siler. Çünkü gerçek konuşma tanıma problemi, iki durumlu örneklerden değil veri, varyasyon ve hesaplama ölçeğinden ibarettir.
