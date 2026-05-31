# HMM Speech Recognition

Hidden Markov Model (HMM) mantığını kullanarak iki kelimeyi ayırt eden küçük bir konuşma tanıma sistemi kurmaktır.

## Problem Tanımı
Ödev iki parçadan oluşur:
1. **Teorik bölüm**: `EV` kelimesi için verilen HMM parametreleriyle Viterbi algoritması uygulanır.
2. **Uygulama bölümü**: Python ve `hmmlearn` ile `EV` ve `OKUL` kelimeleri için iki ayrı HMM tanımlanır ve yeni bir gözlem dizisinin hangi modele daha uygun olduğu log-likelihood ile belirlenir.

## Dizin Yapısı

```text
hmm-speech-recognition/
├── data/
├── src/
│   └── recognizer.py
├── report/
│   └── report.md
├── requirements.txt
└── README.md
```

## Veri
Bu projede gerçek ses dosyaları yerine, ses spektrumunu temsil eden basitleştirilmiş gözlemler kullanılmıştır:
- `High -> 0`
- `Low -> 1`

Bu yaklaşım, HMM mantığını göstermek için kullanılmış bir oyuncak örnektir.

## Yöntem
- `EV` için 2 durumlu HMM kuruldu.
- `OKUL` için 4 durumlu HMM kuruldu.
- Test gözlem dizisi her iki modele de verildi.
- `score()` ile log-likelihood hesaplandı.
- En yüksek skoru veren model tahmin olarak seçildi.

## Çalıştırma

```bash
pip install -r requirements.txt
python src/recognizer.py
```

## Beklenen Çıktı
Program test dizisinin hem `EV` hem de `OKUL` modeli altındaki skorunu yazdırır ve daha yüksek olanı seçer.

## Sonuç ve Tartışma
- `EV` modeli kısa gözlem dizilerinde daha uygun skor üretir.
- Gerçek konuşma tanıma sistemlerinde gözlem uzayı çok daha büyüktür.
- Bu yüzden HMM'ler eğitimsel olarak faydalı olsa da büyük ölçekli sistemlerde tek başına yeterli değildir.

Detaylı teorik çözüm ve analizler `report/report.md` dosyasındadır.
