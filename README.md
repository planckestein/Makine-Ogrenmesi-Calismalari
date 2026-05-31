# Makine Öğrenmesi Çalışmaları

Bu depo, makine öğrenmesi temellerini uygulamalı olarak öğrenmek ve Python ile modelleme pratiği kazanmak amacıyla hazırladığım çalışmaları içermektedir.

Çalışmalar; olasılıksal modelleme, parametre tahmini, lineer cebir uygulamaları ve Bayesçi çıkarım gibi makine öğrenmesinin temel konularına odaklanmaktadır. Her klasörde ilgili konuya ait kodlar, notebook dosyaları, raporlar ve gerekli açıklamalar yer almaktadır.

## İçerik

| Klasör                               | Konu                           | Kullanılan / İncelenen Kavramlar                                                         |
| ------------------------------------ | ------------------------------ | ---------------------------------------------------------------------------------------- |
| `01-hmm-speech-recognition`          | HMM Tabanlı Konuşma Tanıma     | Hidden Markov Models, gözlem dizileri, geçiş olasılıkları, log-likelihood                |
| `02-mle-poisson-traffic-model`       | MLE ile Trafik Modelleme       | Maximum Likelihood Estimation, Poisson dağılımı, parametre tahmini, aykırı değer analizi |
| `03-eigenvalue-eigenvector-analysis` | Özdeğer ve Özvektör Analizleri | Lineer cebir, özdeğer-özvektör hesabı, PCA, SVD, LDA bağlantıları                        |
| `04-bayesian-inference-mcmc`         | Bayesçi Çıkarım ve MCMC        | Bayesian inference, posterior dağılım, belirsizlik analizi, MCMC yaklaşımı               |

## Kullanılan Teknolojiler

* Python
* Jupyter Notebook
* NumPy
* SciPy
* Matplotlib

## Depo Yapısı

```text
Makine-Ogrenmesi-Calismalari/
├── 01-hmm-speech-recognition/
│   ├── src/
│   ├── report/
│   ├── README.md
│   └── requirements.txt
│
├── 02-mle-poisson-traffic-model/
│   ├── notebooks/
│   ├── src/
│   ├── images/
│   ├── report/
│   └── README.md
│
├── 03-eigenvalue-eigenvector-analysis/
│   ├── EigenVectorsValues.ipynb
│   └── README.md
│
├── 04-bayesian-inference-mcmc/
│   └── README.md
│
└── README.md
```

## Çalışmaların Amacı

Bu depo, makine öğrenmesi konularını yalnızca teorik olarak incelemekten ziyade, matematiksel temellerini Python ile uygulamalı şekilde anlamayı hedeflemektedir.

Bu süreçte özellikle şu beceriler üzerinde çalışılmıştır:

* Olasılık ve istatistik temelli modelleme yapmak
* Parametre tahmini ve optimizasyon mantığını uygulamak
* Lineer cebir kavramlarını makine öğrenmesi bağlamında değerlendirmek
* Deney sonuçlarını grafikler ve raporlarla yorumlamak
* Jupyter Notebook üzerinden okunabilir ve tekrar üretilebilir çalışmalar hazırlamak

## Not

Bu repo, öğrenme ve uygulama amacıyla hazırlanmış kişisel makine öğrenmesi çalışmalarından oluşmaktadır. Odak noktası; üretim seviyesinde yazılım geliştirmeden çok, makine öğrenmesinin matematiksel altyapısını anlamak, uygulama pratiği kazanmak ve düzenli teknik dokümantasyon oluşturmaktır.
