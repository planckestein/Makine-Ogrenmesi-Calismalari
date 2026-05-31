# MLE ile Akıllı Şehir Planlaması

Bu projede bir ana caddeden 1 dakikada geçen araç sayıları Poisson dağılımı ile modellenmiştir. Amaç, trafik yoğunluğunu temsil eden λ parametresini Maximum Likelihood Estimation (MLE) yöntemi ile tahmin etmektir.

## Problem Tanımı
Poisson varsayımı altında trafik yoğunluğu için en uygun λ parametresi hem analitik olarak hem de sayısal optimizasyon ile hesaplanmıştır. Ardından model-histogram uyumu ve outlier etkisi incelenmiştir.

## Veri
Kullanılan veri: `[12, 15, 10, 8, 14, 11, 13, 16, 9, 12, 11, 14, 10, 15]`

## Yöntem
1. Likelihood ve log-likelihood fonksiyonları türetildi.
2. Türev sıfıra eşitlenerek `λ_MLE = örnek ortalaması` sonucu gösterildi.
3. `scipy.optimize.minimize` ile negatif log-olabilirlik minimize edildi.
4. Histogram ve Poisson PMF karşılaştırıldı.
5. Veri setine `200` değerli outlier eklenerek parametre kayması analiz edildi.

## Sonuçlar
- Analitik λ tahmini: `12.142857`
- Sayısal λ tahmini: `12.142851`
- Outlier sonrası λ tahmini: `24.666654`

## Yorum / Tartışma
Orijinal veri için model makul düzeyde uyum göstermektedir. Ancak `200` araçlık outlier λ değerini ciddi biçimde yukarı itmektedir. Bu da MLE'nin, özellikle Poisson modelinde ortalamaya dayandığı için, uç gözlemlere hassas olduğunu gösterir.
