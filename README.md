
# Breast Cancer Classification (Sklearn - Wisconsin Diagnostic)

Bu proje, **Iris** örneğiyle aynı mantığı izleyen ama **yeni bir problem** kullanan bir sınıflandırma çalışmasıdır.
- Amaç: Meme kanseri tanı verisini (benign vs malignant) sınıflandırmak
- Veri: `sklearn.datasets.load_breast_cancer` → `data/breast_cancer.csv` olarak kaydedildi
- Araçlar: Python, scikit-learn, matplotlib
- Çıktılar:
  - EDA figürleri → `reports/figures/`
  - En iyi model → `models/best_model.joblib`
  - Metrikler → `reports/metrics.json`
  - Tahmin betiği → `src/predict.py` (tek satırda CLI çalıştırma desteği)

## Hızlı Başlangıç
```bash
# 1) Gerekli paketleri kurun (aşağıdaki requirements.txt)
pip install -r requirements.txt

# 2) EDA üret
python src/eda.py

# 3) Eğit ve en iyi modeli seç
python src/train.py

# 4) Örnek tahmin
python src/predict.py --input "mean radius=14.0,mean texture=20.0,mean perimeter=90.0,mean area=600.0,mean smoothness=0.1"
```

## Yapı
```
project_breast_cancer/
├── data/
│   └── breast_cancer.csv
├── models/
│   └── best_model.joblib
├── notebooks/
├── reports/
│   ├── figures/
│   └── metrics.json
├── src/
│   ├── eda.py
│   ├── train.py
│   └── predict.py
└── requirements.txt
```

## Notlar
- Grafiklerde **yalnızca matplotlib** kullanıldı, **tek figür** kuralı ve **renk belirtmeme** koşulları sağlandı.
- Model seçimi: Stratified KFold ile validasyon F1 skoruna göre.
- Iris örneğindeki gibi birden fazla model denenir (LogisticRegression, RandomForest, SVC, MLPClassifier).

