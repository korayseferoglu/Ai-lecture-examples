# How to Run (Quick Start)

## ▶️ Run on Google Colab
1. Open a new Colab notebook.
2. Clone the repo and enter the folder:
   !git clone https://github.com/korayseferoglu/Ai-lecture-examples.git
   %cd Ai-lecture-examples
3. Install dependencies:
   !pip install -q -r requirements.txt
4. Generate EDA figures (saved to reports/figures/):
   !python src/eda.py
5. Train and save best model (saved to models/best_model.joblib, metrics to reports/metrics.json):
   !python src/train.py
6. Make a sample prediction (0=malignant, 1=benign):
   !python src/predict.py --input "mean radius=14.0,mean texture=20.0,mean perimeter=90.0,mean area=600.0,mean smoothness=0.1"
7. (Optional) View outputs:
   !ls -lh models
!cat reports/metrics.json
  


