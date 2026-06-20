# Student Mental Health & Academic Performance Prediction System

> **Final Year University ML Project** — Production-quality AI system built with Python, Scikit-Learn & Flask

---

## 🎯 Project Overview

This system predicts four key student outcomes using **Linear Regression** and **KNN** algorithms:

| Target | Scale | Best R² |
|--------|-------|---------|
| CGPA | 0.0 – 4.0 | **1.000** |
| Stress Level | 0 – 10 | **0.996** |
| Depression Score | 0 – 10 | **0.885** |
| Anxiety Score | 0 – 10 | **0.777** |

---

## 📁 Project Structure

```
project/
├── app.py                    # Flask web application
├── train.py                  # ML training pipeline
├── requirements.txt
├── README.md
│
├── preprocessing/
│   └── preprocess.py         # Data cleaning & feature engineering
│
├── utils/
│   └── eda.py                # EDA visualizations
│
├── dataset/
│   ├── students_mental_health_survey.csv   # Raw data
│   └── processed_student_data.csv          # Engineered features
│
├── trained_models/           # Saved .pkl models
│
├── static/
│   ├── css/style.css
│   ├── js/predict.js
│   └── graphs/               # Auto-generated EDA & performance plots
│
└── templates/
    ├── index.html
    ├── predict.html
    ├── result.html
    └── dashboard.html
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the models (already done — models are in trained_models/)
```bash
python train.py
```

### 3. Run the Flask app
```bash
python app.py
```

### 4. Open in browser
```
http://127.0.0.1:5000
```

---

## 🔬 ML Pipeline

### Phase 1 — Data Cleaning
- Median imputation for CGPA (12 missing)
- "Never" fill for Substance_Use (15 missing)
- Mode fill for remaining categoricals

### Phase 2 — Encoding
- **Ordinal**: Sleep Quality, Physical Activity, Diet Quality, Social Support, Extracurricular, Counseling, Substance Use
- **Binary**: Gender, Family History, Chronic Illness
- **One-Hot** (drop_first): Course, Relationship Status, Residence Type

### Phase 3 — Outlier Handling
- IQR-based clipping at 1st & 99th percentile for CGPA, Stress, Depression, Anxiety

### Phase 4 — Feature Engineering (50 features total)
- Wellness Score, Lifestyle Index, Wellbeing vs Burden
- Mental Burden, Support Deficit, Academic Pressure
- 7 interaction terms (Sleep×Activity, Stress×Financial, etc.)
- Risk Score, Family History Risk, Substance/Counseling ratio
- Total Mental Load, Avg Mental Score
- Polynomial features (sq) for Sleep, Activity, Diet, Social Support
- CGPA deviation & percentile rank

### Phase 5 — Model Training
Each target trains **5 models**:
1. Linear Regression
2. Ridge Regression (α=1.0)
3. Lasso Regression (α=0.01)
4. ElasticNet (α=0.01, l1_ratio=0.5)
5. **KNN Regressor** (k auto-tuned 3–20 via 5-fold CV, distance-weighted)

Pipeline: `StandardScaler → Model`

---

## 📊 Model Performance

```
Target             Model                MAE      RMSE     R²
─────────────────────────────────────────────────────────────
CGPA               Linear Regression   0.0000   0.0000   1.0000
Stress_Level       Linear Regression   0.0823   0.1086   0.9956
Depression_Score   Linear Regression   0.4169   0.5520   0.8849
Anxiety_Score      Linear Regression   0.5978   0.7795   0.7765
```

---

## 🌐 Web Application Pages

| Route | Description |
|-------|-------------|
| `/` | Homepage with project overview |
| `/predict` | Dynamic prediction form |
| `/dashboard` | Analytics & EDA dashboard |
| `/api/metrics` | JSON API for all model metrics |

---

## 🛠 Tech Stack

- **ML**: Scikit-Learn, NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Web**: Flask, Bootstrap 5, Font Awesome 6
- **Persistence**: Joblib
- **Language**: Python 3.10+
"# finak-year" 
