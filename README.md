# ⚙️ AeroPulse AI — Enterprise Industrial Predictive Maintenance Engine

An end-to-end machine learning web application built with **Python**, **Scikit-Learn**, and **Streamlit** to forecast factory machinery breakdowns using real-world sensor telemetry and ensemble models.

---

## 🚀 Key Features
- **Production-Grade Pipelines:** Uses Scikit-Learn `Pipeline` and `StandardScaler` to prevent data leakage during feature preprocessing.
- **Ensemble vs. Baseline Modeling:** Compares a non-linear **Random Forest Classifier** against a linear **Logistic Regression** baseline.
- **Handling Class Imbalance:** Employs balanced class weighting (`class_weight='balanced'`) to ensure high recall on rare machinery failure events.
- **Enterprise SaaS UI:** Built with custom dark-theme styling, top KPI metric tracking, and high-end professional alert components.

---

## 📂 Project Structure
```text
aeropulse-predictive-maintenance/
│
├── ai4i2020.csv      # Real UCI AI4I Predictive Maintenance Dataset
├── model.py          # ML pipelines, data preprocessing, and training logic
├── app.py            # Streamlit enterprise dashboard UI layout
└── README.md         # Project documentation
