# 🏥 Medical Insurance Cost Prediction System

A Machine Learning project that predicts medical insurance costs based on user details such as age, sex, BMI, children, smoking status, and region.  
This project integrates **data preprocessing, feature engineering, model training, hyperparameter tuning, and deployment** using **Gradio** and can be hosted on **Hugging Face Spaces**.

---

## 📌 Features
- End-to-end ML pipeline with preprocessing (scaling + encoding).
- Feature engineering: BMI category creation.
- Model training using **Random Forest Regressor**.
- Hyperparameter tuning with **GridSearchCV**.
- Evaluation metrics: R², MSE, RMSE.
- Interactive web interface built with **Gradio**.
- Easy deployment to Hugging Face Spaces.

---

## 📂 Project Structure

```

├── insurance.csv                # Dataset
├── insurance_train.py           # Training script (preprocessing, training, evaluation, save model)
├── app.py                       # Gradio app for prediction
├── requirements.txt             # Dependencies
├── insurance_model.pkl          # Saved trained ML pipeline
└── README.md                    # Project documentation

```


---

## ⚙️ Installation

Clone the repository:
```bash
git clone https://github.com/your-username/medical-insurance-prediction.git
cd medical-insurance-prediction



