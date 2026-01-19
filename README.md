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

**Clone the repository:**
```bash
git clone https://github.com/your-username/medical-insurance-prediction.git
cd medical-insurance-prediction

```

**Install dependencies:**
```
pip install -r requirements.txt

```

# 🚀 Usage

## 1. Train the Model
Run the training script to preprocess data, train the model, and save the pipeline:

```
python insurance_train.py

```
This will generate a file:

```
insurance_model.pkl

```

## 2. Launch Gradio App
Run the app to start the web interface:

```
python app.py

```

Localhost link will appear (e.g., http://127.0.0.1:7860).
If share=True is enabled, a temporary public link will be generated.


# 📊 Example Input

```
Age	Sex	BMI	Children	Smoker	Region
30	male	35.3	0	yes	southwest

```

**Example Output**

```
Predicted Insurance Cost: ~36,000
```

# 🌐 Deployment on Hugging Face Spaces

- Push your repository to GitHub.

- Connect Hugging Face account with GitHub.

- Create a new Space → Select Gradio as SDK.

- Add requirements.txt with dependencies:

```
pandas
numpy
scikit-learn
gradio

```

- Deploy and get a public URL.

👉 Live Demo: Medical Insurance Cost Predictor on Hugging Face Spaces (huggingface.co in Bing)  
(Replace your-username with your Hugging Face username after deployment.)


# 📈 Evaluation Metrics

- R² Score: Measures goodness of fit.

- MSE: Mean Squared Error.

- RMSE: Root Mean Squared Error.


# 🙌 Acknowledgements
- Dataset: Kaggle - Medical Insurance Dataset

- Libraries: Scikit-learn, Pandas, NumPy, Gradio


# 👨‍💻 Author
Developed by **Rubina Begum**
Feel free to contribute or suggest improvements!