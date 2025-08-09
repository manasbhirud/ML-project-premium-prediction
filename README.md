# 🏥 Health Insurance Premium Prediction

This project is part of an **ML** and aims to predict the **health insurance premium** a customer will pay based on personal and health-related factors.  
It demonstrates a complete **machine learning workflow** — from data preprocessing to model training, evaluation, and deployment-ready code.

---

## 📌 Project Overview
Insurance companies determine premiums based on several factors such as age, BMI, smoking habits, and region.  
Our task is to **train a predictive model** using historical data to estimate the premium amount for new customers.

---

## 🎯 Objectives
- Understand the health insurance dataset.
- Perform **data cleaning** and **feature engineering**.
- Train multiple machine learning models.
- Select the best-performing model for predictions.
- Evaluate model performance using appropriate metrics.
- Deploy the model via a simple API or web app.

---

## 📂 Dataset
- **Features:**
  - `age` — Age of the individual
  - `gender` — Gender (`male`, `female`)
  - `bmi_category` — Body Mass Index
  - `Maritial_status` — Marraige status (`yes`, `no`)
  - `smoking_status` — Smoking habit (`yes`, `no`)
  - `region` — Residential region (`northeast`, `northwest`, `southeast`, `southwest`)
  - `insurance_plan` — Insurance Plan (`Gold`, `Silver`, `Bronze`)
  - `number_of_dependants` — Number Of Dependants (`1`, `2`, `3`, `4`, `5`)
  - `Employment_Status` — Employment_Status (`Salaried`, `Self_employed`, `Freelancer`)
  - `income_level` — Income_Lakhs (`<10L`, `10L - 25L'`, `> 40L`,`25L - 40L`)
  - `annual_premium_amount` — Insurance premium (Target variable)
---

## 🛠 Tech Stack
- **Language:** Python 3.12.5
- **Libraries:**
  - Data Analysis: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`
  - Model Deployment (Optional): `flask` / `streamlit`
- **Environment:** Jupyter Notebook / VS Code

---

## 🚀 Steps Implemented
1. **Data Exploration & Visualization**
   - Checked missing values, data types.
   - Visualized feature distributions & correlations.
2. **Data Preprocessing**
   - Encoded categorical variables.
   - Standardized numerical features (where required).
   - Split data into training & testing sets.
3. **Model Training**
   - Linear Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
4. **Model Evaluation**
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - R² Score
5. **Hyperparameter Tuning**
   - Used GridSearchCV for model optimization.
6. **Final Model Selection**
   - Chose the model with the best test performance.
7. **(Optional) Deployment**
   - Exported model using `joblib` for API/web integration.

---

## 📊 Results
| Model                   | MAE      | RMSE  | R² Score |
|-------------------------|----------|-------|----------|
| Linear Regression       | 2183844.3|1477.7 | 0.95     |
| XG Boost Regressor      | 96814.2  |311.14 | 0.99     |


## 📦 Installation & Usage

### 1️⃣ Clone the repository
```
git clone https://github.com/manasbhirud/ML-project-premium-prediction.git
cd ML-project-premium-prediction

### 2️⃣ Install dependencies
pip install -r requirements.txt

### 3️⃣ Run the notebook
jupyter notebook health_insurance_prediction.ipynb

### 4️⃣ Run the web app
streamlit run app.py
```

## 📌 Future Improvements
```
Add deep learning models for experimentation.

Improve feature engineering (interaction terms, polynomial features).

Deploy model on cloud (Heroku, Render, etc.).

Integrate with a frontend UI for user-friendly predictions.
```
