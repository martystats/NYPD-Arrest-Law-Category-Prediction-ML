# 🚔 NYPD Arrest Law Category Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-blue?logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Application-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

# 📖 Project Overview

This project presents a **leakage-controlled machine learning application** that predicts the likely **NYPD arrest law category** using historical arrest records published by the New York City Police Department (NYPD).

The project was rebuilt after identifying and removing target leakage from an earlier version. The final model follows good machine learning practices by using only the features available at prediction time.

The application demonstrates an end-to-end machine learning workflow including:

- Data cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Target leakage detection and removal
- Model comparison
- Hyperparameter tuning
- Model evaluation
- Streamlit deployment
- GitHub project documentation

---

# 🎯 Project Objective

The objective of this project is to build a machine learning classifier capable of predicting the likely NYPD arrest law category based on information available at the time of an arrest.

This project is intended for educational and portfolio purposes to demonstrate practical machine learning development and deployment. It should not be used for legal, policing, or operational decision-making.

---

# 🧠 Machine Learning Pipeline

The project follows a complete end-to-end supervised machine learning workflow.

1. Raw NYPD Arrest Dataset
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Target Leakage Detection
6. Leakage Removal
7. Train / Validation / Test Split
8. Pipeline Construction
9. Model Training
10. Hyperparameter Tuning
11. Model Evaluation
12. Streamlit Deployment
13. GitHub Documentation

---

# 📊 Dataset Information

Dataset Source:
New York City Open Data (NYPD Arrest Data)

Target Variable:

- LAW_CAT_CD

Target Classes

- Felony (F)
- Misdemeanor (M)
- Violation (V)
- Infraction (I)

Original Dataset Size

- 584,852 arrest records

Final Model Features (Leakage-Controlled)

- Arrest Year
- Arrest Month
- Arrest Day
- Arrest Weekday
- Arrest Precinct
- Jurisdiction Code
- Arrest Borough
- Age Group
- Sex
- Race

---

# 🤖 Final Model

## Selected Algorithm

Gradient Boosting Classifier

### Why Gradient Boosting?

Several classification algorithms were evaluated during model development, including:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

After evaluating multiple classification algorithms, Gradient Boosting Classifier was selected as the final model because it provided the best balance between predictive performance, generalization, and robustness on the leakage-controlled dataset.
---

# 🔒 Target Leakage Removal

One of the major objectives of this project was to eliminate target leakage.

The original dataset contained variables that directly revealed or strongly implied the target class, including offence descriptions and offence codes.

To produce a realistic machine learning model, these variables were removed before training.

The final model uses only information that would realistically be available at prediction time.

This results in a more reliable evaluation and a deployment-ready machine learning workflow.

---

# 📈 Model Performance

The final model was evaluated using a separate validation dataset after removing all target leakage variables.

Evaluation focused on measuring the model's ability to generalize to unseen arrest records rather than memorizing target-related information.

Performance metrics considered during model selection included:

- Accuracy
- Macro F1 Score
- Balanced Accuracy
- Classification Report
- Confusion Matrix

The final Gradient Boosting Classifier demonstrated the best balance between predictive performance, generalization, and robustness on the leakage-controlled validation dataset.

---

# 💻 Streamlit Application

The project includes an interactive Streamlit web application that allows users to generate predictions from leakage-controlled arrest information.

### Application Features

- Modern responsive interface
- Automatic date feature generation
- Leakage-controlled prediction pipeline
- Prediction confidence score
- Gradient Boosting classifier
- Clean sidebar with model information
- Real-time prediction

The deployed application only accepts variables available before an arrest law category is known, ensuring realistic deployment behaviour.

---

# 📷 Application Preview

The Streamlit application provides a simple interface for predicting the likely NYPD arrest law category from information that would realistically be available before the final law category is known.

The interface includes:

- Two-column responsive layout
- Leakage-controlled prediction inputs
- Automatic date feature generation
- Prediction confidence score
- Model information sidebar
- Real-time classification results

*(Application screenshots can be added here.)*

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/martystats/NYPD-Arrest-Law-Category-Prediction-ML.git
```

Move into the project directory

```bash
cd NYPD-Arrest-Law-Category-Prediction-ML
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

# 🚀 Future Improvements

Possible future enhancements include:

- Deploy the application online using Streamlit Community Cloud
- Add geographical visualization of arrest locations
- Introduce explainable AI (SHAP) for prediction interpretation
- Compare additional ensemble learning algorithms
- Expand prediction using temporal crime trend analysis

---

# 📄 License

This project is released under the MIT License.

---

# 👨‍💻 Author

**Martin Ude**

Data Analyst | Machine Learning Developer

GitHub:
https://github.com/martystats

---

## ⭐ If you found this project useful, consider starring the repository.