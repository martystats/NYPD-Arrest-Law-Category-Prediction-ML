\# 🚔 NYPD Arrest Law Category Prediction ML



![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-blue?logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)



\# 📖 Project Overview



The \*\*NYPD Arrest Law Category Prediction ML\*\* project is an end-to-end machine learning application that predicts the likely \*\*NYPD Law Category\*\* based on arrest information entered by the user.



The project demonstrates the complete machine learning lifecycle, including:



\- Data Cleaning

\- Exploratory Data Analysis (EDA)

\- Feature Engineering

\- Machine Learning Model Development

\- Model Evaluation

\- Streamlit Web Application Deployment



The application provides an intuitive web interface where users can enter arrest details and instantly receive:



\- Predicted Law Category

\- Prediction Confidence

\- Model Information

\- Model Accuracy



\---



\# 🎯 Problem Statement



The New York Police Department records thousands of arrests each year.



Given important arrest information such as:



\- Age Group

\- Sex

\- Arrest Borough

\- Race

\- Offense Description

\- PD Description

\- Key Offense Code

\- Police Department Offense Code



can we accurately predict the \*\*Law Category\*\* associated with that arrest?



This project answers that question using supervised machine learning.



\---



\# 🎯 Project Objectives



The objectives of this project are to:



\- Build an end-to-end machine learning classification pipeline.

\- Clean and preprocess real-world NYPD arrest data.

\- Train a Decision Tree Classification model.

\- Evaluate model performance.

\- Deploy the model using Streamlit.

\- Provide an interactive interface for predictions.



\---



\# 📂 Dataset Information



\*\*Dataset:\*\* NYPD Arrest Data



The dataset contains historical arrest records collected by the New York Police Department.



\### Target Variable



\- Law Category



\### Input Features



\- Age Group

\- Sex

\- Arrest Borough

\- Race

\- Offense Description

\- Police Department Description

\- Key Offense Code (KY\_CD)

\- Police Department Offense Code (PD\_CD)



\---



\# 🧹 Data Cleaning \& Preprocessing



The following preprocessing steps were performed:



\- Removed duplicate records

\- Handled missing values

\- Removed invalid categories

\- Filtered null dropdown options

\- Encoded categorical variables

\- Prepared deployment feature columns

\- Saved reusable encoding objects



Deployment artifacts include:



\- best\_clean\_arrest\_model.pkl

\- clean\_feature\_columns.pkl

\- category\_mappings.pkl

\- label\_encoder.pkl



\---



\# 📊 Exploratory Data Analysis (EDA)



The dataset was explored using:



\- Summary statistics

\- Missing value analysis

\- Class distribution

\- Feature distributions

\- Correlation analysis

\- Data quality checks



The EDA provided insights into the characteristics of NYPD arrest records before model development.



\---



\# ⚙️ Feature Engineering



Feature engineering included:



\- Category encoding

\- Feature alignment for deployment

\- Consistent encoding between training and prediction

\- Deployment-ready feature vector creation



This ensures that user inputs are transformed into the exact format expected by the trained model.



\---



\# 🤖 Machine Learning Model



| Property | Value |

|----------|-------|

| Algorithm | Decision Tree Classifier |

| Problem Type | Multi-Class Classification |

| Accuracy | \*\*94.82%\*\* |

| Features | 8 Arrest Attributes |

| Deployment | Streamlit |



\---



\# 📈 Model Performance



\## Overall Accuracy



\*\*94.82%\*\*



The trained Decision Tree model achieved high classification accuracy while maintaining fast prediction speed suitable for deployment.



The application also displays:



\- Predicted Law Category

\- Prediction Confidence



for every prediction.



\---



\# 🖥️ Streamlit Application Features



The deployed application includes:



✅ Professional User Interface



✅ Two-column responsive layout



✅ Clean dropdown menus



✅ Borough names displayed instead of abbreviations



✅ Automatic feature encoding



✅ Prediction Confidence



✅ Model Accuracy



✅ Model Information Sidebar



✅ Deployment Ready Status



✅ Version Information



\---



\# 📁 Project Structure



```

NYPD\_Arrest\_Classification/



│

├── app.py

├── README.md

├── requirements.txt

├── best\_clean\_arrest\_model.pkl

├── clean\_feature\_columns.pkl

├── category\_mappings.pkl

├── label\_encoder.pkl

├── nypd\_arrest\_data\_cleaned.csv

└── Notebook.ipynb

```



\---



\# 🛠️ Technologies Used



\- Python

\- Pandas

\- NumPy

\- Scikit-Learn

\- Joblib

\- Streamlit

\- Jupyter Notebook



\---



\# 🚀 Installation



Clone the repository



```bash

git clone https://github.com/martystats/NYPD-Arrest-Law-Category-Prediction-ML.git

```



Move into the project



```bash

cd NYPD-Arrest-Law-Category-Prediction-ML

```



Install dependencies



```bash

pip install -r requirements.txt

```



Run the application



```bash

streamlit run app.py

```



\---



\# 📸 Application Screenshots



\## Home Page



\*(Insert Screenshot Here)\*



\---



\## Prediction Example



\*(Insert Screenshot Here)\*



\---



\## Model Information Sidebar



\*(Insert Screenshot Here)\*



\---



\# 🔮 Future Improvements



Potential enhancements include:



\- Random Forest comparison

\- XGBoost implementation

\- Hyperparameter tuning

\- SHAP explainability

\- Feature importance visualization

\- Online deployment

\- REST API integration



\---



\# 👨‍💻 Developer



\*\*Martin Ude\*\*



Machine Learning | Data Analytics | Python | Streamlit



GitHub:



https://github.com/martystats



\---



\# 📜 License



This project is licensed under the MIT License.



\---



\# ⭐ Acknowledgements



This project was developed for educational and portfolio purposes to demonstrate practical machine learning skills using publicly available NYPD arrest data.



If you found this project helpful, consider giving the repository a ⭐ on GitHub.

