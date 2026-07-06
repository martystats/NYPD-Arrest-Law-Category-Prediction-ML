import streamlit as st
import pandas as pd
import joblib

# Load saved files
model = joblib.load("best_clean_arrest_model.pkl")
feature_columns = joblib.load("clean_feature_columns.pkl")
category_mappings = joblib.load("category_mappings.pkl")
label_encoder = joblib.load("label_encoder.pkl")
# Load the cleaned dataset
df = pd.read_csv("nypd_arrest_data_cleaned.csv")

st.set_page_config(
    page_title="NYPD Arrest Law Category Prediction",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🚔 NYPD Arrest Law Category Prediction</div>',
    unsafe_allow_html=True
)

st.write(
    "Predict the likely arrest law category using a deployment-ready Decision Tree classification model."
)

st.sidebar.subheader("📊 Model Information")
st.sidebar.markdown("---")

st.sidebar.markdown("**Algorithm**")
st.sidebar.info("Decision Tree Classifier")

st.sidebar.markdown("**Target**")
st.sidebar.info("NYPD Law Category")

st.sidebar.markdown("**Features**")
st.sidebar.info("8 Arrest Attributes")

st.sidebar.markdown("**Model Accuracy**")
st.sidebar.info("94.82%")

st.sidebar.markdown("**Status**")
st.sidebar.success("Deployment Ready ✅")

st.header("Enter Arrest Information")
col1, col2 = st.columns(2)

age_options = [x for x in category_mappings["AGE_GROUP"]
               if str(x).lower() != "nan" and str(x) != "(null)"]

sex_options = [x for x in category_mappings["PERP_SEX"]
               if str(x).lower() != "nan" and str(x) != "(null)"]

boro_map = {
    "B": "Bronx",
    "K": "Brooklyn",
    "M": "Manhattan",
    "Q": "Queens",
    "S": "Staten Island"
}

with col1:
    age_group = st.selectbox("Age Group", age_options)

    sex = st.selectbox("Sex", sex_options)

    borough_full = st.selectbox(
        "Arrest Borough",
        list(boro_map.values())
    )

    borough = {v: k for k, v in boro_map.items()}[borough_full]

    race = st.selectbox(
        "Race",
        category_mappings["PERP_RACE"]
    )

with col2:
    ofns_desc = st.selectbox(
    "Offense Description",
    sorted(df["OFNS_DESC"].dropna().unique())
    )

    filtered_pd = df[df["OFNS_DESC"] == ofns_desc]

    pd_desc = st.selectbox(
        "PD Description",
        sorted(filtered_pd["PD_DESC"].dropna().unique())
    )

    filtered_codes = filtered_pd[filtered_pd["PD_DESC"] == pd_desc]

    ky_cd = st.selectbox(
        "Key (KY) Offense Code",
        sorted(filtered_codes["KY_CD"].dropna().unique())
    )

    pd_cd = st.selectbox(
        "Police Department (PD) Offense Code",
        sorted(filtered_codes["PD_CD"].dropna().unique())
    )

input_raw = pd.DataFrame({
    "ARREST_BORO": [borough],
    "AGE_GROUP": [age_group],
    "PERP_SEX": [sex],
    "PERP_RACE": [race],
    "OFNS_DESC": [ofns_desc],
    "PD_DESC": [pd_desc],
    "KY_CD": [ky_cd],
    "PD_CD": [pd_cd],
})

input_encoded = pd.get_dummies(input_raw)

input_final = pd.DataFrame(0, index=[0], columns=feature_columns)

for col in input_encoded.columns:
    if col in input_final.columns:
        input_final[col] = input_encoded[col]

law_category_meanings = {
    "M": "Misdemeanor",
    "F": "Felony",
    "V": "Violation",
    "I": "Infraction",
    "9": "Unknown / Other"
}

st.markdown("""
<style>
div.stButton > button {
    background-color: #28a745;
    color: white;
    border-radius: 8px;
    font-weight: bold;
    padding: 0.6em 1.2em;
}
</style>
""", unsafe_allow_html=True)
if st.button("Predict Law Category"):
    prediction = model.predict(input_final)

    probability = model.predict_proba(input_final)
    confidence = probability.max() * 100
    try:
        predicted_code = label_encoder.inverse_transform(prediction)[0]
    except:
        predicted_code = prediction[0]

    predicted_code = str(predicted_code)

    prediction_meaning = law_category_meanings.get(
        predicted_code,
        "Unknown Category"
    )

    st.divider()

    st.subheader("📊 Prediction Result")

    st.success(
        f"Predicted Law Category: {prediction_meaning} ({predicted_code})"
    )

    st.info(f"Prediction Confidence: {confidence:.2f}%")
    st.caption(
        "Prediction generated successfully using the final deployment Decision Tree model."
    )
    st.markdown("---")

    st.caption(
    "Version 1.0 | Developed by Martin Ude | NYPD Arrest Law Category Prediction | Decision Tree Classifier | Python • Streamlit • Scikit-learn"
)