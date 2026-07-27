import streamlit as st
import pandas as pd
import joblib
from datetime import date


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="NYPD Arrest Law Category Prediction",
    page_icon="🚓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# LOAD LEAKAGE-CONTROLLED DEPLOYMENT FILES
# ---------------------------------------------------------
@st.cache_resource
def load_model_files():
    model = joblib.load("best_clean_arrest_model.pkl")
    category_mappings = joblib.load("category_mappings.pkl")
    target_classes = joblib.load("target_classes.pkl")
    return model, category_mappings, target_classes


@st.cache_data
def load_dataset():
    return pd.read_csv("nypd_arrest_data_cleaned.csv", low_memory=False)


try:
    model, category_mappings, target_classes = load_model_files()
    df = load_dataset()

except Exception as error:
    st.error(f"Unable to load the deployment files: {error}")
    st.stop()


# ---------------------------------------------------------
# HELPER FUNCTION
# ---------------------------------------------------------
def clean_options(values):
    cleaned = []

    for value in values:
        if pd.notna(value):
            text = str(value).strip()

            if text.lower() not in ["", "nan", "none", "(null)"]:
                cleaned.append(value)

    return sorted(set(cleaned), key=lambda item: str(item))


# ---------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------
classifier = model.named_steps.get("classifier")

classifier_name = (
    type(classifier).__name__
    if classifier is not None
    else type(model).__name__
)

display_algorithm = (
    "Gradient Boosting Classifier"
    if classifier_name == "GradientBoostingClassifier"
    else classifier_name
)

expected_features = [
    "ARREST_YEAR",
    "ARREST_MONTH",
    "ARREST_DAY",
    "ARREST_WEEKDAY",
    "ARREST_PRECINCT",
    "JURISDICTION_CODE",
    "ARREST_BORO",
    "AGE_GROUP",
    "PERP_SEX",
    "PERP_RACE"
]


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.header("📊 Model Information")
    st.divider()

    st.subheader("Algorithm")
    st.info(display_algorithm)

    st.subheader("Target")
    st.info("NYPD Arrest Law Category")

    st.subheader("Model Inputs")
    st.info(f"{len(expected_features)} leakage-controlled attributes")

    st.subheader("Status")
    st.warning("Model Validation in Progress")

    st.divider()

    st.caption(
        "Offence descriptions, law codes, KY codes and PD codes were "
        "removed to reduce target leakage."
    )


# ---------------------------------------------------------
# PAGE TITLE AND DESCRIPTION
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .subtitle {
        font-size: 17px;
        margin-bottom: 28px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">🚓 NYPD Arrest Law Category Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Predict the likely arrest law category using a leakage-controlled
    Gradient Boosting machine-learning pipeline.
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    "This educational model intentionally excludes offence descriptions "
    "and offence/law codes because those fields may directly reveal the "
    "target category."
)

st.header("Enter Arrest Information")


# ---------------------------------------------------------
# PREPARE DROPDOWN OPTIONS
# ---------------------------------------------------------
age_options = clean_options(
    category_mappings.get(
        "AGE_GROUP",
        df["AGE_GROUP"].dropna().unique()
    )
)

sex_options = clean_options(
    category_mappings.get(
        "PERP_SEX",
        df["PERP_SEX"].dropna().unique()
    )
)

race_options = clean_options(
    category_mappings.get(
        "PERP_RACE",
        df["PERP_RACE"].dropna().unique()
    )
)

borough_codes = clean_options(
    category_mappings.get(
        "ARREST_BORO",
        df["ARREST_BORO"].dropna().unique()
    )
)

borough_names = {
    "B": "Bronx",
    "K": "Brooklyn",
    "M": "Manhattan",
    "Q": "Queens",
    "S": "Staten Island"
}

available_boroughs = {
    borough_names[code]: code
    for code in borough_codes
    if code in borough_names
}

precinct_options = sorted(
    pd.to_numeric(
        df["ARREST_PRECINCT"],
        errors="coerce"
    )
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)

jurisdiction_options = sorted(
    pd.to_numeric(
        df["JURISDICTION_CODE"],
        errors="coerce"
    )
    .dropna()
    .unique()
    .tolist()
)


# ---------------------------------------------------------
# USER INPUTS
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    age_group = st.selectbox(
        "Age Group",
        age_options
    )

    sex = st.selectbox(
        "Sex",
        sex_options
    )

    borough_full = st.selectbox(
        "Arrest Borough",
        list(available_boroughs.keys())
    )

    borough = available_boroughs[borough_full]

    race = st.selectbox(
        "Race",
        race_options
    )


with col2:
    arrest_precinct = st.selectbox(
        "Arrest Precinct",
        precinct_options,
        help=(
            "The NYPD precinct number where the arrest was recorded. "
            "Precincts are identified numerically."
        )
    )

    jurisdiction_code = st.selectbox(
        "Jurisdiction Code",
        jurisdiction_options,
        help=(
            "A numerical code describing the jurisdiction responsible "
            "for the arrest record."
        )
    )

    st.markdown("#### Automatically generated date attributes")

    current_date = date.today()

    st.write(f"**Arrest year:** {current_date.year}")
    st.write(f"**Arrest month:** {current_date.month}")
    st.write(f"**Arrest day:** {current_date.day}")
    st.write(f"**Weekday code:** {current_date.weekday()}")


# ---------------------------------------------------------
# CREATE RAW PIPELINE INPUT
# ---------------------------------------------------------
input_final = pd.DataFrame({
    "ARREST_YEAR": [int(current_date.year)],
    "ARREST_MONTH": [int(current_date.month)],
    "ARREST_DAY": [int(current_date.day)],
    "ARREST_WEEKDAY": [int(current_date.weekday())],
    "ARREST_PRECINCT": [int(arrest_precinct)],
    "JURISDICTION_CODE": [float(jurisdiction_code)],
    "ARREST_BORO": [str(borough)],
    "AGE_GROUP": [str(age_group)],
    "PERP_SEX": [str(sex)],
    "PERP_RACE": [str(race)]
})


# ---------------------------------------------------------
# LAW CATEGORY MEANINGS
# ---------------------------------------------------------
law_category_meanings = {
    "F": "Felony",
    "M": "Misdemeanor",
    "V": "Violation",
    "I": "Infraction",
    "9": "Unknown / Other"
}


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
st.divider()

if st.button(
    "Predict Law Category",
    type="primary",
    use_container_width=False
):
    try:
        prediction = model.predict(input_final)
        predicted_code = str(prediction[0])

        prediction_meaning = law_category_meanings.get(
            predicted_code,
            "Unknown Category"
        )

        probability = model.predict_proba(input_final)[0]
        confidence = float(probability.max() * 100)

        st.divider()
        st.subheader("📊 Prediction Result")

        st.success(
            f"Predicted Law Category: "
            f"{prediction_meaning} ({predicted_code})"
        )

        st.info(
            f"Prediction Confidence: {confidence:.2f}%"
        )

        st.caption(
            "The confidence value represents the model's estimated "
            "probability for its selected class. It should not be "
            "interpreted as legal advice or certainty."
        )

    except Exception as error:
        st.error(f"Prediction could not be generated: {error}")


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()

st.caption(
    "Leakage-Controlled Rebuild | Developed by Martin Ude | "
    "NYPD Arrest Law Category Prediction | "
    "Gradient Boosting Classification"
)