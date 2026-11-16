
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------

st.set_page_config(
    page_title="DiaPredict | Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

# ------------------------------------------
# CUSTOM DESIGN
# ------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #F4F7FB;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

.hero {
    background: linear-gradient(120deg, #102C46, #155C69);
    padding: 38px;
    border-radius: 22px;
    color: white;
    margin-bottom: 28px;
}

.hero h1 {
    color: white;
    font-size: 36px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #D6E9ED;
    font-size: 16px;
}

.eyebrow {
    color: #77D5CB;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.section-title {
    color: #17354D;
    font-size: 23px;
    font-weight: 800;
    margin-top: 15px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #718096;
    font-size: 14px;
    margin-bottom: 20px;
}

.info-card {
    background: white;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #E4EAF1;
    margin-bottom: 16px;
}

.info-card h3 {
    color: #17354D;
    font-size: 17px;
    margin-bottom: 6px;
}

.info-card p {
    color: #718096;
    font-size: 13px;
}

.result-card {
    background: white;
    padding: 28px;
    border-radius: 18px;
    border: 1px solid #E4EAF1;
    margin-top: 20px;
}

.result-positive {
    border-left: 6px solid #E6A23C;
}

.result-negative {
    border-left: 6px solid #18A999;
}

.result-heading {
    color: #17354D;
    font-size: 24px;
    font-weight: 800;
}

.small-label {
    color: #718096;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #E4EAF1;
    text-align: center;
}

.metric-number {
    color: #147D83;
    font-size: 25px;
    font-weight: 800;
}

.footer {
    text-align: center;
    color: #8492A6;
    font-size: 12px;
    padding: 25px;
}

div.stButton > button {
    background: #147D83;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 13px 20px;
    font-weight: 700;
    width: 100%;
}

div.stButton > button:hover {
    background: #105F65;
    color: white;
    border: none;
}

[data-testid="stForm"] {
    background: white;
    border: 1px solid #E4EAF1;
    border-radius: 16px;
    padding: 22px;
}

[data-testid="stSidebar"] {
    background: #102C46;
}

[data-testid="stSidebar"] * {
    color: white;
}

hr {
    border-color: #E4EAF1;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------
# LOAD MODEL
# ------------------------------------------

MODEL_PATH = Path(__file__).parent / "diabetes_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as error:
    st.error(f"Model loading error: {error}")
    st.stop()


# ------------------------------------------
# SIDEBAR
# ------------------------------------------

with st.sidebar:
    st.markdown("## 🩺 DiaPredict")
    st.markdown("---")

    st.markdown("### About this project")
    st.write(
        "An AI-powered educational prototype "
        "exploring machine learning applications "
        "in diabetes prediction."
    )

    st.markdown("---")
    st.markdown("### Project information")
    st.write("**Model:** Trained ML Pipeline")
    st.write("**Application:** Streamlit")
    st.write("**Domain:** Healthcare AI")

    st.markdown("---")
    st.caption("Student AI Engineering Project")


# ------------------------------------------
# HERO SECTION
# ------------------------------------------

st.markdown("""
<div class="hero">
    <div class="eyebrow">ARTIFICIAL INTELLIGENCE · HEALTHCARE</div>
    <h1>Diabetes Prediction System</h1>
    <p>
        Exploring the power of machine learning to support
        data-driven healthcare insights.
    </p>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------
# INTRODUCTION
# ------------------------------------------

st.markdown(
    '<div class="section-title">Patient Assessment</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Enter the required health measurements to generate a model prediction.'
    '</div>',
    unsafe_allow_html=True
)


# ------------------------------------------
# INPUT FORM
# ------------------------------------------

with st.form("patient_form"):

    st.markdown("### 👤 Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=0
        )

        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=30
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=100.0,
            value=25.0
        )

    with col2:
        glucose = st.number_input(
            "Glucose (mg/dL)",
            min_value=0.0,
            max_value=300.0,
            value=120.0
        )

        blood_pressure = st.number_input(
            "Blood Pressure (mm Hg)",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )

        skin_thickness = st.number_input(
            "Skin Thickness (mm)",
            min_value=0.0,
            max_value=100.0,
            value=20.0
        )

    with col3:
        insulin = st.number_input(
            "Insulin (mu U/ml)",
            min_value=0.0,
            max_value=1000.0,
            value=80.0
        )

        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5
        )

        st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "Generate Prediction →",
        use_container_width=True
    )


# ------------------------------------------
# PREDICTION
# ------------------------------------------

if submitted:

    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }])

    try:
        prediction = model.predict(input_data)[0]

        st.markdown("---")
        st.markdown(
            '<div class="section-title">Assessment Results</div>',
            unsafe_allow_html=True
        )

        if prediction == 1:
            st.markdown("""
            <div class="result-card result-positive">
                <div class="small-label">MODEL OUTPUT</div>
                <div class="result-heading">Higher likelihood indicated</div>
                <p>
                    The model classified this input as belonging
                    to the diabetes-positive class.
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="result-card result-negative">
                <div class="small-label">MODEL OUTPUT</div>
                <div class="result-heading">Lower likelihood indicated</div>
                <p>
                    The model classified this input as belonging
                    to the diabetes-negative class.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Model probability, if available
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]
            probability = probabilities[1] * 100

            st.write("")
            st.markdown("### Model Estimate")

            st.progress(float(probability / 100))

            st.markdown(
                f"**Diabetes-positive class score: {probability:.1f}%**"
            )

            st.caption(
                "This is the model's estimated class probability, "
                "not a clinically validated measure of individual risk."
            )

        st.info(
            "This output is an educational machine learning result. "
            "It cannot confirm or rule out diabetes."
        )

    except Exception as error:
        st.error(f"Prediction error: {error}")


# ------------------------------------------
# PROJECT OVERVIEW
# ------------------------------------------

st.markdown("---")

st.markdown(
    '<div class="section-title">About the Technology</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'A closer look at the system behind the prediction.'
    '</div>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="info-card">
        <h3>🧠 Machine Learning</h3>
        <p>A trained classification pipeline processes
        patient measurements to generate predictions.</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
        <h3>📊 Health Data</h3>
        <p>The project uses eight selected numerical
        features from a diabetes dataset.</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="info-card">
        <h3>💻 Interactive AI</h3>
        <p>A Streamlit interface connects user inputs
        with the saved machine learning model.</p>
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------
# DISCLAIMER AND FOOTER
# ------------------------------------------

st.markdown("---")

st.warning(
    "**Medical Disclaimer:** This application is an educational "
    "prototype. It is not intended for clinical diagnosis, treatment, "
    "or medical decision-making. Consult a qualified healthcare "
    "professional for proper assessment."
)

st.markdown("""
<div class="footer">
    DIAPREDICT · AI ENGINEERING 2026
    <br>
    Built with Python, Scikit-learn and Streamlit
</div>
""", unsafe_allow_html=True)
