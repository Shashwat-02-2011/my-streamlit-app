import pickle
import numpy as np
import streamlit as st
import plotly.express as px

# Page config
st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="assests/ChatGPT Image Apr 27, 2025, 10_18_25 AM", layout="centered")

# Load model and scaler
with open('model/heart_disease_svm_model.sav', 'rb') as model_file:
    svm_model = pickle.load(model_file)

with open('model/scaler.sav', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Custom Styling
st.markdown(
    """
    <style>
    body {
        background-color: #f8fbff;
    }
    .reportview-container {
        background: linear-gradient(135deg, #e0f7fa, #e1f5fe);
    }
    .sidebar .sidebar-content {
        background-color: #f1f3f6;
    }
    h1, h2, h3, h4 {
        color: #263238;
    }
    .stButton button {
        background-color: #1abc9c;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #16a085;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main Title
st.markdown("<h1 style='text-align: center;'> Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'>A Clinical Tool for Heart Disease Prediction</h5>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🔎 About This App")
    st.write("Advanced predictive tool for heart disease risk analysis.")
    st.info("Ensure accurate data input.")
    st.write("**Version:** 1.0 ")
    st.write("**Built for Academic & Clinical Research.")
    st.caption("Created for helping and diagnosing Heart Disease before it's too Late")

# Tabs
tab1, tab2 = st.tabs(["👩‍⚕️ Patient Form", "📊 Advanced Analysis"])

# --- Tab 1: Patient Form ---
with tab1:
    st.subheader("🤔 Basic Clinical Inputs")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Age (Years)', min_value=1, max_value=120, value=50)
        sex = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
        cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3],
                          format_func=lambda x: ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'][x])
        trestbps = st.number_input('Resting BP (mm Hg)', min_value=80, max_value=200, value=120)
        chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
        fbs = st.selectbox('Fasting Blood Sugar > 120?', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

    with col2:
        restecg = st.selectbox('Resting ECG', [0, 1, 2],
                               format_func=lambda x: ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy'][x])
        thalach = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=250, value=150)
        exang = st.selectbox('Exercise Induced Angina', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
        oldpeak = st.number_input('ST Depression (Oldpeak)', min_value=0.0, max_value=6.0, value=1.0, step=0.1)
        slope = st.selectbox('Slope of Peak Exercise ST Segment', [0, 1, 2],
                             format_func=lambda x: ['Upsloping', 'Flat', 'Downsloping'][x])
        ca = st.selectbox('Major Vessels Colored', [0, 1, 2, 3])
        thal = st.selectbox('Thalassemia', [0, 1, 2],
                            format_func=lambda x: ['Normal', 'Fixed Defect', 'Reversible Defect'][x])

    # Predict button
    if st.button('🔍 Assess Heart Disease Risk'):
        input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg,
                               thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)

        input_data_scaled = scaler.transform(input_data)
        prediction = svm_model.predict(input_data_scaled)

        st.success('✅ Analysis Complete!')

        st.subheader("📈 Result:")
        if prediction[0] == 1:
            st.error('⚠️ **High Risk Detected!** Immediate clinical evaluation is recommended.')
        else:
            st.success('✅ **Low Risk Detected!** Continue regular monitoring.')

        # Save for visualization
        st.session_state['patient_data'] = input_data.flatten()

# --- Tab 2: Advanced Analysis ---
with tab2:
    st.subheader("📊 Patient Risk Factors Pie Chart")

    if 'patient_data' in st.session_state:
        labels = ['Major Risk Factors(Cholestrol, Max Heart Rate, Old Peak)', 'Other Factors']

        # Example logic: assume Cholesterol, Oldpeak, Thalach are major risk factors
        data = st.session_state['patient_data']
        major_risk = data[4] + data[7] + data[9]  # Cholesterol + Max HR + Oldpeak
        other_risk = data.sum() - major_risk

        values = [major_risk, other_risk]

        fig = px.pie(values=values, names=labels, title='Risk Contribution', color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("📈 Please complete the prediction first in the 'Patient Form' tab.")

st.markdown("---")
st.caption("🔒 Disclaimer: This tool is intended for educational purposes only. Consult healthcare professionals for medical advice.")
