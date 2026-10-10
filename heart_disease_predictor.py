import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('best_xgboost_model.joblib')

# Define feature options
diabetes_options = {
    0: 'No Diabetes (0)',
    1: 'Diabetes (1)'
}

tnm_stage_options = {
    1: 'TNM Stage 1 (1)',
    2: 'TNM Stage 2 (2)',
    3: 'TNM Stage 3 (3)',
    4: 'TNM Stage 4 (4)'
}

rln_options = {
    0: 'No (0)',
    1: 'Yes (1)'
}

# Define feature names
feature_names = [
    "Diabetes",
    "FPG",
    "CA199",
    "TNM_Stage",
    "CTI",
    "HDL",
    "TC",
    "RLN"
]

# Streamlit user interface
st.title("Death Risk Predictor")

# Diabetes: categorical selection
diabetes = st.selectbox(
    "Diabetes:",
    options=list(diabetes_options.keys()),
    format_func=lambda x: diabetes_options[x]
)

# FPG: numerical input
fpg = st.number_input(
    "FPG:",
    min_value=0.0,
    max_value=50.0,
    value=5.80,
    step=0.01
)

# CA199: numerical input
ca199 = st.number_input(
    "CA199:",
    min_value=0.0,
    max_value=10000.0,
    value=30.0,
    step=0.01
)

# TNM Stage: categorical selection
tnm_stage = st.selectbox(
    "TNM Stage:",
    options=list(tnm_stage_options.keys()),
    format_func=lambda x: tnm_stage_options[x]
)

# CTI: numerical input
cti = st.number_input(
    "CTI:",
    min_value=0.0,
    max_value=100.0,
    value=9.0,
    step=0.01
)

# HDL: numerical input
hdl = st.number_input(
    "HDL:",
    min_value=0.0,
    max_value=20.0,
    value=1.20,
    step=0.01
)

# TC: numerical input
tc = st.number_input(
    "TC:",
    min_value=0.0,
    max_value=50.0,
    value=5.00,
    step=0.01
)

# RLN: categorical selection
rln = st.selectbox(
    "RLN:",
    options=list(rln_options.keys()),
    format_func=lambda x: rln_options[x]
)

# Process inputs and make predictions
feature_values = [
    diabetes,
    fpg,
    ca199,
    tnm_stage,
    cti,
    hdl,
    tc,
    rln
]

features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, the patient has a high risk of death. "
            f"The model predicts that the probability of death is {probability:.1f}%. "
            "This result is only an estimate and should not be considered a final diagnosis. "
            "It is recommended to consult a qualified doctor for further clinical evaluation "
            "and comprehensive treatment planning."
        )
    else:
        advice = (
            f"According to our model, the patient has a low risk of death. "
            f"The model predicts that the probability of survival is {probability:.1f}%. "
            "However, this result is only an estimate and does not guarantee the clinical outcome. "
            "Regular follow-up and professional medical evaluation are still recommended."
        )

    st.write(advice)

    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        pd.DataFrame(
            [feature_values],
            columns=feature_names
        )
    )

    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        pd.DataFrame(
            [feature_values],
            columns=feature_names
        ),
        matplotlib=True
    )

    plt.savefig(
        "shap_force_plot.png",
        bbox_inches='tight',
        dpi=1200
    )

    st.image("shap_force_plot.png")
