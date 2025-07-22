import streamlit as st
import numpy as np
import pickle
from pathlib import Path

# --- Configuration: Input Ranges ---
INPUT_RANGES = {
    "Glucose": {"min": 50.0, "max": 300.0, "label": "Glucose Level (mg/dL)"},
    "Weight": {"min": 20.0, "max": 250.0, "label": "Weight (kg)"},
    "Height": {"min": 100.0, "max": 250.0, "label": "Height (cm)"},
    "Age": {"min": 1, "max": 120, "label": "Age (years)"},
}

# --- Model Loading ---
@st.cache_resource
def load_pipeline_model():
    base_dir = Path(__file__).parent.resolve()
    model_path = base_dir / 'Diabetesmodel.pkl'

    try:
        with open(model_path, 'rb') as f:
            pipeline_model = pickle.load(f)
        return pipeline_model
    except FileNotFoundError as e:
        st.error(f"Error loading the model. Make sure 'Diabetesmodel.pkl' is in the same directory as this Streamlit app file. Details: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the model. Details: {e}")
        st.stop()

pipeline_model = load_pipeline_model()

def calculate_bmi(weight, height_cm):
    height_m = height_cm / 100
    if height_m <= 0:
        return 0
    return weight / (height_m ** 2)

# --- Streamlit UI ---
st.set_page_config(page_title="AI Diabetes Risk Assessment", layout="centered")
st.title("AI Diabetes Risk Assessment")
st.markdown("Please enter your health details below to get a risk assessment.")

with st.form("diabetes_prediction_form"):
    st.header("Patient Information")

    glucose = st.number_input(
        INPUT_RANGES["Glucose"]["label"],
        min_value=INPUT_RANGES["Glucose"]["min"],
        max_value=INPUT_RANGES["Glucose"]["max"],
        value=120.0,
        help=f"Enter your glucose level in mg/dL (between {INPUT_RANGES['Glucose']['min']} and {INPUT_RANGES['Glucose']['max']})"
    )
    weight = st.number_input(
        INPUT_RANGES["Weight"]["label"],
        min_value=INPUT_RANGES["Weight"]["min"],
        max_value=INPUT_RANGES["Weight"]["max"],
        value=70.0,
        help=f"Enter your weight in kilograms (between {INPUT_RANGES['Weight']['min']} and {INPUT_RANGES['Weight']['max']})"
    )
    height_cm = st.number_input(
        INPUT_RANGES["Height"]["label"],
        min_value=INPUT_RANGES["Height"]["min"],
        max_value=INPUT_RANGES["Height"]["max"],
        value=170.0,
        help=f"Enter your height in centimeters (between {INPUT_RANGES['Height']['min']} and {INPUT_RANGES['Height']['max']})"
    )
    age = st.number_input(
        INPUT_RANGES["Age"]["label"],
        min_value=INPUT_RANGES["Age"]["min"],
        max_value=INPUT_RANGES["Age"]["max"],
        value=30,
        help=f"Enter your age in years (between {INPUT_RANGES['Age']['min']} and {INPUT_RANGES['Age']['max']})"
    )

    family_history_option = st.selectbox(
        "Family history of diabetes?",
        ("No family history", "One close relative", "Multiple relatives"),
        help="Select the option that best describes your family's history of diabetes. This contributes to your genetic risk assessment."
    )

    submitted = st.form_submit_button("Predict Diabetes Risk")

    if submitted:
        errors = []
        bmi = calculate_bmi(weight, height_cm)
        if bmi <= 0:
            errors.append("Invalid height entered. Height must be a positive value.")
        elif not (10.0 <= bmi <= 60.0):
            errors.append(f"Calculated BMI ({bmi:.2f}) is outside a realistic range (10-60). Please check weight/height inputs.")

        dp_mapping = {
            "No family history": 0.1,
            "One close relative": 0.5,
            "Multiple relatives": 1.0
        }
        diabetes_pedigree_function = dp_mapping[family_history_option]

        if errors:
            for error in errors:
                st.error(error)
        else:
            try:
                # Input feature order matches model training: [Glucose, BMI, Age, DiabetesPedigreeFunction]
                features_for_prediction = np.array([[glucose, bmi, age, diabetes_pedigree_function]])
                prediction = pipeline_model.predict(features_for_prediction)[0]

                st.subheader("Prediction Result:")
                if prediction == 1:
                    st.error("Based on the provided information, you are **likely to have diabetes.**")
                    st.warning("Please consult a healthcare professional for accurate diagnosis and advice.")
                else:
                    st.success("Based on the provided information, you are **not likely to have diabetes.**")
                    st.info("Regular check-ups and a healthy lifestyle are always recommended.")
            except Exception as e:
                st.error(f"An error occurred during prediction. Please check your inputs. Details: {e}")
