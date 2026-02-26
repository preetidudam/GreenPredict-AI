import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model/random_forest.pkl")

# Load climate data
climate_df = pd.read_csv("data/climate.csv")

st.set_page_config(page_title="GreenPredict-AI", layout="centered")

st.title("🌱 GreenPredict-AI")
st.subheader("AI-Based Plant Survival Prediction System")

st.markdown("Enter soil parameters and select a city to evaluate plant survival probability.")

# Soil Inputs
soil_type = st.selectbox(
    "Select Soil Type",
    ["Sandy", "Loamy", "Alluvial", "Lateritic", "Red loam"]
)

pH = st.number_input("Soil pH", min_value=4.0, max_value=10.0, value=6.5)
nitrogen = st.number_input("Nitrogen (kg/ha)", min_value=0.0, max_value=800.0, value=300.0)
phosphorus = st.number_input("Phosphorus (kg/ha)", min_value=0.0, max_value=100.0, value=25.0)
potassium = st.number_input("Potassium (kg/ha)", min_value=0.0, max_value=500.0, value=150.0)
organic_carbon = st.number_input("Organic Carbon (%)", min_value=0.0, max_value=2.0, value=0.7)
ec = st.number_input("Electrical Conductivity (dS/m)", min_value=0.0, max_value=5.0, value=0.5)

city = st.selectbox("Select Maharashtra City", climate_df["city"].unique())

selected_plant = st.selectbox(
    "Select Plant to Evaluate",
    ["Neem", "Banyan", "Peepal", "Mango", "Jamun", "Tamarind", "Arjun", "Gulmohar"]
)

if st.button("Predict Survival"):

    city_data = climate_df[climate_df["city"] == city]
    rainfall = float(city_data["rainfall"].values[0])
    temperature = float(city_data["avg_temp"].values[0])

    input_data = pd.DataFrame([{
        "pH": pH,
        "nitrogen": nitrogen,
        "phosphorus": phosphorus,
        "potassium": potassium,
        "organic_carbon": organic_carbon,
        "ec": ec,
        "rainfall": rainfall,
        "temperature": temperature,
        "soil_type": soil_type
    }])

    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_
    prob_dict = dict(zip(classes, probabilities))

    selected_prob = prob_dict[selected_plant] * 100
    best_plant = max(prob_dict, key=prob_dict.get)
    best_prob = prob_dict[best_plant] * 100

    st.markdown("---")
    st.subheader("📊 Results")

    st.write(f"**Survival Probability for {selected_plant}: {selected_prob:.2f}%**")

    if best_plant != selected_plant:
        st.warning(f"Better recommendation found: **{best_plant} ({best_prob:.2f}%)**")
    else:
        st.success("Selected plant is optimal for given conditions.")
