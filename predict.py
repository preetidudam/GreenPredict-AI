import pandas as pd
import joblib

# Load trained model
model = joblib.load("model/random_forest.pkl")

# Load climate data
climate_df = pd.read_csv("data/climate.csv")

print("\n🌱 GREENPREDICTAI - Plant Survival Prediction System\n")

# 1️⃣ Soil Inputs
soil_type = input("Enter soil type (Sandy, Loamy, Alluvial, Lateritic, Red loam): ")
pH = float(input("Enter soil pH: "))
nitrogen = float(input("Enter Nitrogen (kg/ha): "))
phosphorus = float(input("Enter Phosphorus (kg/ha): "))
potassium = float(input("Enter Potassium (kg/ha): "))
organic_carbon = float(input("Enter Organic Carbon (%): "))
ec = float(input("Enter Electrical Conductivity (dS/m): "))

# 2️⃣ City Input
city = input("Enter Maharashtra city name (exactly as in climate.csv): ")

city_data = climate_df[climate_df["city"] == city]

if city_data.empty:
    print("❌ City not found.")
    exit()

rainfall = float(city_data["rainfall"].values[0])
temperature = float(city_data["avg_temp"].values[0])

# 3️⃣ Plant Selection
selected_plant = input("Enter plant to evaluate (Neem, Banyan, Peepal, Mango, Jamun, Tamarind, Arjun, Gulmohar): ")

# 4️⃣ Prepare Input
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

model_columns = model.feature_names_in_
input_data = input_data.reindex(columns=model_columns, fill_value=0)

# 5️⃣ Get probabilities
probabilities = model.predict_proba(input_data)[0]
classes = model.classes_

prob_dict = dict(zip(classes, probabilities))

selected_prob = prob_dict.get(selected_plant, 0) * 100
best_plant = max(prob_dict, key=prob_dict.get)
best_prob = prob_dict[best_plant] * 100

print("\n📊 Survival Probability for", selected_plant, ":", round(selected_prob,2), "%")

if best_plant != selected_plant:
    print("\n🌳 Better Recommendation Found!")
    print("Recommended Plant:", best_plant)
    print("Estimated Survival Probability:", round(best_prob,2), "%")
else:
    print("\n✅ Selected plant is optimal for given conditions.")