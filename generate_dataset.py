import pandas as pd
import numpy as np
import random

species_data = {
    "Neem": (450,1150,20,40,6.0,8.5,["Sandy","Loamy"]),
    "Banyan": (500,4000,9,40,6.0,8.5,["Alluvial","Loamy"]),
    "Peepal": (600,3000,10,40,6.0,8.5,["Alluvial","Loamy"]),
    "Mango": (750,1900,15,40,5.5,7.5,["Loamy"]),
    "Jamun": (700,3000,10,40,6.0,9.0,["Alluvial","Lateritic"]),
    "Tamarind": (750,1900,15,40,5.5,6.8,["Red loam","Sandy"]),
    "Arjun": (750,1900,20,35,6.5,7.5,["Alluvial"]),
    "Gulmohar": (700,1200,14,35,6.0,8.0,["Sandy","Loamy"])
}

rows = []

def generate_mid_weighted(min_val, max_val):
    mid = (min_val + max_val) / 2
    return round(np.random.normal(mid, (max_val-min_val)/6),2)

for plant, values in species_data.items():
    min_r, max_r, min_t, max_t, min_pH, max_pH, soils = values
    
    for i in range(400):
        row = {
            "soil_type": random.choice(soils),
            "pH": generate_mid_weighted(min_pH,max_pH),
            "nitrogen": random.randint(100,700),
            "phosphorus": random.randint(5,60),
            "potassium": random.randint(50,400),
            "organic_carbon": round(random.uniform(0.3,1.2),2),
            "ec": round(random.uniform(0.1,3.5),2),
            "rainfall": generate_mid_weighted(min_r,max_r),
            "temperature": generate_mid_weighted(min_t,max_t),
            "plant": plant
        }
        rows.append(row)

df = pd.DataFrame(rows)

df.to_csv("data/plant_data.csv",index=False)

print("New dataset created:", df.shape)