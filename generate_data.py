import pandas as pd
import numpy as np
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

n = 200  # samples per disease
label_noise = 0.05  # 5% label noise

def generate_data(disease_name, label):
    data = []
    for _ in range(n):
        # Basic patient profile
        age = np.random.randint(10, 70)
        gender = np.random.randint(0, 2)
        family_history = np.random.choice([0, 1], p=[0.6, 0.4])

        # Normal baseline values
        hemoglobin = round(np.random.normal(13.5, 1.2), 2)
        fetal_hb = round(np.random.normal(1.0, 0.5), 2)
        rdw_cv = round(np.random.normal(13.5, 1.0), 2)
        ferritin = round(np.random.normal(100, 30), 2)
        brca1_expr = round(np.random.normal(0.2, 0.1), 2)
        p53_mut = 0
        sweat_chloride = round(np.random.normal(40, 7), 2)
        sickled_rbc = round(np.random.normal(1.0, 0.8), 2)
        il6 = round(np.random.normal(5.0, 2.0), 2)

        # Disease-specific feature tuning
        if disease_name == "Thalassemia":
            hemoglobin = round(np.random.normal(8.0, 1.2), 2)
            fetal_hb = round(np.random.normal(15, 4), 2)
            ferritin = round(np.random.normal(50, 15), 2)
            rdw_cv = round(np.random.normal(17.5, 2.0), 2)

        elif disease_name == "Hemophilia":
            hemoglobin = round(np.random.normal(10.5, 1.2), 2)
            ferritin = round(np.random.normal(120, 20), 2)
            fetal_hb = round(np.random.normal(5, 3), 2)  # slight overlap with Thalassemia
            rdw_cv = round(np.random.normal(14.0, 1.0), 2)

        elif disease_name == "Breast Cancer":
            brca1_expr = round(np.random.normal(0.75, 0.15), 2)
            p53_mut = np.random.choice([0, 1], p=[0.3, 0.7])  # less distinct
            il6 = round(np.random.normal(20, 5), 2)

        elif disease_name == "Sickle Cell Anemia":
            hemoglobin = round(np.random.normal(7.0, 1.0), 2)
            fetal_hb = round(np.random.normal(18, 4), 2)
            sickled_rbc = round(np.random.normal(30, 6), 2)
            rdw_cv = round(np.random.normal(18, 2.0), 2)

        elif disease_name == "Cystic Fibrosis":
            sweat_chloride = round(np.random.normal(70, 6), 2)
            il6 = round(np.random.normal(18, 4), 2)
            ferritin = round(np.random.normal(130, 25), 2)

        # Add label noise
        final_label = label
        if random.random() < label_noise:
            final_label = random.choice([i for i in range(5) if i != label])

        # Collect row
        data.append([
            age, gender, family_history, hemoglobin, fetal_hb, rdw_cv,
            ferritin, brca1_expr, p53_mut, sweat_chloride,
            sickled_rbc, il6, final_label
        ])
    return data

# Label map
disease_labels = {
    0: "Thalassemia",
    1: "Hemophilia",
    2: "Breast Cancer",
    3: "Sickle Cell Anemia",
    4: "Cystic Fibrosis"
}

# Combine all diseases
all_data = []
for label, name in disease_labels.items():
    all_data.extend(generate_data(name, label))

# Create DataFrame
df = pd.DataFrame(all_data, columns=[
    "Age", "Gender", "Family_History", "Hemoglobin", "Fetal_Hemoglobin",
    "RDW_CV", "Serum_Ferritin", "BRCA1_Expression", "p53_Mutation",
    "Sweat_Chloride", "Sickled_RBC_Percent", "IL6_Level", "Disease"
])

# Save to file
df.to_csv("genetic_disease_dataset_advanced_realistic.csv", index=False)
print("🧬 Realistic advanced dataset saved as 'genetic_disease_dataset_advanced_realistic.csv'")
