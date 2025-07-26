from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Load model
model = joblib.load("disease_predictor_model.pkl")

# Disease label map
disease_labels = {
    0: "Thalassemia",
    1: "Hemophilia",
    2: "Breast Cancer",
    3: "Sickle Cell Anemia",
    4: "Cystic Fibrosis"
}

# Dummy disease data for explorer/detail (replace with DB if needed)
disease_info = [
    {
        "id": 0,
        "name": "Thalassemia",
        "description": "A blood disorder involving less than normal amounts of an oxygen-carrying protein.",
        "inheritance_pattern": "Autosomal recessive",
        "gene_involved": "HBB, HBA1, HBA2",
        "prevalence": "Common in Mediterranean, South Asian populations",
        "symptoms": ["Fatigue", "Pale skin", "Shortness of breath"],
        "risk_factors": ["Family history", "Certain ethnic backgrounds"]
    },
    {
        "id": 1,
        "name": "Hemophilia",
        "description": "A disorder in which blood doesn't clot normally.",
        "inheritance_pattern": "X-linked recessive",
        "gene_involved": "F8, F9",
        "prevalence": "Rare, mostly males",
        "symptoms": ["Excessive bleeding", "Easy bruising", "Joint pain"],
        "risk_factors": ["Family history", "Male gender"]
    },
    {
        "id": 2,
        "name": "Breast Cancer",
        "description": "A cancer that forms in the cells of the breasts.",
        "inheritance_pattern": "Multifactorial",
        "gene_involved": "BRCA1, BRCA2",
        "prevalence": "Common worldwide",
        "symptoms": ["Lump in breast", "Change in breast shape", "Skin changes"],
        "risk_factors": ["Family history", "BRCA mutations", "Age"]
    },
    {
        "id": 3,
        "name": "Sickle Cell Anemia",
        "description": "A group of inherited red blood cell disorders.",
        "inheritance_pattern": "Autosomal recessive",
        "gene_involved": "HBB",
        "prevalence": "Common in African, Mediterranean populations",
        "symptoms": ["Pain episodes", "Anemia", "Swelling in hands/feet"],
        "risk_factors": ["Family history", "Certain ethnic backgrounds"]
    },
    {
        "id": 4,
        "name": "Cystic Fibrosis",
        "description": "A disorder that causes severe damage to the lungs and digestive system.",
        "inheritance_pattern": "Autosomal recessive",
        "gene_involved": "CFTR",
        "prevalence": "Rare, mostly Caucasians",
        "symptoms": ["Persistent cough", "Frequent lung infections", "Poor growth"],
        "risk_factors": ["Family history", "Northern European descent"]
    }
]

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/diseases")
def diseases():
    return render_template("disease_explorer.html", diseases=disease_info)

@app.route("/disease/<int:id>")
def disease_detail(id):
    disease = next((d for d in disease_info if d["id"] == id), None)
    if not disease:
        flash("Disease not found.", "error")
        return redirect(url_for("diseases"))
    return render_template("disease_detail.html", disease=disease)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    prediction = None
    probability = None
    form_data = None
    if request.method == "POST":
        try:
            # Normalize and preprocess input features
            age = float(request.form.get("age", 0)) / 100  # Normalize age to 0-1 range
            gender = int(request.form.get("gender", 1))
            family_history = int(request.form.get("family_history", 0))
            
            # Clinical markers
            hemoglobin = min(float(request.form.get("hemoglobin", 0)), 1.0)  # Cap at 1.0
            fetal_hemoglobin = min(float(request.form.get("fetal_hemoglobin", 0)), 1.0)
            rdw_cv = min(float(request.form.get("rdw_cv", 0)), 1.0)
            serum_ferritin = min(float(request.form.get("serum_ferritin", 0)), 1.0)
            
            # Genetic markers
            brca1_expression = min(float(request.form.get("brca1_expression", 0)), 1.0)
            p53_mutation = int(request.form.get("p53_mutation", 0))
            
            # Disease-specific markers
            sweat_chloride = min(float(request.form.get("sweat_chloride", 0)), 1.0)
            sickled_rbc = min(float(request.form.get("sickled_rbc_percent", 0)), 1.0)
            il6_level = min(float(request.form.get("il6_level", 0)), 1.0)
            
            # Create feature vector with normalized values
            features = [
                age,
                gender,
                family_history,
                hemoglobin,
                fetal_hemoglobin,
                rdw_cv,
                serum_ferritin,
                brca1_expression,
                p53_mutation,
                sweat_chloride,
                sickled_rbc,
                il6_level
            ]
            # If your model expects more features, adjust accordingly.
            X = np.array(features).reshape(1, -1)
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0][pred]
            prediction = pred
            probability = proba
            result = True
            form_data = request.form
        except Exception as e:
            flash(f"Error in prediction: {str(e)}", "error")
    return render_template("predict.html", 
                         result=result, 
                         prediction=prediction, 
                         probability=probability, 
                         form_data=form_data,
                         disease_labels=disease_labels,
                         disease_info=disease_info)

@app.route("/contact", methods=["GET", "POST"])
def contact():
    success = False
    if request.method == "POST":
        # You can add email sending logic here
        success = True
    return render_template("contact.html", success=success)

if __name__ == "__main__":
    app.run(debug=True)
