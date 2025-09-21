from flask import Flask, render_template, request, redirect, url_for, flash, session
from functools import wraps
import joblib
import numpy as np
import pandas as pd
from models import User, supabase
from dotenv import load_dotenv
load_dotenv()
import os
from supabase import create_client



SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_KEY")


app = Flask(__name__)
app.secret_key = "supersecretkey"

# Custom login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def current_user():
    if 'user' in session:
        return session['user']
    return None

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

@app.route("/login", methods=["GET", "POST"])
def login():
    if 'user' in session:
        return redirect(url_for('home'))
    
    if request.method == "POST":
        email = request.form.get('email')
        password = request.form.get('password')
        
        try:
            auth_response = User.login(email, password)
            if auth_response and auth_response.user:
                # Store the complete session data
                session['access_token'] = auth_response.session.access_token
                session['refresh_token'] = auth_response.session.refresh_token
                
                # Get and store user data
                user_data = User.get_user_by_id(auth_response.user.id)
                if user_data:
                    session['user'] = user_data
                    next_page = request.args.get('next')
                    return redirect(next_page or url_for('home'))
                else:
                    raise Exception("Failed to fetch user data")
            else:
                flash('Invalid email or password', 'error')
        except Exception as e:
            flash(str(e), 'error')
    
    return render_template('login.html')

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if 'user' in session:
        return redirect(url_for('home'))
    
    if request.method == "POST":
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('signup.html')
        
        try:
            user = User.create_user(
                email=email,
                password=password,
                username=username,
                first_name=first_name,
                last_name=last_name
            )
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash(str(e), 'error')
            return render_template('signup.html')
    
    return render_template('signup.html')

@app.route("/logout")
@login_required
def logout():
    User.logout()
    session.pop('user', None)
    return redirect(url_for('home'))

@app.route("/")
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template("home.html")

@app.route("/diseases")
@login_required
def diseases():
    return render_template("disease_explorer.html", diseases=disease_info)

@app.route("/disease/<int:id>")
@login_required
def disease_detail(id):
    disease = next((d for d in disease_info if d["id"] == id), None)
    if not disease:
        flash("Disease not found.", "error")
        return redirect(url_for("diseases"))
    return render_template("disease_detail.html", disease=disease)

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    result = None
    prediction = None
    probability = None
    form_data = None
    result = None
    prediction = None
    probability = None
    form_data = None
    if request.method == "POST":
        try:
            # Get raw input values
            age = float(request.form.get("age", 0))
            gender = int(request.form.get("gender", 1))
            family_history = int(request.form.get("family_history", 0))
            hemoglobin = float(request.form.get("hemoglobin", 0))
            fetal_hemoglobin = float(request.form.get("fetal_hemoglobin", 0))
            rdw_cv = float(request.form.get("rdw_cv", 0))
            serum_ferritin = float(request.form.get("serum_ferritin", 0))
            brca1_expression = float(request.form.get("brca1_expression", 0))
            p53_mutation = int(request.form.get("p53_mutation", 0))
            sweat_chloride = float(request.form.get("sweat_chloride", 0))
            sickled_rbc = float(request.form.get("sickled_rbc_percent", 0))
            il6_level = float(request.form.get("il6_level", 0))

            # Flag (not reject) out-of-range values
            warnings = []
            if not (0 <= age <= 120): warnings.append("Unusual age: please check entry (years: 0–120).")
            if gender not in [0, 1]: warnings.append("Gender must be 0 (Female) or 1 (Male).")
            if family_history not in [0, 1]: warnings.append("Family history must be 0 or 1.")
            if not (3.0 <= hemoglobin <= 20.0): warnings.append("Hemoglobin outside typical range (3–20 g/dL).")
            if not (0 <= fetal_hemoglobin <= 100): warnings.append("Fetal Hemoglobin % outside typical range (0–100%).")
            if not (8 <= rdw_cv <= 30): warnings.append("RDW-CV outside typical range (8–30%).")
            if not (1 <= serum_ferritin <= 2000): warnings.append("Serum Ferritin outside typical range (1–2000 ng/mL).")
            if not (0 <= brca1_expression <= 1): warnings.append("BRCA1 expression outside 0–1 (check units).")
            if p53_mutation not in [0, 1]: warnings.append("p53 mutation must be 0 or 1.")
            if not (0 <= sweat_chloride <= 200): warnings.append("Sweat chloride outside typical range (0–200 mmol/L).")
            if not (0 <= sickled_rbc <= 100): warnings.append("Sickled RBC % outside 0–100%.")
            if not (0 <= il6_level <= 1000): warnings.append("IL-6 outside typical range (0–1000 pg/mL).")

            for warn in warnings:
                flash(warn, "warning")

            # Normalize for model input
            age_norm = age / 100
            hemoglobin_norm = (hemoglobin - 6) / (9.5 - 6)
            fetal_hemoglobin_norm = (fetal_hemoglobin - 7) / (18 - 7)
            rdw_cv_norm = (rdw_cv - 13) / (21 - 13)
            serum_ferritin_norm = (serum_ferritin - 20) / (60 - 20)
            brca1_expression_norm = brca1_expression / 0.4
            sweat_chloride_norm = (sweat_chloride - 30) / (60 - 30)
            sickled_rbc_norm = sickled_rbc / 2
            il6_level_norm = (il6_level - 1) / (9 - 1)

            # Create features with column names matching training data
            features_dict = {
                'Age': age_norm,
                'Gender': gender,
                'Family_History': family_history,
                'Hemoglobin': hemoglobin_norm,
                'Fetal_Hemoglobin': fetal_hemoglobin_norm,
                'RDW_CV': rdw_cv_norm,
                'Serum_Ferritin': serum_ferritin_norm,
                'BRCA1_Expression': brca1_expression_norm,
                'p53_Mutation': p53_mutation,
                'Sweat_Chloride': sweat_chloride_norm,
                'Sickled_RBC_Percent': sickled_rbc_norm,
                'IL6_Level': il6_level_norm
            }
            X = pd.DataFrame([features_dict])
            proba_all = model.predict_proba(X)[0]
            pred = model.predict(X)[0]
            # Prevent breast cancer prediction for males
            if gender == 1 and disease_labels.get(pred) == "Breast Cancer":
                breast_cancer_idx = [k for k, v in disease_labels.items() if v == "Breast Cancer"]
                if breast_cancer_idx:
                    proba_all[breast_cancer_idx[0]] = 0
                    pred = int(np.argmax(proba_all))
            proba = proba_all[pred]
            prediction = pred
            probability = proba
            
            # Calculate risk level based on multiple factors
            def calculate_risk_level(prob, warnings_count, family_hist, disease_specific_factors):
                # Base risk from probability
                if prob < 0.2:
                    risk = "Very Low"
                elif prob < 0.4:
                    risk = "Low"
                elif prob < 0.6:
                    risk = "Moderate"
                elif prob < 0.8:
                    risk = "High"
                else:
                    risk = "Very High"
                
                # Adjust for number of out-of-range values
                if warnings_count >= 3:
                    risk = "High" if risk == "Moderate" else risk
                
                # Adjust for family history
                if family_hist == 1 and risk in ["Low", "Moderate"]:
                    risk = "Moderate" if risk == "Low" else "High"
                
                # Adjust for disease-specific factors
                if disease_specific_factors:
                    if risk != "Very High":
                        risk = "High"
                
                return risk
            
            # Check for disease-specific risk factors
            disease_specific_risk = False
            predicted_disease = disease_labels.get(pred)
            
            if predicted_disease == "Thalassemia" and hemoglobin < 9:
                disease_specific_risk = True
            elif predicted_disease == "Sickle Cell Anemia" and sickled_rbc > 40:
                disease_specific_risk = True
            elif predicted_disease == "Breast Cancer" and (brca1_expression < 0.3 or p53_mutation == 1):
                disease_specific_risk = True
            elif predicted_disease == "Cystic Fibrosis" and sweat_chloride > 60:
                disease_specific_risk = True
            
            risk_level = calculate_risk_level(
                probability,
                len(warnings),
                family_history,
                disease_specific_risk
            )
            
            result = True
            form_data = request.form
            
            # Convert NumPy types to Python native types before storing in session
            session['prediction_result'] = {
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_level': risk_level,
                'result': bool(result),
                'form_data': dict(request.form)
            }
            return redirect(url_for('predict'))
        except Exception as e:
            flash(f"Error in prediction: {str(e)}", "error")
    # Only pass form_data if POST, otherwise None (so form is blank on refresh)
    if request.method == "POST":
        fd = form_data
    else:
        fd = None
    # If redirected after POST, get result from session
    prediction_result = session.pop('prediction_result', None)
    if prediction_result:
        return render_template("predict.html", 
                             result=prediction_result['result'], 
                             prediction=prediction_result['prediction'], 
                             probability=prediction_result['probability'],
                             risk_level=prediction_result['risk_level'], 
                             form_data=prediction_result['form_data'],
                             disease_labels=disease_labels,
                             disease_info=disease_info)
    return render_template("predict.html", 
                         result=None, 
                         prediction=None, 
                         probability=None,
                         risk_level=None, 
                         form_data=None,
                         disease_labels=disease_labels,
                         disease_info=disease_info)

@app.route("/contact", methods=["GET", "POST"])
@login_required
def contact():
    success = False
    user_data = session.get('user', {})
    name = f"{user_data.get('first_name', '')} {user_data.get('last_name', '')}".strip()
    email = user_data.get('email', '')
    
    if request.method == "POST":
        try:
            subject = request.form.get('subject')
            message = request.form.get('message')
            
            if not subject or not message:
                flash("Subject and message are required.", "error")
                return render_template("contact.html", success=False, name=name, email=email)

            # Insert directly using supabase client
            insert_data = {
                "user_id": user_data.get('id'),
                "subject": subject,
                "message": message,
                "email": email,
                "name": name
            }
            
            # Use the global supabase client from models
            result = supabase.table('contact_messages').insert(insert_data).execute()
            
            if result.data:
                success = True
                flash("Your message has been sent successfully!", "success")
            else:
                flash("Failed to send message.", "error")
                
        except Exception as e:
            flash("Error sending message. Please try again.", "error")
            
    return render_template("contact.html", success=success, name=name, email=email)

if __name__ == "__main__":
    app.run(debug=True)
