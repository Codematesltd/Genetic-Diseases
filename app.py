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

def get_user_client():
    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    token = session.get('access_token')
    if token:
        client.postgrest.auth(token)
    return client

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
            if auth_response.user:
                user_data = User.get_user_by_id(auth_response.user.id)
                session['user'] = user_data
                # Store access token for authenticated requests
                session['access_token'] = auth_response.session.access_token
                next_page = request.args.get('next')
                return redirect(next_page or url_for('home'))
        except Exception as e:
            flash('Invalid email or password', 'error')
    
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
            X = np.array(features).reshape(1, -1)
            proba_all = model.predict_proba(X)[0]
            pred = model.predict(X)[0]
            # Prevent breast cancer prediction for males
            if gender == 1 and disease_labels.get(pred) == "Breast Cancer":
                # Set probability of breast cancer to 0 and pick next highest
                breast_cancer_idx = [k for k, v in disease_labels.items() if v == "Breast Cancer"]
                if breast_cancer_idx:
                    proba_all[breast_cancer_idx[0]] = 0
                    pred = int(np.argmax(proba_all))
            proba = proba_all[pred]
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
@login_required
def contact():
    success = False
    name = ""
    email = ""
    if 'user' in session:
        name = f"{session['user'].get('first_name', '')} {session['user'].get('last_name', '')}".strip()
        email = session['user'].get('email', '')

    if request.method == "POST":
        try:
            subject = request.form.get("subject")
            message = request.form.get("message")
            user_id = session['user']['id']
            # Use pre-filled name and email
            # If user edits, use form value
            name = request.form.get("name", name)
            email = request.form.get("email", email)

            # Store in Supabase using authenticated client
            user_client = get_user_client()
            user_client.table('contact_messages').insert({
                "user_id": user_id,
                "subject": subject,
                "message": message,
                "email": email,
                "name": name
            }).execute()
            
            success = True
            flash("Your message has been sent successfully!", "success")
        except Exception as e:
            flash(f"Error sending message: {str(e)}", "error")
            
    return render_template("contact.html", success=success, name=name, email=email)

if __name__ == "__main__":
    app.run(debug=True)
