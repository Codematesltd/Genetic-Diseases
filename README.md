# Genetic Disease Predictor

## Overview

Genetic Disease Predictor is a web-based application that uses machine learning to predict the risk of five major genetic diseases based on clinical and genetic parameters. The platform also provides educational resources, a disease explorer, and a contact form for user support.

---

## Features

- **User Authentication:** Secure signup, login, and logout using Supabase.
- **Disease Risk Prediction:** ML-powered risk assessment for Thalassemia, Hemophilia, Breast Cancer, Sickle Cell Anemia, and Cystic Fibrosis.
- **Disease Explorer:** Browse detailed information about genetic diseases.
- **Contact Form:** Users can send messages to the support team.
- **Responsive UI:** Built with Tailwind CSS for modern, mobile-friendly design.

---

## Modules

### 1. `app.py`
Main Flask application. Handles routing, form processing, model inference, user session management, and rendering templates.

- **Routes:**
  - `/login`, `/signup`, `/logout`: User authentication.
  - `/`: Home page (requires login).
  - `/diseases`: Disease explorer.
  - `/disease/<id>`: Disease detail.
  - `/predict`: Disease risk prediction form and results.
  - `/contact`: Contact form.

- **Model Loading:** Loads `disease_predictor_model.pkl` using `joblib`.
- **Session Management:** Uses Flask session for user state.

### 2. `models.py`
Handles user management and Supabase integration.

- **Supabase Client:** Initialized using credentials from `.env`.
- **User Class:** Methods for creating users, logging in, fetching user data, and logging out.
- **Contact Messages:** Stores contact form submissions in Supabase.

### 3. Templates (`templates/`)
HTML templates rendered by Flask using Jinja2.

- `base.html`: Layout and navigation.
- `home.html`: Landing page.
- `login.html`, `signup.html`: Authentication forms.
- `predict.html`: Prediction form and results.
- `disease_explorer.html`: List of diseases.
- `disease_detail.html`: Disease details.
- `contact.html`: Contact form.

### 4. Static Files (`static/`)
- **main.js:** Custom JavaScript for UI interactions (e.g., mobile menu).
- **Tailwind CSS & FontAwesome:** Loaded via CDN for styling and icons.

### 5. Model File
- `disease_predictor_model.pkl`: Trained RandomForest model for disease prediction.

### 6. Environment File
- `.env`: Stores Supabase URL and API key.

---

## Libraries Used

- **Flask:** Web framework for Python.
- **joblib:** For loading the ML model.
- **numpy, pandas:** Data manipulation and preprocessing.
- **supabase-py:** Python client for Supabase (database and authentication).
- **werkzeug:** Utilities for WSGI applications (used by Flask).
- **dotenv:** Loads environment variables from `.env`.
- **Tailwind CSS:** Utility-first CSS framework for styling.
- **FontAwesome:** Icon library.

---

## How It Works

1. **User Registration/Login:** Users create accounts and log in via Supabase authentication.
2. **Disease Prediction:** Users enter clinical/genetic data. The backend normalizes inputs and predicts disease risk using the ML model.
3. **Disease Explorer:** Users browse genetic diseases and view detailed information.
4. **Contact:** Authenticated users can send messages to the support team, stored in Supabase.

---

## Input Features for Prediction

| Feature Name          | Type   | Description                                 |
|----------------------|--------|---------------------------------------------|
| Age                  | int    | Patient age (normalized)                    |
| Gender               | int    | 0 = Male, 1 = Female                        |
| Family_History       | int    | 0 = No, 1 = Yes                             |
| Hemoglobin           | float  | g/dL (normalized)                           |
| Fetal_Hemoglobin     | float  | % (normalized)                              |
| RDW_CV               | float  | % (normalized)                              |
| Serum_Ferritin       | float  | ng/mL (normalized)                          |
| BRCA1_Expression     | float  | 0.0–1.0 (normalized)                        |
| p53_Mutation         | int    | 0 = Normal, 1 = Mutated                     |
| Sweat_Chloride       | float  | mmol/L (normalized)                         |
| Sickled_RBC_Percent  | float  | % (normalized)                              |
| IL6_Level            | float  | pg/mL (normalized)                          |

---

## Setup Instructions

1. **Clone the repository**
2. **Install dependencies**
    ```
    pip install -r requirements.txt
    ```
3. **Configure Supabase**
    - Add your Supabase credentials to `.env`:
      ```
      SUPABASE_URL=your_supabase_url
      SUPABASE_KEY=your_supabase_key
      ```
4. **Run the application**
    ```
    python app.py
    ```
5. **Access the app**
    - Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Disclaimer

This tool is for research and educational purposes only. It does not provide medical advice. Always consult healthcare professionals for diagnosis and treatment.

---

## Contact

For questions or support, use the contact form in the app or email: `geneticdisease18@gmail.com`

