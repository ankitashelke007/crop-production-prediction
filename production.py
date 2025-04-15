import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from dateutil.relativedelta import relativedelta
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, request, render_template, redirect, url_for, session
import sqlite3

from werkzeug.security import generate_password_hash
# Connect to SQLite database
conn = sqlite3.connect('users.db')
cursor = conn.cursor()



# Create users table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
''')


# Commit changes and close connection
conn.commit()
conn.close()

print("Database reset successfully! Users can register and log in dynamically.")


app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
# 🔑 Set Secret Key for Session Management
app.secret_key = 'your_secret_key_here'  # Add this line here


# Load models
MODEL_PATHS = {
    "soyabean": "soyabean.pkl",
    "rice": "rice.pkl",
    "sugarcane": "sugarcane.pkl",
    "groundnut": "groundnut.pkl",
    "maize": "maize.pkl",
    "sunflower" : "sunflower.pkl"
}

model_paths = {
    'groundnut': 'models/groundnut_price_model.pkl',
    'maize': 'models/maize_price_model.pkl',
    'rice': 'models/rice_price_model.pkl',
    'soyabean': 'models/soyabean_price_model.pkl',
    'sunflower': 'models/sunflower_price_model.pkl',
    'wheat': 'models/new_wheat.pkl'
}


# Load reference CSV
df = pd.read_csv("final_crop_yield.csv")

# Rename columns to match expected names
df.rename(columns={
    "Dist Name": "dist_name",
    "Crop": "crop",
    "Area(1000 ha)": "area",
    "Production(1000 tons)": "production",
    "Total Rainfall": "total_rainfall",
    "Avg Temp": "avg_temp"
}, inplace=True)

# Validate required columns
expected_columns = {"dist_name", "crop", "area", "total_rainfall", "avg_temp"}
missing_columns = expected_columns - set(df.columns)
if missing_columns:
    raise KeyError(f"Missing columns in CSV: {missing_columns}")

# Normalize text columns
df["crop"] = df["crop"].str.strip().str.lower()
df["dist_name"] = df["dist_name"].str.strip().str.lower()

# Encoding
district_mapping = {name: idx for idx, name in enumerate(sorted(df["dist_name"].unique()))}
crop_mapping = {name: idx for idx, name in enumerate(sorted(df["crop"].unique()))}

# Scaling
scaler = StandardScaler()
X_train = df[['dist_name', 'crop', 'area', 'total_rainfall', 'avg_temp']].copy()
X_train.replace({'dist_name': district_mapping, 'crop': crop_mapping}, inplace=True)
scaler.fit(X_train)

# 🏠 Home Page: Redirects to login or dashboard
@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('dashboard'))  
    return redirect(url_for('login'))

# 📌 User Registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Username already exists!"
        finally:
            conn.close()
    return render_template('register.html')

# 🔑 User Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        if user and check_password_hash(user[2], password):
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return "Invalid credentials!"
    return render_template('login.html')

# 🚪 Logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# 🎛 Dashboard
@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route("/yield1", methods=["GET", "POST"])
def yield1():
    prediction = None
    yield_result = None
    error_message = None

    # Make sure these mappings and scaler/model paths are defined globally or above this route
    districts = list(district_mapping.keys())
    crops = list(crop_mapping.keys())

    if request.method == "POST":
        try:
            dist_name = request.form["dist_name"].strip().lower()
            crop = request.form["crop"].strip().lower()

            if dist_name not in district_mapping:
                error_message = f"Error: District '{dist_name}' not found in reference data."
                return render_template("final.html", error_message=error_message, districts=districts, crops=crops)

            if crop not in crop_mapping:
                error_message = f"Error: Crop '{crop}' not found in reference data."
                return render_template("final.html", error_message=error_message, districts=districts, crops=crops)

            area = float(request.form["area"])
            total_rainfall = float(request.form["total_rainfall"])
            avg_temp = float(request.form["avg_temp"])

            dist_encoded = district_mapping[dist_name]
            crop_encoded = crop_mapping[crop]

            if crop in MODEL_PATHS:
                model_path = MODEL_PATHS[crop]
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
            else:
                error_message = "Invalid crop selection."
                return render_template("final.html", error_message=error_message, districts=districts, crops=crops)

            input_data = np.array([[dist_encoded, crop_encoded, area, total_rainfall, avg_temp]])
            input_data = scaler.transform(input_data)

            prediction = model.predict(input_data)[0]  # production in 1000 tons
            yield_result = prediction / area

        except Exception as e:
            error_message = f"Error: {str(e)}"

    return render_template(
        "final.html",
        prediction=prediction,
        yield_result=yield_result,
        error_message=error_message,
        districts=districts,
        crops=crops
    )

@app.route('/price')
def price():
    return render_template('index.html', crops=model_paths.keys())

@app.route('/predict/<crop_name>')
def predict(crop_name):
    if crop_name not in model_paths:
        return f"Model for {crop_name} not found", 404

    with open(model_paths[crop_name], 'rb') as f:
        model = pickle.load(f)

    future_dates = pd.date_range(start=datetime.now(), periods=12, freq='M')
    future_df = pd.DataFrame({'ds': future_dates})
    forecast = model.predict(future_df)
    forecast = forecast[['ds', 'yhat']].round(2)

    return render_template('prediction.html', crop=crop_name.title(), forecast=forecast.to_dict(orient='records'))

if __name__ == "__main__":
    app.run(debug=True)
