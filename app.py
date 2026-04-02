"""
Iris Intelligence - Machine Learning Web Application
This module serves as the primary inference engine for Iris species classification.
"""

import os

from flask import Flask, jsonify, render_template, request # type: ignore
import joblib           # type: ignore
import numpy as np       # type: ignore
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier         # type: ignore



# Initialize the Flask application
app = Flask(__name__, 
            template_folder='templates', 
            static_folder='static')

# Global variables for model and configuration
model = None
feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
target_names = ['setosa', 'versicolor', 'virginica']

def load_ml_resources():
    """Initializes and loads the machine learning model and metadata."""
    global model, feature_names, target_names
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'iris_model.joblib')
    info_path = os.path.join(script_dir, 'model_info.joblib')
    
    print("🔄 Loading Iris AI model...")
    try:
        if os.path.exists(model_path) and os.path.exists(info_path):
            model = joblib.load(model_path)
            model_info = joblib.load(info_path)
            feature_names = model_info.get('feature_names', feature_names)
            target_names = model_info.get('target_names', target_names)
            print("✅ Model loaded successfully!")
        else:
            print("⚠️ Model files not found. Inference engine disabled.")
    except Exception as e:
        print(f"❌ Critical error during model initialization: {str(e)}")
        model = None

# Persistence Layer for Prediction History
HISTORY_FILE = 'prediction_history.json'

def load_history():
    """Loads prediction history from a persistent JSON file."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)[-10:] # Return last 10
        except: return []
    return []

def save_prediction(data):
    """Saves a new prediction entry to history."""
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except: pass
    
    data['timestamp'] = datetime.now().strftime("%H:%M:%S")
    history.append(data)
    
    # Keep last 50 records
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history[-50:], f, indent=2)

# Initialize resources on startup
load_ml_resources()

# Application Routes
@app.route('/')
def home():
    """
    Render the home page with the input form and historical data.
    """
    history = load_history()
    return render_template('index.html', history=history)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Robust prediction handler with input validation and species analysis.
    """
    if model is None:
        return render_template('index.html', error="AI Engine is offline. Please run training script.")

    try:
        # Securely retrieve features from form
        sl = request.form.get('sepal_length')
        sw = request.form.get('sepal_width')
        pl = request.form.get('petal_length')
        pw = request.form.get('petal_width')

        # Check for missing values
        if not all([sl, sw, pl, pw]):
             return render_template('index.html', error="Incomplete morphology data provided.")

        # Convert to float
        input_data = [float(x) for x in [sl, sw, pl, pw]]
        
        # Validation range
        if any(x < 0 or x > 15 for x in input_data):
            return render_template('index.html', error="Botanical measurements out of logical bounds (0-15cm).")

        # Execute Inference
        features = np.array([input_data])
        prediction_idx = model.predict(features)[0]
        prediction = target_names[prediction_idx]
        probabilities = model.predict_proba(features)[0]

        # Structure probabilities
        prob_dict = {species: round(prob * 100, 1) for species, prob in zip(target_names, probabilities)}
        
        # Sort results by confidence
        sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

        # Persistent Logging
        save_prediction({
            'prediction': prediction,
            'features': input_data,
            'confidence': max(probabilities) * 100
        })

        history = load_history()

        return render_template('index.html', 
                             prediction=prediction,
                             probabilities=sorted_probs,
                             sepal_length=sl,
                             sepal_width=sw,
                             petal_length=pl,
                             petal_width=pw,
                             history=history)

    except ValueError:
        return render_template('index.html', error="Invalid numerical format in morphological data.")
    except Exception as e:
        print(f"🤖 Inference error: {str(e)}")
        return render_template('index.html', error="Interference detected in neural inference path.")

# API Endpoints
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    JSON API endpoint for predictions.
    Useful if you want to call your model from another application.
    """
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Extract features
        features = np.array([[data['sepal_length'],
                            data['sepal_width'],
                            data['petal_length'],
                            data['petal_width']]])
        
        # Make prediction
        prediction_idx = model.predict(features)[0]
        prediction = target_names[prediction_idx]
        probabilities = model.predict_proba(features)[0].tolist()
        
        # Return JSON response
        return jsonify({
            'prediction': prediction,
            'probabilities': dict(zip(target_names, probabilities)),
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Utility Routes
@app.route('/test')
def test():
    """
    Test route with sample predictions.
    """
    if model is None:
        return "Model not loaded. Please run train.py first."
    
    # Test samples (typical values for each species)
    test_samples = [
        [5.1, 3.5, 1.4, 0.2],  # Setosa
        [6.0, 2.7, 4.2, 1.3],  # Versicolor
        [6.7, 3.1, 5.6, 2.4],  # Virginica
    ]
    
    results = []
    for sample in test_samples:
        pred = model.predict([sample])[0]
        prob = model.predict_proba([sample])[0]
        results.append({
            'features': sample,
            'prediction': target_names[pred],
            'probabilities': dict(zip(target_names, prob.tolist()))
        })
    
    return jsonify(results)

if __name__ == '__main__':
    # Initial startup log
    print(f"\n🌸 Iris Intelligence is coming online...")
    print(f"📊 Analyzing morphology using Random Forest engine.")
    print(f"🌐 Server active at: http://127.0.0.1:8080")
    
    # Using 8080 as a production-friendly fallback to the default 5000
    app.run(debug=True, host='127.0.0.1', port=8080)