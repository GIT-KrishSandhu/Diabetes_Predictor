from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
import requests
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

# Global variables for model and scaler
model = None
scaler = None
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

def load_and_prepare_data():
    """Load diabetes dataset from URL and prepare for training"""
    try:
        # Load data from the provided URL
        url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/diabetes-sw7GApN5hqM8jCwRFwJXFcmWbsd0io.csv"
        df = pd.read_csv('static/diabetes.csv')
        
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First few rows:\n{df.head()}")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Prepare features and target
        X = df[feature_names]
        y = df['Outcome']
        
        # Handle missing values (replace 0s with median for certain features)
        features_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for feature in features_to_replace:
            if feature in X.columns:
                X[feature] = X[feature].replace(0, X[feature].median())
        
        return X, y, df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def train_model():
    """Train the diabetes prediction model"""
    global model, scaler
    
    X, y, df = load_and_prepare_data()
    if X is None:
        return None
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print("Model Training Complete!")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1_score']:.3f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.3f}")
    
    # Feature importance
    feature_importance = dict(zip(feature_names, model.feature_importances_))
    print(f"Feature Importance: {feature_importance}")
    
    # Save model and scaler
    joblib.dump(model, 'diabetes_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    return metrics, feature_importance, df

def load_model():
    """Load pre-trained model and scaler"""
    global model, scaler
    
    try:
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler.pkl')
        print("Model and scaler loaded successfully!")
        return True
    except FileNotFoundError:
        print("Model files not found. Training new model...")
        return False

def get_risk_factors_and_recommendations(data, prediction_proba):
    """Generate risk factors and recommendations based on input data"""
    factors = []
    recommendations = []
    
    # Age factor
    if data['Age'] > 45:
        factors.append('Advanced age (>45 years)')
        recommendations.append('Regular health screenings recommended due to age')
    elif data['Age'] > 35:
        factors.append('Age factor (>35 years)')
        recommendations.append('Monitor health indicators regularly')
    
    # BMI factor
    if data['BMI'] > 30:
        factors.append('Obesity (BMI > 30)')
        recommendations.append('Consider weight management through diet and exercise')
    elif data['BMI'] > 25:
        factors.append('Overweight (BMI > 25)')
        recommendations.append('Maintain healthy weight through balanced diet')
    
    # Glucose factor
    if data['Glucose'] > 140:
        factors.append('High glucose levels (>140 mg/dL)')
        recommendations.append('Monitor blood sugar levels regularly and consult a doctor')
    elif data['Glucose'] > 120:
        factors.append('Elevated glucose levels (>120 mg/dL)')
        recommendations.append('Monitor blood sugar levels and maintain healthy diet')
    
    # Blood pressure factor
    if data['BloodPressure'] > 90:
        factors.append('High blood pressure (>90 mmHg)')
        recommendations.append('Monitor blood pressure and consider lifestyle changes')
    elif data['BloodPressure'] > 80:
        factors.append('Elevated blood pressure (>80 mmHg)')
        recommendations.append('Maintain healthy blood pressure through exercise')
    
    # Insulin factor
    if data['Insulin'] > 120:
        factors.append('High insulin levels')
        recommendations.append('Consult healthcare provider about insulin levels')
    
    # Diabetes pedigree factor
    if data['DiabetesPedigreeFunction'] > 0.5:
        factors.append('Family history of diabetes')
        recommendations.append('Regular screening due to family history')
    
    # Pregnancies factor
    if data['Pregnancies'] > 3:
        factors.append('Multiple pregnancies')
        recommendations.append('Regular health monitoring recommended')
    
    # Add general recommendations if no specific factors
    if len(factors) == 0:
        factors.append('No significant risk factors identified')
        recommendations.append('Maintain current healthy lifestyle')
    
    # Add general healthy lifestyle recommendations
    if len(recommendations) < 3:
        recommendations.extend([
            'Maintain a balanced diet rich in vegetables and whole grains',
            'Engage in regular physical activity (150 minutes per week)',
            'Schedule regular health screenings'
        ])
    
    return factors[:5], recommendations[:5]  # Limit to 5 items each

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict_page():
    """Serve the prediction page"""
    return render_template('predict.html')

@app.route('/dashboard')
def dashboard():
    """Serve the dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for diabetes prediction"""
    try:
        # Get data from request
        data = request.json
        
        # Validate required fields
        required_fields = ['pregnancies', 'glucose', 'bloodPressure', 'skinThickness', 
                          'insulin', 'bmi', 'diabetesPedigree', 'age']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Prepare input data
        input_data = {
            'Pregnancies': data['pregnancies'],
            'Glucose': data['glucose'],
            'BloodPressure': data['bloodPressure'],
            'SkinThickness': data['skinThickness'],
            'Insulin': data['insulin'],
            'BMI': data['bmi'],
            'DiabetesPedigreeFunction': data['diabetesPedigree'],
            'Age': data['age']
        }
        
        # Create feature array
        features = np.array([[input_data[feature] for feature in feature_names]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # Calculate risk probability (0-100%)
        risk_probability = prediction_proba[1] * 100
        
        # Determine risk level
        if risk_probability < 30:
            risk_level = 'Low'
        elif risk_probability < 60:
            risk_level = 'Moderate'
        else:
            risk_level = 'High'
        
        # Get risk factors and recommendations
        factors, recommendations = get_risk_factors_and_recommendations(input_data, risk_probability)
        
        # Prepare response
        response = {
            'prediction': int(prediction),
            'probability': round(risk_probability, 1),
            'risk_level': risk_level,
            'factors': factors,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/api/model-metrics')
def model_metrics():
    """API endpoint to get model performance metrics"""
    try:
        # Load data and retrain if needed to get metrics
        X, y, df = load_and_prepare_data()
        if X is None:
            return jsonify({'error': 'Failed to load data'}), 500
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': round(accuracy_score(y_test, y_pred) * 100, 1),
            'precision': round(precision_score(y_test, y_pred) * 100, 1),
            'recall': round(recall_score(y_test, y_pred) * 100, 1),
            'f1_score': round(f1_score(y_test, y_pred) * 100, 1),
            'auc_roc': round(roc_auc_score(y_test, y_pred_proba) * 100, 1)
        }
        
        # Feature importance
        feature_importance = [
            {'feature': feature, 'importance': round(importance * 100, 1)}
            for feature, importance in zip(feature_names, model.feature_importances_)
        ]
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        # Dataset statistics
        stats = {
            'total_samples': len(df),
            'positive_cases': int(df['Outcome'].sum()),
            'negative_cases': int(len(df) - df['Outcome'].sum()),
            'positive_rate': round(df['Outcome'].mean() * 100, 1)
        }
        
        return jsonify({
            'metrics': metrics,
            'feature_importance': feature_importance,
            'dataset_stats': stats
        })
    
    except Exception as e:
        print(f"Metrics error: {e}")
        return jsonify({'error': 'Failed to get metrics'}), 500

@app.route('/api/dataset-analysis')
def dataset_analysis():
    """API endpoint for dataset analysis"""
    try:
        X, y, df = load_and_prepare_data()
        if df is None:
            return jsonify({'error': 'Failed to load data'}), 500
        
        # Basic statistics
        analysis = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'summary_stats': df.describe().to_dict(),
            'outcome_distribution': df['Outcome'].value_counts().to_dict(),
            'correlation_matrix': df.corr().to_dict()
        }
        
        return jsonify(analysis)
    
    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({'error': 'Failed to analyze dataset'}), 500

if __name__ == '__main__':
    # Initialize the model
    if not load_model():
        print("Training new model...")
        metrics, feature_importance, df = train_model()
        if metrics:
            print("Model training completed successfully!")
        else:
            print("Model training failed!")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
