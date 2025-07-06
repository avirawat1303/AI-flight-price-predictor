from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# Global variables for model and encoders
model = None
label_encoders = {}
feature_columns = []

def create_sample_data():
    """Create sample flight data for training"""
    np.random.seed(42)
    
    airlines = ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet', 'Multiple carriers', 'GoAir', 'Vistara', 'Air Asia']
    sources = ['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai']
    destinations = ['New Delhi', 'Banglore', 'Cochin', 'Kolkata', 'Delhi', 'Hyderabad']
    
    data = []
    for _ in range(1000):
        airline = np.random.choice(airlines)
        source = np.random.choice(sources)
        destination = np.random.choice([d for d in destinations if d != source])
        
        # Generate realistic flight data
        stops = np.random.choice([0, 1, 2, 3], p=[0.4, 0.35, 0.2, 0.05])
        duration_hours = np.random.randint(1, 25)
        duration_mins = np.random.randint(0, 60)
        dep_time_hour = np.random.randint(0, 24)
        arrival_time_hour = (dep_time_hour + duration_hours) % 24
        
        # Create price based on realistic factors
        base_price = 3000
        
        # Airline factor
        airline_multiplier = {'IndiGo': 1.0, 'Air India': 1.1, 'Jet Airways': 1.3, 
                             'SpiceJet': 0.9, 'Multiple carriers': 1.2, 'GoAir': 0.85,
                             'Vistara': 1.4, 'Air Asia': 0.8}
        
        # Route factor
        route_multiplier = 1.0
        if source in ['Mumbai', 'Delhi'] or destination in ['Mumbai', 'Delhi']:
            route_multiplier = 1.2
        
        # Stops factor
        stops_multiplier = 1.0 + (stops * 0.1)
        
        # Duration factor
        duration_multiplier = 1.0 + (duration_hours * 0.02)
        
        # Time factor
        if 6 <= dep_time_hour <= 10 or 18 <= dep_time_hour <= 22:
            time_multiplier = 1.15
        else:
            time_multiplier = 1.0
        
        price = base_price * airline_multiplier[airline] * route_multiplier * stops_multiplier * duration_multiplier * time_multiplier
        price = int(price + np.random.normal(0, 500))  # Add some noise
        price = max(1000, price)  # Minimum price
        
        data.append({
            'Airline': airline,
            'Source': source,
            'Destination': destination,
            'Dep_Time': f"{dep_time_hour:02d}:{np.random.randint(0, 60):02d}",
            'Arrival_Time': f"{arrival_time_hour:02d}:{np.random.randint(0, 60):02d}",
            'Duration': f"{duration_hours}h {duration_mins}m",
            'Total_Stops': stops,
            'Price': price,
            'Dep_Hour': dep_time_hour
        })
    
    return pd.DataFrame(data)

def preprocess_data(df):
    """Preprocess the flight data"""
    # Extract hour from time
    df['Dep_Hour'] = df['Dep_Time'].str.split(':').str[0].astype(int)
    df['Arrival_Hour'] = df['Arrival_Time'].str.split(':').str[0].astype(int)
    
    # Extract duration in minutes
    df['Duration_mins'] = df['Duration'].str.extract('(\d+)h').astype(int) * 60 + \
                         df['Duration'].str.extract('(\d+)m').astype(int)
    
    # Features for encoding
    categorical_features = ['Airline', 'Source', 'Destination']
    
    # Initialize label encoders
    global label_encoders
    for feature in categorical_features:
        le = LabelEncoder()
        df[f'{feature}_encoded'] = le.fit_transform(df[feature])
        label_encoders[feature] = le
    
    # Select features for model
    global feature_columns
    feature_columns = ['Airline_encoded', 'Source_encoded', 'Destination_encoded', 
                      'Dep_Hour', 'Arrival_Hour', 'Duration_mins', 'Total_Stops']
    
    return df

def train_model():
    """Train the flight price prediction model"""
    global model
    
    print("Creating sample data...")
    df = create_sample_data()
    
    print("Preprocessing data...")
    df = preprocess_data(df)
    
    # Prepare features and target
    X = df[feature_columns]
    y = df['Price']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    print("Training model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model trained successfully!")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R² Score: {r2:.3f}")
    
    # Save model and encoders
    joblib.dump(model, 'flight_price_model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    
    return model, mae, r2

def load_model():
    """Load the trained model"""
    global model, label_encoders
    try:
        model = joblib.load('flight_price_model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        print("Model loaded successfully!")
        return True
    except:
        print("Model not found. Training new model...")
        train_model()
        return True

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict flight price"""
    try:
        data = request.get_json()
        
        # Extract features
        airline = data['airline']
        source = data['source']
        destination = data['destination']
        dep_time = data['dep_time']
        arrival_time = data['arrival_time']
        duration = data['duration']
        total_stops = int(data['total_stops'])
        
        # Preprocess input
        dep_hour = int(dep_time.split(':')[0])
        arrival_hour = int(arrival_time.split(':')[0])
        
        # Extract duration in minutes
        duration_parts = duration.replace('h', '').replace('m', '').split()
        duration_mins = int(duration_parts[0]) * 60 + int(duration_parts[1])
        
        # Encode categorical features
        airline_encoded = label_encoders['Airline'].transform([airline])[0]
        source_encoded = label_encoders['Source'].transform([source])[0]
        destination_encoded = label_encoders['Destination'].transform([destination])[0]
        
        # Create feature vector
        features = np.array([[airline_encoded, source_encoded, destination_encoded, 
                            dep_hour, arrival_hour, duration_mins, total_stops]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get prediction confidence (using std of tree predictions)
        tree_predictions = [tree.predict(features)[0] for tree in model.estimators_]
        confidence = np.std(tree_predictions)
        
        return jsonify({
            'success': True,
            'predicted_price': round(prediction, 2),
            'confidence_interval': round(confidence, 2),
            'message': f'Predicted flight price: ₹{prediction:.2f}'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/visualize')
def visualize():
    """Generate comprehensive price trend visualizations"""
    try:
        # Create sample data for visualization
        df = create_sample_data()

        # Clean and prepare data for visualization
        print("Preparing data for visualization...")

        # ✅ Ensure Dep_Hour is properly extracted
        df['Dep_Hour'] = pd.to_datetime(df['Dep_Time'], errors='coerce').dt.hour

        # ✅ Ensure Total_Stops is numeric
        df['Total_Stops'] = pd.to_numeric(df['Total_Stops'], errors='coerce')

        # ✅ Drop rows with missing data
        df.dropna(subset=['Price', 'Total_Stops', 'Dep_Hour', 'Airline', 'Source', 'Destination'], inplace=True)

        # ✅ Standardize strings
        df['Airline'] = df['Airline'].astype(str)
        df['Source'] = df['Source'].astype(str)
        df['Destination'] = df['Destination'].astype(str)

        charts = {}

        # 1. Average Price by Airline
        airline_prices = df.groupby('Airline')['Price'].mean().sort_values(ascending=False)
        fig1 = px.bar(
            x=airline_prices.index,
            y=airline_prices.values,
            title='Average Flight Prices by Airline',
            labels={'x': 'Airline', 'y': 'Average Price (₹)'},
            color=airline_prices.values,
            color_continuous_scale='viridis'
        )
        fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12))
        charts['airline_prices'] = fig1.to_json()


        # 5. Price Range by Source City
        fig5 = px.violin(df, x='Source', y='Price', box=True, title='Price Distribution by Source City')
        fig5.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12))
        charts['source_prices'] = fig5.to_json()

        return jsonify({'success': True, 'charts': charts})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/model-info')
def model_info():
    """Get model information"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'})
        
        # Get feature importances
        feature_names = ['Airline', 'Source', 'Destination', 'Dep_Hour', 'Arrival_Hour', 'Duration_mins', 'Total_Stops']
        importances = model.feature_importances_
        
        feature_importance = dict(zip(feature_names, importances))
        
        return jsonify({
            'success': True,
            'feature_importance': feature_importance,
            'n_estimators': model.n_estimators,
            'model_type': 'Random Forest Regressor'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    # Load or train model
    load_model()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)