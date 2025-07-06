# Flight Price Prediction Web App

A full-stack web application that predicts flight prices using machine learning. Built with Flask backend and HTML/CSS/JavaScript frontend.

## Features

- 🤖 **Machine Learning Model**: Random Forest Regressor trained on flight data
- 🚀 **Flask API**: RESTful API with `/predict` endpoint
- 📊 **Visualizations**: Interactive charts showing price trends
- 🎨 **Beautiful UI**: Modern, responsive design
- 📱 **Mobile Friendly**: Works on all devices

## Model Features

The ML model considers these factors for price prediction:
- Airline
- Source and Destination
- Departure and Arrival Times
- Flight Duration
- Number of Stops

## Installation & Setup

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Local Setup

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Access the app**:
   - Open your browser and go to `http://localhost:5000`
   - The model will be trained automatically on first run

## API Endpoints

### `/predict` (POST)
Predict flight price based on input parameters.

**Request Body**:
```json
{
    "airline": "IndiGo",
    "source": "Delhi",
    "destination": "Mumbai",
    "dep_time": "06:00",
    "arrival_time": "08:30",
    "duration": "2h 30m",
    "total_stops": "0"
}
```

**Response**:
```json
{
    "success": true,
    "predicted_price": 4250.75,
    "confidence_interval": 325.50,
    "message": "Predicted flight price: ₹4250.75"
}
```

### `/visualize` (GET)
Get visualization data for price trends.

### `/model-info` (GET)
Get model information including feature importance.

## File Structure

```
├── app.py              # Main Flask application
├── templates/
│   └── index.html      # Frontend HTML template
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── flight_price_model.pkl    # Trained model (generated)
└── label_encoders.pkl        # Encoders (generated)
```

## Model Details

- **Algorithm**: Random Forest Regressor
- **Features**: 7 input features including airline, route, time, and stops
- **Training Data**: 1000 synthetic flight records with realistic pricing patterns
- **Evaluation**: Mean Absolute Error and R² score
- **Persistence**: Model saved using joblib

## Usage

1. Fill in the flight details form
2. Click "Predict Price"
3. View the predicted price and confidence interval
4. Load price trends visualization to see airline comparison

## Technical Details

- **Backend**: Flask with CORS support
- **ML Library**: scikit-learn
- **Data Processing**: pandas, numpy
- **Visualization**: Plotly for interactive charts
- **Model Storage**: joblib for persistence

## Features Explained

### Machine Learning Pipeline
1. **Data Generation**: Creates synthetic flight data with realistic pricing patterns
2. **Preprocessing**: Encodes categorical variables and extracts time features
3. **Training**: Uses Random Forest with 100 estimators
4. **Evaluation**: Provides MAE and R² metrics
5. **Persistence**: Saves model and encoders for reuse

### Price Prediction Logic
The model considers multiple factors:
- **Airline Premium**: Different airlines have different base prices
- **Route Popularity**: Popular routes (Mumbai, Delhi) cost more
- **Time Slots**: Peak hours (6-10 AM, 6-10 PM) are more expensive
- **Stops**: More stops generally increase price
- **Duration**: Longer flights cost more

## Customization

### Adding New Airlines/Routes
1. Update the sample data generation in `create_sample_data()`
2. Add new options to the HTML form
3. Retrain the model

### Changing the Model
1. Replace `RandomForestRegressor` with your preferred algorithm
2. Update hyperparameters as needed
3. Modify evaluation metrics if required

## Production Considerations

- Use a proper database instead of generating synthetic data
- Add authentication and rate limiting
- Implement proper error handling and logging
- Add input validation and sanitization
- Use environment variables for configuration
- Deploy using WSGI server (Gunicorn, uWSGI)

## License

This project is for educational purposes. Feel free to modify and use as needed."# AI-flight-price-predictor" 
"# AI-flight-price-predictor" 
