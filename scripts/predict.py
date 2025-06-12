import pandas as pd
import joblib
import os
from datetime import timedelta


# Configuration
MODEL_PATH = "models/weather_predictor.pkl"
INPUT_PATH = "data/latest_drone_log.csv"
OUTPUT_PATH = "data/predicted_weather.csv"


def load_model(model_path):
    """Load the trained weather prediction model."""
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from: {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


def load_input_data(input_path):
    """Load and validate input drone log data."""
    try:
        df = pd.read_csv(input_path, parse_dates=["time"])
        print(f"Input data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return None
    except Exception as e:
        print(f"Error loading input data: {str(e)}")
        return None


def prepare_feature_data(df):
    """Prepare feature data for model prediction."""
    # Standardize column names
    column_mapping = {
        "wind_speed_x": "wind_speed",
        "wind_direction_x": "wind_direction",
        "temperature_x": "temperature",
        "humidity_x": "humidity"
    }
    df = df.rename(columns=column_mapping)
    
    # Define required features
    required_features = [
        "latitude", "longitude", "altitude",
        "wind_speed", "wind_direction",
        "temperature", "humidity",
        "cloudcover", "precipitation"
    ]
    
    # Validate required features exist
    missing_features = [col for col in required_features if col not in df.columns]
    if missing_features:
        print(f"Error: Missing required features: {missing_features}")
        return None
    
    feature_data = df[required_features]
    print(f"Feature data prepared. Features: {len(required_features)}")
    return feature_data


def generate_predictions(model, feature_data):
    """Generate weather predictions using the trained model."""
    try:
        predictions = model.predict(feature_data)
        print(f"Predictions generated for {len(predictions)} samples")
        return predictions
    except Exception as e:
        print(f"Error generating predictions: {str(e)}")
        return None


def save_prediction_results(original_df, predictions, output_path):
    """Save prediction results to CSV file."""
    try:
        # Calculate future time (12 hours ahead)
        future_time = original_df["time"] + timedelta(hours=12)
        
        # Create prediction dataframe
        prediction_df = pd.DataFrame(predictions, columns=[
            "predicted_temperature", 
            "predicted_humidity", 
            "predicted_wind_speed"
        ])
        prediction_df["time"] = future_time
        
        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        prediction_df.to_csv(output_path, index=False)
        
        print(f"Predictions saved to: {output_path}")
        print(f"Total predictions: {len(prediction_df)}")
        
        # Display sample predictions
        print("\nSample predictions:")
        print("-" * 80)
        print(prediction_df.head().to_string(index=False))
        
        return prediction_df
        
    except Exception as e:
        print(f"Error saving predictions: {str(e)}")
        return None


def validate_inputs():
    """Validate that required input files exist."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return False
    
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input file not found at {INPUT_PATH}")
        return False
    
    return True


def main():
    """Main execution function for weather prediction inference."""
    print("Weather Prediction Inference System")
    print("=" * 50)
    
    # Validate inputs
    if not validate_inputs():
        return
    
    # Load input data
    input_df = load_input_data(INPUT_PATH)
    if input_df is None:
        return
    
    # Prepare features
    feature_data = prepare_feature_data(input_df)
    if feature_data is None:
        return
    
    # Load model
    model = load_model(MODEL_PATH)
    if model is None:
        return
    
    # Generate predictions
    predictions = generate_predictions(model, feature_data)
    if predictions is None:
        return
    
    # Save results
    result_df = save_prediction_results(input_df, predictions, OUTPUT_PATH)
    if result_df is not None:
        print("\nPrediction process completed successfully")
    else:
        print("\nPrediction process failed")


if __name__ == "__main__":
    main()