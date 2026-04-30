import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

def electric_car_price_prediction(file_path, user_input):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Select target variable (Price) and features
    target = 'Price'
    X = data.drop(columns=[target, 'Model'])  # Dropping 'Model' since it's very specific to each car
    y = data[target]

    # Encode categorical variables using LabelEncoder
    label_encoders = {}
    for column in X.select_dtypes(include='object').columns:
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Function to handle unseen labels during encoding
    def handle_unseen_labels(label_encoder, value):
        # Return encoded value if it exists; else return -1 or a numeric placeholder
        if value in label_encoder.classes_:
            return label_encoder.transform([value])[0]
        else:
            print(f"Warning: Unseen label '{value}' detected. Assigning it to a placeholder.")
            return -1  # Placeholder for unseen labels

    # Create a DataFrame with the same structure as the training data
    input_df = pd.DataFrame([user_input])
    
    # Apply Label Encoding for categorical features
    for column in input_df.select_dtypes(include='object').columns:
        if column in label_encoders:
            input_df[column] = input_df[column].apply(lambda x: handle_unseen_labels(label_encoders[column], x))
        else:
            raise ValueError(f"Unknown category for column '{column}' during prediction.")
    
    # Ensure the input columns match the training columns
    input_df = input_df.reindex(columns=X_train.columns, fill_value=0)  # Filling missing columns with 0

    # Make the prediction
    prediction = rf_model.predict(input_df)
    
    return prediction[0]  # Return the predicted price

# Example user input (including 'Brand')
user_input = {
    'AccelSec': 5.3,          # Example acceleration in seconds
    'TopSpeed_KmH': 220,      # Top speed in km/h
    'Range_Km': 450,          # Range in km
    'Efficiency_WhKm': 150,   # Efficiency in Wh/km
    'FastCharge_KmH': 250,    # Fast charge speed in km/h
    'RapidCharge': 'Yes',     # Rapid charge capability
    'PowerTrain': 'AWD',      # Powertrain type
    'PlugType': 'Type 2',     # Plug type
    'BodyStyle': 'SUV',       # Body style
    'Segment': 'D',           # Segment type
    'Seats': 5,               # Number of seats
    'Brand': 'Tesla'          # Include 'Brand' to match training features
}

# Call the function with the file path and user input
predicted_price = electric_car_price_prediction('Dataset/ElectricCarData_Clean3.csv', user_input)
print(f"Predicted Car Price: ${predicted_price}")
