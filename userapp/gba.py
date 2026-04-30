import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Dummy dataset
data = pd.read_csv('Dataset/Health1.csv')

# Create DataFrame
df = pd.DataFrame(data)

# Features and target
X = df.drop('Imbalance', axis=1)
y = df['Imbalance']

# Train Gradient Boosting Classifier
model = GradientBoostingClassifier()
model.fit(X, y)

# Save the model
joblib.dump(model, 'gba.pkl')



   

   







