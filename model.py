import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib  # To save the model

# Load dataset
data = pd.read_csv("C:/Users/kesha/Downloads/r/patient_data.csv", encoding="latin1")

# Prepare features and labels
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'disease_model.pkl')

# Test the model and print accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
