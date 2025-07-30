import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

def mymlApp():
    # Load data
    data = pd.read_excel("./mymodel/classification_sample2.xlsx")

    # Data preprocessing
    # Fill NaN values with a placeholder (e.g., 0 for numerical columns)
    data.fillna(0, inplace=True)

    # Features and labels
    X = data[["핵의 유무", "세포벽의 유무", "광합성 여부", "운동성 여부", "구성 세포수", "줄기 잎 뿌리 여부"]]
    y = data["5계"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")

    # Save the trained model to a file
    with open("classification.pkl", "wb") as file:
        pickle.dump(model, file)

    print("Model has been saved as 'classification.pkl'.")

    # Function to predict the classification
    def predict_classification(features):
        with open("classification.pkl", "rb") as file:
            loaded_model = pickle.load(file)
        prediction = loaded_model.predict([features])
        return prediction[0]

    # Example usage
    # Replace with user input for real-world use
    example_features = [1, 1, 0, 1, 2, 1]  # Example feature set
    predicted_class = predict_classification(example_features)
    print(f"The predicted class for the features {example_features} is: {predicted_class}")