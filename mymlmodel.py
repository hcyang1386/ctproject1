import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import os

def mymlApp():
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), "mymodel", "classification_sample2.xlsx")
    data = pd.read_excel(data_path)

    # Data preprocessing
    data.fillna(0, inplace=True)

    # Features and labels
    X = data[["핵의 유무", "광합성 여부", "운동성 여부", "구성 세포수", "줄기 잎 뿌리 여부"]]
    y = data["5계"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # (선택) 평가
    # accuracy = model.score(X_test, y_test)
    # print(f"정확도: {accuracy:.2f}")  # Streamlit 앱에서는 print 대신 st.write 사용

    # Return trained model to be used in prediction
    return model
