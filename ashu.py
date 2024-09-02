import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Title and Description
st.title("Water Quality Prediction")
st.write("This app predicts water quality using various machine learning models.")

# File Upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    # Load the dataset
    Water_data = pd.read_csv(uploaded_file, encoding='latin-1')
    st.write("### Data Overview")
    st.write(Water_data.head())
    
    # Data Preprocessing
    st.write("## Data Preprocessing")
    
    # Define target and features
    X = Water_data.drop('TargetColumn', axis=1)  # Replace 'TargetColumn' with actual target column name
    y = Water_data['TargetColumn']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    st.write("Data has been preprocessed.")
    
    # Model Training
    st.write("## Model Training")
    
    model_options = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(),
        "AdaBoost": AdaBoostClassifier(),
        "XGBoost": XGBClassifier()
    }
    
    model_choice = st.selectbox("Choose a model", list(model_options.keys()))
    
    if st.button('Train Model'):
        model = model_options[model_choice]
        model.fit(X_train, y_train)
        st.write(f"{model_choice} model trained.")
        
        # Model Evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        st.write(f"### {model_choice} Model Performance")
        st.write(f"Accuracy: {accuracy}")
        st.write("Confusion Matrix:")
        st.write(conf_matrix)
        st.write("Classification Report:")
        st.text(class_report)
        
    # Visualization
    st.write("## Data Visualization")
    st.write("### Correlation Heatmap")
    if st.button('Show Heatmap'):
        plt.figure(figsize=(10,6))
        sns.heatmap(Water_data.corr(), annot=True, cmap='coolwarm')
        st.pyplot(plt)

    st.write("### Pairplot")
    if st.button('Show Pairplot'):
        sns.pairplot(Water_data)
        st.pyplot(plt)
