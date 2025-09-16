# Name: NGUYEN XUAN HUY LINCOLN 阮春輝
# Student ID: 10914193
# Machine Learning Assignment: White Wine Classification Web App

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, f1_score
from imblearn.over_sampling import SMOTE

# Load the dataset
def load_data():
    file_path = './winequality-white.csv'  # Update this path as needed
    df = pd.read_csv(file_path, delimiter=';')
    return df

def preprocess_data(df):
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    # Check class distribution
    class_counts = y.value_counts()
    min_samples = class_counts.min()

    # Set k_neighbors based on minimum class size
    smote = SMOTE(random_state=42, k_neighbors=min(min_samples - 1, 5))
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    return X_resampled, y_resampled

def main():
    st.set_page_config(page_title="White Wine Classification App", page_icon="🍷", layout="wide")
    st.markdown(
        """
        <style>
        .main {background-color: white; color: black;}
        .block-container {padding: 2rem;}
        h1, h2, h3, h4 {color: #000000; text-align: center;}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🍷 White Wine Classification App")
    st.markdown(
        """
        ### Developed by a Data Scientist
        This app utilizes advanced Machine Learning models to classify the quality of white wine. You can:
        - Explore the dataset
        - Visualize the data
        - Train and evaluate machine learning models
        - Make real-time predictions
        """
    )

    # Load and display dataset
    df = load_data()
    st.sidebar.header("Data Options")
    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader("Raw Dataset")
        st.dataframe(df)

    # Data visualization
    st.subheader("Data Visualization")
    if st.checkbox("Show Feature Distributions"):
        for col in df.columns[:-1]:
            fig = px.histogram(df, x=col, title=f"Distribution of {col}", marginal="box")
            st.plotly_chart(fig)

    # Data preprocessing
    st.subheader("Data Preprocessing")
    st.write("Encoding target variable and handling class imbalance using SMOTE...")
    X, y = preprocess_data(df)
    st.write(pd.DataFrame(X).describe())

    # Train-test split
    test_size = st.sidebar.slider("Test Data Size", 0.1, 0.5, 0.3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    st.write(f"Training Set Size: {X_train.shape}")
    st.write(f"Test Set Size: {X_test.shape}")

    # Model selection and hyperparameters
    st.sidebar.header("Model Selection and Hyperparameters")
    model_name = st.sidebar.selectbox("Choose a Model", ["Random Forest", "Gradient Boosting", "Decision Tree"])

    if model_name == "Random Forest":
        n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    elif model_name == "Gradient Boosting":
        learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1)
        model = GradientBoostingClassifier(learning_rate=learning_rate, random_state=42)
    else:
        max_depth = st.sidebar.slider("Max Depth", 1, 50, 10)
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

    # Model training
    st.subheader(f"Training {model_name}")
    model.fit(X_train, y_train)

    # Model evaluation
    st.subheader("Model Evaluation")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    st.write(f"**Accuracy:** {acc:.2f}")
    st.write(f"**Precision:** {precision:.2f}")
    st.write(f"**F1 Score:** {f1:.2f}")

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # Real-time prediction
    st.subheader("Real-Time Prediction")
    input_features = []
    for col in X.columns:
        value = st.number_input(f"{col}", min_value=float(df[col].min()), max_value=float(df[col].max()), value=float(df[col].mean()))
        input_features.append(value)

    if st.button("Predict Quality"):
        prediction = model.predict([input_features])
        st.write(f"**Predicted Quality:** {int(prediction[0])}")

    # Animated banner or footer
    st.markdown(
        """
        <style>
        .footer {background-color: white; color: black; text-align: center; padding: 10px;}
        </style>
        <div class="footer">✨ Made with ❤️ and Streamlit ✨</div>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
