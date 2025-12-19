import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Video Game Hit Prediction", layout="wide")
st.title("🎮 Video Game Sales Analysis & Hit Prediction")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    file_path = "Video_Games_Sales_as_at_22_Dec_2016.csv"
    encodings = ["utf-8", "latin1", "ISO-8859-1", "cp1252"]
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            return df
        except:
            continue
    st.error("Failed to load dataset")
    return None

df = load_data()
if df is not None:
    st.success("Dataset loaded successfully")

    # -----------------------------
    # Data Cleaning
    # -----------------------------
    df.drop(columns=['User_Count', 'Critic_Count', 'Rating', 'Developer'], inplace=True, errors="ignore")
    df['Critic_Score'] = pd.to_numeric(df['Critic_Score'], errors='coerce')
    df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')
    df['Critic_Score'].fillna(df['Critic_Score'].median(), inplace=True)
    df['User_Score'].fillna(df['User_Score'].median(), inplace=True)
    df.dropna(subset=['Year_of_Release'], inplace=True)
    df['Year_of_Release'] = df['Year_of_Release'].astype(int)
    df['Publisher'].fillna('Unknown', inplace=True)
    df.dropna(subset=['Name','Genre'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # -----------------------------
    # Cleaned Dataset
    # -----------------------------
    st.header("🧹 Cleaned Dataset (Dataset Nettoyé)")
    st.dataframe(df.head(20))
    
    st.download_button(
        "Download Cleaned Dataset",
        df.to_csv(index=False),
        "Video_Games_Cleaned.csv",
        "text/csv"
    )

    # -----------------------------
    # Descriptive Statistics
    # -----------------------------
    st.header("📊 Descriptive Statistics (Statistique Descriptive)")
    st.dataframe(df.describe(include='all').T)

    # -----------------------------
    # Exploratory Data Analysis
    # -----------------------------
    st.header("📈 Exploratory Data Analysis")
    
    st.subheader("Top 10 Best-Selling Games")
    top_games = df[['Name', 'Global_Sales']].sort_values(by='Global_Sales', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(x='Global_Sales', y='Name', data=top_games, ax=ax, palette="viridis")
    st.pyplot(fig)

    st.subheader("Distribution of Critic Scores")
    fig, ax = plt.subplots()
    sns.histplot(df['Critic_Score'], bins=20, ax=ax, color='skyblue')
    st.pyplot(fig)

    st.subheader("Global Sales by Genre")
    sales_by_genre = df.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(x=sales_by_genre.index, y=sales_by_genre.values, ax=ax, palette="magma")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # -----------------------------
    # Hit / Flop Label
    # -----------------------------
    st.header("🤖 Machine Learning: Hit Prediction")
    hit_threshold = st.slider("Hit threshold (million sales)", 0.5, 5.0, 1.0)
    df['Hit'] = (df['Global_Sales'] > hit_threshold).astype(int)
    st.write("Hit vs Flop distribution:")
    st.write(df['Hit'].value_counts())

    # -----------------------------
    # Features & Target
    # -----------------------------
    X = df[['Platform','Genre','Year_of_Release','Critic_Score','User_Score']].copy()
    y = df['Hit']
    le = LabelEncoder()
    for col in ['Platform','Genre']:
        X[col] = le.fit_transform(X[col].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    num_cols = ['Critic_Score','User_Score','Year_of_Release']
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # -----------------------------
    # Model Training
    # -----------------------------
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    st.subheader("Model Performance")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # -----------------------------
    # Feature Importance
    # -----------------------------
    st.subheader("Feature Importance")
    feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
    fig, ax = plt.subplots()
    feat_imp.plot(kind='barh', ax=ax)
    st.pyplot(fig)

    # -----------------------------
    # Predictions
    # -----------------------------
    df['Predicted_Hit'] = rf.predict(X)
    df['Predicted_Hit_Label'] = df['Predicted_Hit'].map({1: 'Hit', 0: 'Flop'})

    st.subheader("Predicted Results")
    st.dataframe(df[['Name', 'Predicted_Hit_Label']].head(20))

    st.download_button(
        "Download Predictions CSV",
        df[['Name', 'Predicted_Hit_Label']].to_csv(index=False),
        "Video_Game_HitFlop_Predictions.csv",
        "text/csv"
    )