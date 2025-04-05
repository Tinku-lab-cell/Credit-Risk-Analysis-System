import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Page Configuration
st.set_page_config(
    page_title="Credit Risk Analysis System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .st-emotion-cache-16idsys p {
        font-size: 1.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Utility Functions
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df['Credit_Score'] = pd.to_numeric(df['Credit_Score'])
    df['Loan_Amount'] = pd.to_numeric(df['Loan_Amount'])
    df['Income'] = pd.to_numeric(df['Income'])
    df['Debt_to_Income_Ratio'] = pd.to_numeric(df['Debt_to_Income_Ratio'])
    return df

def create_score_distribution(df):
    fig = px.histogram(df, x='Credit_Score', color='Risk_Category', 
                       title='Credit Score Distribution', template='plotly_dark')
    return fig

def create_risk_breakdown(df):
    risk_counts = df['Risk_Category'].value_counts()
    fig = go.Figure(data=[go.Pie(labels=risk_counts.index, values=risk_counts.values, hole=.3)])
    fig.update_layout(title='Risk Category Distribution')
    return fig

def prepare_features(df):
    X = df[['Credit_Score', 'Loan_Amount', 'Income', 'Debt_to_Income_Ratio']]
    y = df['Good_Loan_Candidate']
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    return model, scaler, accuracy

def predict_risk(model, scaler, features):
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)
    return prediction[0], probability[0]

# Main UI using Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Home", "Data Analysis", "Risk Prediction", "About"])

with tab1:
    st.title("Credit Risk Analysis System 💳")
    uploaded_file = st.file_uploader("Upload your credit data (CSV)", type=['csv'])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        st.session_state['data'] = df
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(df))
        col2.metric("High Risk Cases", len(df[df['Risk_Category'] == 'High']))
        col3.metric("Good Candidates", len(df[df['Good_Loan_Candidate'] == True]))
        st.subheader("Sample Data")
        st.dataframe(df.head())
    else:
        st.info("Please upload a CSV file to begin analysis")

with tab2:
    st.title("Credit Data Analysis 📊")
    if 'data' in st.session_state:
        df = st.session_state['data']
        col1, col2 = st.columns(2)
        col1.metric("Avg. Credit Score", f"{df['Credit_Score'].mean():.0f}")
        col2.metric("Avg. Loan Amount", f"${df['Loan_Amount'].mean():,.2f}")
        st.subheader("Credit Score Distribution")
        st.plotly_chart(create_score_distribution(df), use_container_width=True)
        st.subheader("Risk Category Breakdown")
        st.plotly_chart(create_risk_breakdown(df), use_container_width=True)
    else:
        st.warning("Please upload data first!")

with tab3:
    st.title("Credit Risk Prediction 🎯")
    if 'data' in st.session_state:
        df = st.session_state['data']
        X, y = prepare_features(df)
        model, scaler, accuracy = train_model(X, y)
        st.success(f"Model Accuracy: {accuracy:.2%}")
        col1, col2 = st.columns(2)
        credit_score = col1.number_input("Credit Score", min_value=300, max_value=850, value=650)
        loan_amount = col1.number_input("Loan Amount ($)", min_value=1000, max_value=1000000, value=50000)
        income = col2.number_input("Annual Income ($)", min_value=10000, max_value=1000000, value=60000)
        dti = col2.number_input("Debt to Income Ratio (%)", min_value=0.0, max_value=100.0, value=30.0)
        
        if st.button("Predict Risk"):
            features = [credit_score, loan_amount, income, dti]
            prediction, probability = predict_risk(model, scaler, features)
            confidence = max(probability)
            fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=confidence*100,
                title={"text": "Prediction Confidence"}, domain={'x': [0, 1], 'y': [0, 1]}))
            st.subheader("Prediction Result")
            if prediction:
                st.success("Good Loan Candidate ✅")
            else:
                st.error("High Risk Candidate ⚠️")
            st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        st.warning("Please upload data first!")

with tab4:
    st.title("About ℹ️")
    st.markdown("""This Credit Risk Analysis System helps financial institutions assess credit risk using historical data
    and machine learning techniques.
                
🌟 Key Features  
- 📊 **Data Visualization & Analysis** – Gain insights through interactive charts.  
- 🤖 **AI-Powered Risk Prediction** – Make data-driven decisions.  
- 🎯 **User-Friendly Dashboard** – Navigate with ease.  
- 📂 **Upload & Process Data** – Supports CSV format.  

### 🚀 How It Works  
1. **Upload** your credit data CSV file on the home page.  
2. **Explore** key statistics and interactive visualizations.  
3. **Predict** risk for new applicants with AI-powered analytics.  
4. **Download** results for further analysis.  

### 🛠️ Technologies Used  
- **Python & Streamlit** – For an interactive web experience.  
- **Plotly** – For stunning data visualizations.  
- **Scikit-Learn** – For machine learning predictions.  
- **Pandas & NumPy** – For efficient data processing.  

### 📞 Get in Touch  
💌 For support or inquiries, contact  **Angeline Yuvancy S, MSc Data Science** at **angelineyuvancy02@gmail.com**.""")
