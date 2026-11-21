import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score, accuracy_score
import plotly.express as px
import streamlit as st
import joblib


st.title("📞:violet[Telco Customer Churn Prediction]")

st.subheader(':red[Prediction using Machine learning]',divider='red')
st.sidebar.title("About Data Processing")
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #F3E8FF;
}

[data-testid="stSidebar"] * {
    color: #7E57C2;
}
</style>
""", unsafe_allow_html=True)


st.sidebar.markdown("""
### Categorical Features (Encoded)

| Feature | Encoding |
|----------|----------|
| gender | Female=0, Male=1 |
| Dependents | No=0, Yes=1 |
| PhoneService | No=0, Yes=1 |
| MultipleLines | No=0, Yes=1 |
| OnlineSecurity | No=0, Yes=1 |
| OnlineBackup | No=0, Yes=1 |
| DeviceProtection | No=0, Yes=1 |
| TechSupport | No=0, Yes=1 |
| StreamingTV | No=0, Yes=1 |
| StreamingMovies | No=0, Yes=1 |
| Contract | Month-to-month=0, One year=1, Two year=2 |
""")

st.sidebar.markdown("""
### Continuous Features (Scaled)

The following features were standardized before model training:

- tenure
- MonthlyCharges
- TotalCharges

Scaling helps bring numerical values to a similar range and improves model performance.
""")

df=pd.read_csv('cleanedtele.csv')
col1,col2=st.columns(2)
with col1:
    st.write(':blue[No. of Rows :]', df.shape[0])
with col2:
    st.write(':blue[No. of Columns :]',df.shape[1])
with st.expander('📂 View Dataset Preview'):
    data=st.slider(":blue[Select Number Of Rows :]",5,len(df))
    st.dataframe(df.head(data))
with st.expander("📊 Dataset visualization") :
    num_cols1 = df.select_dtypes(include = "number").columns.tolist()
    num_cols2 = df.select_dtypes(include = "number").columns.tolist()
    cat_cols = df.select_dtypes(include = "object").columns.tolist()
    st.subheader(":blue[Choose your chart and columns]")
    chart_type = st.selectbox(":rainbow[Select Chart Type]",["Bar Chart","Scatter Plot","Box Plot","Count Plot"])
    if chart_type=='Scatter Plot':
        st.subheader(':orange[Scatter Plot]')
        x_col = st.selectbox(":rainbow[Select numerical column (x-axis)]",num_cols1)
        y_col = st.selectbox(":rainbow[Select numerical column (Y-axis)]",num_cols2)
        hue=st.selectbox(":rainbow[Select Categorical column (hue)]",cat_cols)
    
        if x_col==y_col:
            st.warning('⚠️Select different numerical column on Y-axis')
        elif st.button(':rainbow[Create chart]'):
            fig=px.scatter(df,x_col,y_col,
                       title=f'{x_col} vs {x_col} vs {hue} Distribution',
                       color=hue)
            st.plotly_chart(fig)

    if chart_type=='Bar Chart':
        st.subheader(':violet[Bar Plot]')
        x_col = st.selectbox(":rainbow[Select Categorical column (x-axis)]",cat_cols)
        y_col = st.selectbox(":rainbow[Select Numerical column (Y-axis)]",num_cols1)
        if st.button(':rainbow[Create chart]'):
            fig=px.bar(df
                    ,x_col,y_col,
                    title=f'{x_col} vs {y_col} Distribution',
                    width=100,height=600)
            st.plotly_chart(fig)   

    if chart_type=='Count Plot':
        st.subheader(":green[Count Chart]")
        val = st.selectbox(":rainbow[Select Categorical column (x-axis)]",cat_cols)
        if st.button(':rainbow[Create chart]'):
            fig=px.histogram(df,x=val,color=val,
                         title=f'{val} Distribution')
            st.plotly_chart(fig)
        
            st.pyplot(plt)
    if chart_type=='Box Plot':
        st.subheader(':gray[Box Chart]')
    
        y_col=st.selectbox(":rainbow[Select Numerical column (Y-axis)]",num_cols1)  
        if st.button(':rainbow[Create chart]'):
            fig=px.box(df,y=y_col,width=50,height=600,
                     title=f'{y_col} Distribution')
            st.plotly_chart(fig)    


model=joblib.load('lr_model.pkl')
scaler=joblib.load('sc.pkl')
encoder=joblib.load('encoders.pkl')


st.header(':red[Enter Values Properly!]',divider='red')

import streamlit as st
import pandas as pd
import joblib

# ======================
# LOAD SAVED FILES
# ======================

model = joblib.load("lr_model.pkl")
scaler = joblib.load("sc.pkl")
encoders = joblib.load("encoders.pkl")



# ======================
# USER INPUTS
# ======================
col1,col2=st.columns(2)
with col1:
    gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
    )
with col2:
    dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
    )


col1,col2=st.columns(2)

with col1:
    phone_service = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
    )

with col2:
    multiple_lines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No"]
    )

col1,col2,col3=st.columns(3)    

with col1:
    online_security = st.selectbox(
    "Online Security",
    ["Yes", "No"]
)

with col2:
    online_backup = st.selectbox(
    "Online Backup",
    ["Yes", "No"]
)
    
with col3:
    device_protection = st.selectbox(
    "Device Protection",
    ["Yes", "No"]
)
    
col1,col2=st.columns(2)    

with col1:
    tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No"]
)
with col2:
    contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

col1,col2=st.columns(2)
with col1:
    streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No"]
)
with col2:
    streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No"]
)


col1,col2,col3=st.columns(3)
with col1:
    tenure = st.number_input(
    "Tenure",
    min_value=0
    )

with col2:
    monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0
    )

with col3:
    total_charges = st.number_input(
    "Total Charges",
    min_value=0.0
    )


# ======================
# PREDICT BUTTON
# ======================

if st.button("Predict Churn",type='primary'):

    gender = encoders["gender_le"].transform([gender])[0]
    dependents = encoders["dep_le"].transform([dependents])[0]

    phone_service = 1 if phone_service == "Yes" else 0
    multiple_lines = 1 if multiple_lines == "Yes" else 0
    online_security = 1 if online_security == "Yes" else 0
    online_backup = 1 if online_backup == "Yes" else 0
    device_protection = 1 if device_protection == "Yes" else 0
    tech_support = 1 if tech_support == "Yes" else 0
    streaming_tv = 1 if streaming_tv == "Yes" else 0
    streaming_movies = 1 if streaming_movies == "Yes" else 0

    contract_map = {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2
    }

    contract = contract_map[contract]

    input_df = pd.DataFrame({
        "gender":[gender],
        "Dependents":[dependents],
        "PhoneService":[phone_service],
        "MultipleLines":[multiple_lines],
        "OnlineSecurity":[online_security],
        "OnlineBackup":[online_backup],
        "DeviceProtection":[device_protection],
        "TechSupport":[tech_support],
        "StreamingTV":[streaming_tv],
        "StreamingMovies":[streaming_movies],
        "Contract":[contract],
        "tenure":[tenure],
        "MonthlyCharges":[monthly_charges],
        "TotalCharges":[total_charges]
    })




    # Scaling

    input_df[
        ["tenure", "MonthlyCharges", "TotalCharges"]
    ] = scaler.transform(
        input_df[
            ["tenure", "MonthlyCharges", "TotalCharges"]
        ]
    )
   
    input_df = input_df[model.feature_names_in_]

    # Probability

    prob = model.predict_proba(input_df)[0][1]

    # Your chosen threshold

    threshold = 0.45

    prediction = 1 if prob >= threshold else 0

    st.subheader(f"Churn Probability: {prob:.2%}")

    if prediction == 1:
        st.error("⚠️ Customer Likely To Churn")
    else:
        st.success("✅ Customer Not Likely To Churn")