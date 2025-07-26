import streamlit as st
import pandas as pd
import plotly.express as px
import time
import joblib

st.set_page_config(page_title="Diabetes Prediction App", layout="wide", page_icon="🩺")
st.markdown("""
    <h1 style='text-align: center; color: #e63946; animation: fadeIn 2s;'>🩺 Diabetes Prediction App</h1>
    <style>
    @keyframes fadeIn {
        0% {opacity: 0; transform: translateY(-20px);}
        100% {opacity: 1; transform: translateY(0);}    }
    .center-text {
        text-align: center;
        color: #e63946;
        font-size: 28px;
        font-weight: bold;
    }
    .container-box {
        background-color: #1e1e1e;
        padding: 20px;
        margin: 10px 0;
        border-radius: 10px;
        border-left: 5px solid #e63946;
    }
    header {visibility: hidden;}
    .block-container {padding-top: 2rem;}
    </style>
""", unsafe_allow_html=True)

# Load model once
def load_model():
    return joblib.load("xgb_model.pkl")

model = load_model()

CLASS_LABELS = {
    0: "No Diabetes",
    1: "Has Diabetes",
    2: "Borderline",
    3: "Gestational"
}

# Sidebar visualizations
with st.sidebar:
    st.subheader("📊 Overall Statistics")
    diag_counts = pd.DataFrame({
        "Type": ["Borderline", "Has Diabetes", "No Diabetes", "Gestational"],
        "Percent": [84.4, 13.0, 2.0, 0.6]
    })
    pie = px.pie(diag_counts, names="Type", values="Percent", title="Class Distribution", color_discrete_sequence=px.colors.sequential.Reds)
    st.plotly_chart(pie, use_container_width=True)

    bar = px.bar(diag_counts, x="Type", y="Percent", title="Percentage per Class", color="Type", color_discrete_sequence=px.colors.sequential.Reds)
    st.plotly_chart(bar, use_container_width=True)

# Input form
with st.form("user_input_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 1, 99, 25)
        sex = st.selectbox("Sex", ["Male", "Female"])
        education = st.selectbox("Education Level", ["Never attended school", "Elementary", "High school graduate", "Some college", "College graduate or above"])
        bp_high = st.selectbox("Have high blood pressure?", ["Yes", "No"])
        num_children = st.slider("Number of Children", 0, 10, 0)
        cholesterol = st.selectbox("High cholesterol?", ["Yes", "No"])
        hypertension = st.selectbox("Told you have hypertension?", ["Yes", "No"])
    with col2:
        alcohol = st.selectbox("Do you drink alcohol?", ["Yes", "No"])
        arthritis = st.selectbox("Any recent doctor-diagnosed arthritis?", ["Yes", "No"])
        exercise = st.selectbox("Do you exercise?", ["Yes", "No"])
        bmi = st.slider("BMI (x100)", 1200, 5000, 2500)
        prediab = st.selectbox("Told you have pre-diabetes?", ["Yes", "No"])
        health_status = st.selectbox("Would you say your health is...", ["Excellent", "Very Good", "Good", "Fair", "Poor"])
        smoker = st.selectbox("Have you smoked 100 cigarettes in your life?", ["Yes", "No"])

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = {
        "DIABAGE2": age,
        "SEX": 1 if sex == "Male" else 2,
        "DRNKANY5": 1 if alcohol == "Yes" else 0,
        "_EDUCAG": ["Never attended school", "Elementary", "High school graduate", "Some college", "College graduate or above"].index(education)+1,
        "BPHIGH4": 1 if bp_high == "Yes" else 0,
        "_CHLDCNT": num_children,
        "_DRDXAR1": 1 if arthritis == "Yes" else 0,
        "CHILDREN": num_children,
        "_RFCHOL": 1 if cholesterol == "Yes" else 0,
        "EXERANY2": 1 if exercise == "Yes" else 0,
        "_RFBMI5": bmi,
        "PREDIAB1": 1 if prediab == "Yes" else 0,
        "_RFHYPE5": 1 if hypertension == "Yes" else 0,
        "_RFHLTH": ["Excellent", "Very Good", "Good", "Fair", "Poor"].index(health_status)+1,
        "SMOKE100": 1 if smoker == "Yes" else 0
    }

    # Prepare DataFrame for prediction
    df_input = pd.DataFrame([input_data])
    pred_class = model.predict(df_input)[0]

    st.markdown(f"<h2 class='center-text'>🌟 Prediction: {CLASS_LABELS[pred_class]}</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div class='container-box'>
        <h3 style='color:#e63946;'>🧠 Tips Based on Your Inputs:</h3>
        <ul>
            <li>If you are overweight, consider a diet plan and daily activity to lower diabetes risk.</li>
            <li>Regular exercise helps regulate blood sugar and improves heart health.</li>
            <li>Quitting smoking reduces insulin resistance and inflammation.</li>
            <li>Controlling blood pressure is key to preventing diabetic complications.</li>
            <li>Monitor your cholesterol and avoid trans fats and sugary snacks.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    time.sleep(0.5)
    df_user_vs_avg = pd.DataFrame({
        'Metric': ['Age', 'BMI', 'Children'],
        'You': [age, bmi, num_children],
        'Average': [42, 2700, 2]
    })
    comp_chart = px.bar(df_user_vs_avg, x='Metric', y=['You', 'Average'], barmode='group',
                        title="Your Metrics vs. Average Population",
                        color_discrete_sequence=['#e63946', '#999999'])
    st.plotly_chart(comp_chart, use_container_width=True)

    time.sleep(0.5)
    df_risk_factors = pd.DataFrame({
        'Factor': ['High BMI', 'No Exercise', 'Smoking', 'Blood Pressure', 'Pre-diabetes'],
        'Impact (%)': [80, 60, 55, 65, 70]
    })
    st.markdown("""
    <div class='container-box'>
        <h3 style='color:#e63946;'>📈 Common Risk Factor Impact</h3>
        <p>This chart shows how much each lifestyle factor typically contributes to diabetes risk.</p>
    </div>
    """, unsafe_allow_html=True)
    chart2 = px.bar(df_risk_factors, x='Factor', y='Impact (%)', color='Factor',
                    color_discrete_sequence=px.colors.sequential.Reds, title="Risk Factors")
    st.plotly_chart(chart2, use_container_width=True)
    time.sleep(0.5)

# Stories unchanged
st.markdown("""
<div class='section-title'>
    <h2 style='text-align:center; color:#e63946; font-size:32px; font-weight:700; margin-bottom: 40px;'>
        💖 Real Stories of Diabetes Recovery
    </h2>
</div>
""", unsafe_allow_html=True)

time.sleep(0.5)
st.markdown("""
<div class='container-box'>
    <p><img src='https://www.allprodad.com/wp-content/uploads/2021/03/05-12-21-happy-people.jpg' style='width:100px; float:left; margin-right:20px; border-radius:10px;'>
    <strong>Mohamed Samir</strong><br>
    Mohamed discovered he had diabetes at age 42 after noticing unusual fatigue and frequent thirst. Rather than giving in to fear, he took action. He adjusted his diet, reduced sugar and saturated fats, and began walking daily for 45 minutes. Within a year, Mohamed managed to bring his sugar levels back to normal with minimal medication. Today, he inspires others to stay positive, be proactive, and live healthily.
    </p>
</div>
""", unsafe_allow_html=True)

time.sleep(0.5)
st.markdown("""
<div class='container-box'>
    <p><img src='https://cdn.psychologytoday.com/sites/default/files/styles/article-inline-half/public/field_blog_entry_images/2017-09/shutterstock_243101992.jpg?itok=sxfMiTsD' style='width:100px; float:left; margin-right:20px; border-radius:10px;'>
    <strong>Sarah Mostafa</strong><br>
    Sarah, a 36-year-old teacher, was diagnosed with gestational diabetes during her second pregnancy. Instead of panicking, she embraced the opportunity to learn. With support from her healthcare team, she followed a strict meal plan, practiced yoga, and monitored her glucose levels daily. After her delivery, she maintained her healthy habits and avoided developing type 2 diabetes. Now she mentors expecting mothers through online support groups.
    </p>
</div>
""", unsafe_allow_html=True)

time.sleep(0.5)
st.markdown("""
<div class='container-box'>
    <p><img src='https://as2.ftcdn.net/v2/jpg/03/96/16/79/1000_F_396167959_aAhZiGlJoeXOBHivMvaO0Aloxvhg3eVT.jpg' style='width:100px; float:left; margin-right:20px; border-radius:10px;'>
    <strong>Ahmed Rami</strong><br>
    At just 29, Ahmed was shocked to be told he was prediabetic. A busy software engineer, he had fallen into a sedentary lifestyle and poor eating habits. Determined to take control, he began meal prepping, cut down on sugar drinks, and set alarms to walk hourly. His HbA1c dropped to normal in 6 months, and now he shares his journey through a popular Instagram page to inspire others.
    </p>
</div>
""", unsafe_allow_html=True)
