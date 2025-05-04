
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import io
import base64

st.set_page_config(page_title="🧠 Stroke Risk Dashboard", layout="wide")
st.title("🧠 Advanced Stroke Risk Analysis Dashboard")

uploaded_file = st.file_uploader("📤 Upload Stroke Risk CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("🗂 Data Preview")
    st.dataframe(df.head())

    st.subheader("📈 Data Summary")
    st.write(df.describe())

    with st.expander("🧹 Filter Options"):
        work_filter = st.multiselect("Filter by Work Type", options=df["work_type"].dropna().unique(), default=df["work_type"].dropna().unique())
        age_range = st.slider("Age Range", int(df["age"].min()), int(df["age"].max()), (20, 80))

        df = df[(df["work_type"].isin(work_filter)) &
                (df["age"] >= age_range[0]) & (df["age"] <= age_range[1])]

    st.subheader("📊 Stroke Distribution by Gender")
    fig1 = px.histogram(df, x="gender", color="stroke", barmode="group")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📊 BMI vs. Stroke")
    fig2 = px.box(df, x="stroke", y="bmi", color="stroke")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📌 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    fig3, ax = plt.subplots()
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig3)

    st.subheader("📉 Glucose vs. BMI")
    fig4 = px.scatter(df, x="avg_glucose_level", y="bmi", color="stroke", hover_data=['age'])
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("📊 Count of Smoking Status")
    fig5 = px.histogram(df, x="smoking_status", color="smoking_status")
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("🏡 Count by Residence Type")
    fig6 = px.histogram(df, x="residence_type", color="residence_type")
    st.plotly_chart(fig6, use_container_width=True)

    st.subheader("💼 Count by Work Type")
    fig7 = px.histogram(df, x="work_type", color="stroke", barmode="group")
    st.plotly_chart(fig7, use_container_width=True)

    st.subheader("🧠 Predict Stroke Risk")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", df["gender"].dropna().unique())
        age = st.slider("Age", 0, 100, 50)
        hypertension = st.selectbox("Hypertension", [0, 1])
        heart_disease = st.selectbox("Heart Disease", [0, 1])
        ever_married = st.selectbox("Ever Married", df["ever_married"].dropna().unique())
    with col2:
        work_type = st.selectbox("Work Type", df["work_type"].dropna().unique())
        residence_type = st.selectbox("Residence Type", df["residence_type"].dropna().unique())
        avg_glucose_level = st.number_input("Average Glucose Level", value=100.0)
        bmi = st.number_input("BMI", value=25.0)
        smoking_status = st.selectbox("Smoking Status", df["smoking_status"].dropna().unique())

    if st.button("🧪 Predict"):
        try:
            model = joblib.load("stroke_model.pkl")
            input_df = pd.DataFrame([{
                "gender": gender,
                "age": age,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "ever_married": ever_married,
                "work_type": work_type,
                "residence_type": residence_type,
                "avg_glucose_level": avg_glucose_level,
                "bmi": bmi,
                "smoking_status": smoking_status
            }])
            prediction = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]
            st.success(f"Predicted Stroke Risk: {'Yes' if prediction == 1 else 'No'} with probability {prob:.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Export filtered data
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    st.markdown(f'<a href="data:file/csv;base64,{b64}" download="filtered_data.csv">📥 Download Filtered Data as CSV</a>', unsafe_allow_html=True)
else:
    st.info("👆 Please upload a CSV file to get started.")
