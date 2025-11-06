import streamlit as st
import pandas as pd
import joblib

# Load model and TF-IDF vectorizer
model = joblib.load('severity_model.pkl')
tfidf = joblib.load('tfidf.pkl')

st.set_page_config(page_title="Drug Interaction Severity Predictor", page_icon="💊")
st.title("💊 Drug Interaction Severity Prediction App")

st.write("Enter an interaction description or select a known pair to predict its severity level.")

# Upload dataset for reference (optional)
df = pd.read_csv("ddi_with_final_risk_levels.csv")

# Search existing record
st.subheader("🔍 Search Existing Interaction")
drug1 = st.text_input("Enter Drug 1:")
drug2 = st.text_input("Enter Drug 2:")

if st.button("Search"):
    result = df[
        ((df["drug1"].str.lower() == drug1.lower()) &
         (df["drug2"].str.lower() == drug2.lower())) |
        ((df["drug1"].str.lower() == drug2.lower()) &
         (df["drug2"].str.lower() == drug1.lower()))
    ]

    if not result.empty:
        desc = result.iloc[0]["interactiondescription"]
        st.success(f"**Description Found:** {desc}")
        text_tfidf = tfidf.transform([desc])
        pred = model.predict(text_tfidf)[0]
        st.info(f"**Predicted Severity:** {pred}")
    else:
        st.warning("⚠️ No record found for this pair.")

# Custom text input for new prediction
st.subheader("🧠 Predict from Custom Description")
new_text = st.text_area("Enter interaction description here:")
if st.button("Predict Severity"):
    if new_text.strip() != "":
        new_tfidf = tfidf.transform([new_text])
        prediction = model.predict(new_tfidf)[0]
        st.success(f"Predicted Severity: **{prediction}**")
    else:
        st.warning("Please enter an interaction description.")
