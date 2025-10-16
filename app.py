import streamlit as st

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Flood Forecasting and Analysis System",
    page_icon="🌊",
    layout="wide"
)

# -----------------------------
# MAIN PAGE CONTENT (HOME)
# -----------------------------
st.title("🌊 Flood Forecasting & Analysis System")
st.write("Welcome to the multi-page Streamlit application!")

st.markdown("""
This system allows you to:

✅ Upload and clean flood-related datasets  
✅ Visualize water levels, flood occurrences, and damages  
✅ Analyze trends and patterns by year or area  
✅ Build forecasting models (SARIMA, etc.)  
✅ View summary statistics and insights  

---

### 📌 Navigation Guide
Use the **left sidebar** to access all pages:

**1️⃣ Data Cleaning** – Upload & prepare your dataset  
**2️⃣ Visualization** – Charts & graphs  
**3️⃣ Analysis** – Flood counts, averages, top areas, damages  
**4️⃣ Forecasting** – Time-series prediction (SARIMA)  
**5️⃣ Summary** – Final report and key insights

---

✅ Each page will guide you step by step.  
✅ Start with **Data Cleaning**.  
""")

st.info("👉 Go to the sidebar and click **1_Data_Cleaning** to begin.")
st.write("If you don't see the sidebar, click the **>** icon in the top left.")
