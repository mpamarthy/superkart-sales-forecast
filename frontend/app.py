
import os
import pandas as pd
import requests
import streamlit as st

# Base URL of the Flask backend. Inside the Docker network the container is named "backend".
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:7860")

st.set_page_config(page_title="SuperKart Sales Forecast", layout="centered")
st.title("SuperKart Sales Forecast")
st.write("Predict the total sales revenue of a product in a store, one at a time or in bulk from a CSV file.")

tab_single, tab_batch = st.tabs(["Single prediction", "Batch prediction"])

# ------------------------------------------------------------------ single prediction
with tab_single:
    st.subheader("Product details")
    c1, c2 = st.columns(2)
    with c1:
        Product_Weight = st.number_input("Product weight", min_value=0.0, value=12.66, step=0.1)
        Product_MRP = st.number_input("Product MRP (price)", min_value=0.0, value=117.08, step=1.0)
        Product_Allocated_Area = st.number_input(
            "Allocated display area (ratio, 0 to 1)", min_value=0.0, max_value=1.0, value=0.027, step=0.001, format="%.3f")
        Product_Sugar_Content = st.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
        Product_Id_char = st.selectbox(
            "Product ID prefix", ["FD", "DR", "NC"],
            format_func=lambda x: {"FD": "FD (Food)", "DR": "DR (Drinks)", "NC": "NC (Non-consumable)"}[x])
        Product_Type_Category = st.selectbox("Product category", ["Perishables", "Non Perishables"])
    with c2:
        st.subheader("Store details")
        # The four store attributes describe the same four stores the model was trained on, so a
        # store profile fills them consistently. "Custom" lets an advanced user set them one by one.
        STORE_PROFILES = {
            "Supermarket Type2 | Medium | Tier 2 | 16 years": ("Medium", "Tier 2", "Supermarket Type2", 16),
            "Supermarket Type1 | High | Tier 2 | 38 years": ("High", "Tier 2", "Supermarket Type1", 38),
            "Departmental Store | Medium | Tier 1 | 26 years": ("Medium", "Tier 1", "Departmental Store", 26),
            "Food Mart | Small | Tier 3 | 27 years": ("Small", "Tier 3", "Food Mart", 27),
            "Custom": None,
        }
        profile = st.selectbox("Store profile", list(STORE_PROFILES.keys()))
        if STORE_PROFILES[profile] is not None:
            Store_Size, Store_Location_City_Type, Store_Type, Store_Age_Years = STORE_PROFILES[profile]
        else:
            Store_Size = st.selectbox("Store size", ["Small", "Medium", "High"], index=1)
            Store_Location_City_Type = st.selectbox("City tier", ["Tier 1", "Tier 2", "Tier 3"], index=1)
            Store_Type = st.selectbox("Store type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"], index=1)
            Store_Age_Years = st.number_input("Store age (years)", min_value=0, value=16, step=1)

    product_data = {
        "Product_Weight": Product_Weight,
        "Product_Sugar_Content": Product_Sugar_Content,
        "Product_Allocated_Area": Product_Allocated_Area,
        "Product_MRP": Product_MRP,
        "Store_Size": Store_Size,
        "Store_Location_City_Type": Store_Location_City_Type,
        "Store_Type": Store_Type,
        "Product_Id_char": Product_Id_char,
        "Store_Age_Years": Store_Age_Years,
        "Product_Type_Category": Product_Type_Category,
    }

    if st.button("Predict sales", type="primary"):
        try:
            response = requests.post(f"{BACKEND_URL}/v1/predict", json=product_data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted total sales: {result['Sales']:,.2f}")
                for w in result.get("warnings", []):
                    st.warning(w)
            else:
                st.error(f"The prediction API returned an error: {response.json().get('error', response.text)}")
        except requests.exceptions.RequestException:
            st.error("Unable to connect to the prediction API. Check that the backend is running.")

# ------------------------------------------------------------------ batch prediction
with tab_batch:
    st.subheader("Upload a CSV file")
    st.caption("Required columns: Product_Weight, Product_Sugar_Content, Product_Allocated_Area, Product_MRP, "
               "Store_Size, Store_Location_City_Type, Store_Type, Product_Id_char, Store_Age_Years, Product_Type_Category")
    uploaded_file = st.file_uploader("CSV file", type=["csv"])

    if uploaded_file is not None:
        preview = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)
        st.write(f"Preview ({len(preview)} rows):")
        st.dataframe(preview.head(10), use_container_width=True)

        if st.button("Predict batch", type="primary"):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/v1/predictbatch",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                    timeout=120,
                )
                if response.status_code == 200:
                    predictions = response.json()
                    results = preview.copy()
                    results["Predicted_Sales"] = [predictions[str(i)] for i in range(len(results))]
                    st.success(f"Predictions generated for {len(results)} rows.")
                    st.dataframe(results, use_container_width=True)
                    st.download_button("Download predictions as CSV", results.to_csv(index=False).encode("utf-8"),
                                       file_name="superkart_predictions.csv", mime="text/csv")
                else:
                    st.error(f"The prediction API returned an error: {response.json().get('error', response.text)}")
            except requests.exceptions.RequestException:
                st.error("Unable to connect to the prediction API. Check that the backend is running.")
