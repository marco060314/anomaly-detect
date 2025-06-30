import streamlit as st
import pandas as pd
import tempfile
from anomaly_detector import AnomalyDetector
from organization import organize_data
import matplotlib.pyplot as plt

st.set_page_config(page_title="Anomaly Detection App", layout="wide")
st.title("🔍 Multi-Model Anomaly Detection Dashboard")

uploaded_file = st.file_uploader("Upload your CSV, Excel, or JSON file", type=["csv", "xls", "xlsx", "json"])

if uploaded_file:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_filepath = tmp_file.name

    # Load and clean data using organize_data
    org = organize_data()
    org.load_file(tmp_filepath)
    df = org.clean_data()

    st.subheader("📊 Cleaned Data Preview")
    st.dataframe(df.head())

    # Initialize detector with clean data
    detector = AnomalyDetector(df)

    st.markdown("---")
    st.subheader("⚙️ Run Detection Methods")

    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)

    if col1.button("Run All (Auto)"):
        results = detector.run_auto()
        st.success("✅ Auto anomaly detection completed.")
        st.dataframe(results.head())

    if col2.button("Z-Score Only"):
        detector.detect_statistical()
        st.success("✅ Z-Score detection complete.")
        st.dataframe(detector.results.head())

    if col3.button("Isolation Forest"):
        detector.detect_isolation_forest()
        st.success("✅ Isolation Forest detection complete.")
        st.dataframe(detector.results.head())

    if col4.button("Local Outlier Factor"):
        detector.detect_lof()
        st.success("✅ LOF detection complete.")
        st.dataframe(detector.results.head())

    if col5.button("One-Class SVM"):
        detector.detect_one_class_svm()
        st.success("✅ SVM detection complete.")
        st.dataframe(detector.results.head())

    if st.button("Run Autoencoder (Deep Learning)"):
        detector.detect_autoencoder()
        st.success("✅ Autoencoder detection complete.")
        st.dataframe(detector.results.head())

    if st.button("Show Combined Results"):
        results = detector.combine_results()
        st.success("✅ Combined anomaly score calculated.")
        st.dataframe(results.head())

        if "consensus_flag" in results.columns:
            st.subheader("🔴 Anomalies Detected (Consensus ≥ 2 flags)")
            st.write(f"Total flagged: {results['consensus_flag'].sum()} rows")
            st.dataframe(results[results['consensus_flag'] == 1])

            st.subheader("📈 Anomaly Score Distribution")
            fig, ax = plt.subplots()
            results['anomaly_score'].hist(bins=20, ax=ax)
            st.pyplot(fig)
