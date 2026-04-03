import streamlit as st
import requests
import pandas as pd
import json

API_URL = "http://13.61.7.40:8000/"

st.set_page_config(page_title="AutoDML", layout="wide")

st.title("AutoDML")
st.markdown("Train ML models automatically with zero code")

uploaded_file = st.file_uploader(
    "📂 Upload Dataset (CSV / Excel)", type=["csv", "xlsx"]
)
MAX_SIZE_MB = 5

if uploaded_file:
    # Check file size
    file_size = uploaded_file.size / (1024 * 1024)

    if file_size > MAX_SIZE_MB:
        st.error(f"❌ File too large! Max allowed size is {MAX_SIZE_MB} MB")
        st.stop()

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ Dataset Loaded Successfully")

        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Shape:**", df.shape)

        with col2:
            st.write("**Columns:**", list(df.columns))

        target = st.selectbox("🎯 Select Target Column", df.columns)

        if st.button("🔥 Train Model", use_container_width=True):
            with st.spinner("Training model... please wait ⏳"):
                try:
                    uploaded_file.seek(0)

                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}

                    params = {"target": target}

                    response = requests.post(
                        f"{API_URL}train", files=files, params=params
                    )

                    result = response.json()

                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success("✅ Model Trained Successfully!")

                        st.subheader("📌 Training Summary")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Rows", result["shape"][0])

                        with col2:
                            st.metric("Columns", result["shape"][1])

                        st.write("**Target Column:**", result["target"])

                        tab1, tab2, tab3 = st.tabs(
                            ["📈 Evaluation", "🧠 Analysis", "📥 Input Structure"]
                        )

                        with tab1:
                            st.json(result["evaluation_report"])

                        with tab2:
                            st.json(result["analysis_report"])

                        with tab3:
                            st.write(result["input_structure"])

                        st.subheader("📄 Download Report")

                        try:
                            report_response = requests.get(f"{API_URL}report")
                        
                            if report_response.status_code == 200:
                                st.download_button(
                                    label="📥 Download PDF Report",
                                    data=report_response.content,
                                    file_name="Report.pdf",
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                            else:
                                st.warning("⚠️ Report not available yet. Train model first.")
                        
                        except Exception as e:
                            st.error(f"Error fetching report: {e}")

                except Exception as e:
                    st.error(f"⚠️ Error: {e}")

        st.divider()
        st.subheader("🔮 Make Prediction")

        st.markdown("Enter input as JSON format:")

        example = {col: "value" for col in df.columns if col != target}
        st.code(json.dumps(example, indent=2), language="json")

        input_json = st.text_area("✍️ Input JSON")

        if st.button("Predict", use_container_width=True):
            try:
                data = json.loads(input_json)

                response = requests.post(f"{API_URL}predict", json=data)

                result = response.json()

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success("✅ Prediction Successful")
                    st.write("Prediction:", result["prediction"])

            except Exception as e:
                st.error(f"Invalid JSON or API error: {e}")

    except Exception as e:
        st.error(f"Error loading file: {e}")

else:
    st.info("👆 Upload a dataset to get started")

st.markdown("---")
st.markdown("Built with  using Streamlit | AutoDML Project")
