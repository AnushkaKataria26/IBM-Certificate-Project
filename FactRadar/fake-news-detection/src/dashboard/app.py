import streamlit as st
import pandas as pd
import requests
import time
import math
import matplotlib.pyplot as plt

st.set_page_config(page_title="FactRadar Dashboard", layout="wide")

st.title("FactRadar Batch Inference Dashboard")

# Sidebar
st.sidebar.header("Configuration")
API_URL = st.sidebar.text_input("API Base URL", "http://localhost:8000")

st.header("Upload Articles")
uploaded_file = st.file_uploader("Upload a CSV with 'article_id', 'title', and 'text' columns", type=["csv"])

def process_batch(chunk, api_url):
    articles = chunk.to_dict(orient="records")
    try:
        response = requests.post(
            f"{api_url}/predict/batch",
            json={"articles": articles},
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to the API or received an error status: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
        
    required_cols = {"article_id", "title", "text"}
    if not required_cols.issubset(df.columns):
        st.error(f"Invalid CSV structure. Missing required columns. Found: {list(df.columns)}")
        st.stop()
        
    original_count = len(df)
    
    # Preprocessing: convert 'article_id' to string
    df['article_id'] = df['article_id'].astype(str)
    # Ensure title and text are string
    df['title'] = df['title'].fillna("").astype(str)
    
    # Filter empty texts
    # Handle pandas NA and whitespace
    df['text'] = df['text'].fillna("")
    df['text'] = df['text'].astype(str)
    
    # Filter out empty or whitespace-only
    valid_df = df[df['text'].str.strip() != ""]
    skipped_count = original_count - len(valid_df)
    
    if skipped_count > 0:
        st.warning(f"Skipped {skipped_count} row(s) containing null or empty text.")
        
    if len(valid_df) == 0:
        st.error("No valid rows to process after filtering.")
        st.stop()
        
    if st.button("Process Batch"):
        with st.spinner("Processing..."):
            max_batch_size = 500
            total_rows = len(valid_df)
            num_chunks = math.ceil(total_rows / max_batch_size)
            
            if num_chunks > 1:
                st.info(f"File contains {total_rows} rows. Processing in {num_chunks} chunks of {max_batch_size}.")
                
            all_results = []
            total_processed = 0
            total_failed = 0
            total_time_ms = 0.0
            
            error_occurred = False
            
            progress_bar = st.progress(0)
            
            for i in range(num_chunks):
                chunk = valid_df.iloc[i * max_batch_size : (i + 1) * max_batch_size]
                chunk_records = chunk[['article_id', 'title', 'text']]
                
                resp_data = process_batch(chunk_records, API_URL)
                
                if resp_data is None:
                    error_occurred = True
                    break
                    
                all_results.extend(resp_data.get("results", []))
                
                summary = resp_data.get("summary", {})
                total_processed += summary.get("total_processed", 0)
                total_failed += summary.get("total_failed", 0)
                total_time_ms += summary.get("processing_time_ms", 0.0)
                
                if summary.get("warnings"):
                    for w in summary["warnings"]:
                        st.warning(w)
                
                progress_bar.progress((i + 1) / num_chunks)
                
            if not error_occurred:
                st.success("Batch processing complete!")
                
                st.subheader("Summary Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Processed", total_processed)
                col2.metric("Total Failed", total_failed)
                col3.metric("Processing Time", f"{total_time_ms / 1000:.2f} s")
                
                # Results Table
                results_df = pd.DataFrame(all_results)
                
                st.subheader("Predictions")
                st.dataframe(results_df[['article_id', 'predicted_label', 'confidence', 'warning']])
                
                # Charts
                st.subheader("Analysis")
                
                if not results_df.empty and 'predicted_label' in results_df.columns:
                    valid_predictions = results_df.dropna(subset=['predicted_label'])
                    
                    if not valid_predictions.empty:
                        col_chart1, col_chart2 = st.columns(2)
                        
                        with col_chart1:
                            st.write("**Prediction Distribution**")
                            # Count of Fake vs Real
                            counts = valid_predictions['predicted_label'].value_counts()
                            st.bar_chart(counts)
                            
                        with col_chart2:
                            st.write("**Confidence Score Histogram**")
                            fig, ax = plt.subplots()
                            ax.hist(valid_predictions['confidence'].dropna(), bins=20, color='skyblue', edgecolor='black')
                            ax.set_title("Confidence Scores")
                            ax.set_xlabel("Confidence")
                            ax.set_ylabel("Frequency")
                            st.pyplot(fig)
                    else:
                        st.info("No valid predictions available for charts.")
