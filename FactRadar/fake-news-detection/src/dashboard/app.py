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
        
    # Initialize session state
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = None
        st.session_state.valid_df = None
        st.session_state.summary_metrics = None

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
                st.session_state.batch_results = all_results
                st.session_state.valid_df = valid_df
                st.session_state.summary_metrics = {
                    "total_processed": total_processed,
                    "total_failed": total_failed,
                    "total_time_ms": total_time_ms
                }
                
    if st.session_state.batch_results is not None:
        st.success("Batch processing complete!")
        
        all_results = st.session_state.batch_results
        valid_df = st.session_state.valid_df
        metrics = st.session_state.summary_metrics
        
        st.subheader("Summary Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Processed", metrics["total_processed"])
        col2.metric("Total Failed", metrics["total_failed"])
        col3.metric("Processing Time", f"{metrics['total_time_ms'] / 1000:.2f} s")
                
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

        st.markdown("---")
        
        @st.fragment
        def detailed_analysis_section():
            st.header("Detailed Analysis")
            st.write("Select an article to view LIME attributions, LLM explanation, and RAG verification.")
            
            # Get list of valid article IDs from the batch
            valid_ids = results_df['article_id'].tolist()
            selected_id = st.selectbox("Select Article ID", valid_ids)
            
            if "analysis_results" not in st.session_state:
                st.session_state.analysis_results = {}
                
            if st.button("Run Detailed Analysis"):
                st.session_state.active_analysis_id = selected_id
                
            if st.session_state.get("active_analysis_id") == selected_id:
                # Find the article text in valid_df
                selected_row = valid_df[valid_df['article_id'] == selected_id].iloc[0]
                article_text = selected_row['text']
                
                if selected_id not in st.session_state.analysis_results:
                    with st.spinner("Running deep analysis (this may take up to 30-40 seconds)..."):
                        try:
                            analyze_resp = requests.post(
                                f"{API_URL}/analyze",
                                json={"article_id": str(selected_id), "text": article_text},
                                timeout=120
                            )
                            analyze_resp.raise_for_status()
                            st.session_state.analysis_results[selected_id] = analyze_resp.json()
                        except requests.exceptions.RequestException as e:
                            st.error(f"Failed to fetch analysis: {e}")
                        except Exception as e:
                            st.error(f"An error occurred while displaying analysis: {e}")
                            
                if selected_id in st.session_state.analysis_results:
                    analysis = st.session_state.analysis_results[selected_id]
                    try:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Predicted Label", analysis["predicted_label"].upper())
                        with col2:
                            st.metric("Confidence", f"{analysis['confidence']:.2%}")
                        
                        st.subheader("LLM Explanation")
                        expl = analysis.get("explanation", {})
                        st.info(expl.get("explanation", "No explanation available."))
                        if expl.get("fallback_used"):
                            st.warning("Note: A fallback template was used instead of the LLM.")
                        
                        st.subheader("RAG Verification")
                        verif = analysis.get("verification", {})
                        if verif and verif.get("activated"):
                            st.write(f"**Verdict:** {verif.get('verdict', 'unknown').upper()}")
                            st.write(f"**Justification:** {verif.get('justification', '')}")
                            
                            if verif.get("recommend_review"):
                                st.error("⚠️ Human Review Recommended: Evidence contradicts prediction or is unparseable.")
                            else:
                                st.success("✅ Prediction is consistent with retrieved evidence.")
                                
                            if verif.get("evidence_count", 0) > 0:
                                with st.expander(f"View Retrieved Evidence ({verif.get('evidence_count')} sources)"):
                                    st.write("Evidence IDs: " + ", ".join(verif.get("evidence_ids", [])))
                        else:
                            reason = verif.get("reason", "unknown") if verif else "unknown"
                            st.write(f"RAG Verification not activated. Reason: `{reason}`")
                        
                        st.subheader("LIME Attributions")
                        tokens = analysis.get("top_contributing_tokens", [])
                        if tokens:
                            t_df = pd.DataFrame(tokens)
                            # Display bar chart of weights
                            fig, ax = plt.subplots(figsize=(8, 4))
                            colors = ['#ff9999' if w > 0 else '#99ccff' for w in t_df['weight']]
                            ax.bar(t_df['token'], t_df['weight'], color=colors)
                            ax.set_ylabel('Weight')
                            ax.set_title('Top Contributing Words (Red=Fake, Blue=Real)')
                            plt.xticks(rotation=45, ha='right')
                            fig.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.write("No LIME attributions available.")
                            
                    except Exception as e:
                        st.error(f"An error occurred while displaying analysis: {e}")
                        
        # Call the fragment function
        detailed_analysis_section()
