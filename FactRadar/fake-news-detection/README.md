# FactRadar — Fake News Detection System

FactRadar is an end-to-end machine learning system designed to detect and explain fake news articles. It features a robust FastAPI backend serving a baseline TF-IDF + Logistic Regression model (with support for Hugging Face transformer pipelines), a Streamlit dashboard for interactive exploration, and advanced explainability capabilities using LIME, LLM-based explanations, and RAG (Retrieval-Augmented Generation) verification.

## Project Structure

```
fake-news-detection/
├── configs/            # Configuration files
├── data/               # Datasets and splits
├── deployment/         # Deployment scripts and Dockerfiles
├── mlruns/             # MLflow tracking directory
├── models/             # Serialized models (e.g., joblib)
├── notebooks/          # Jupyter notebooks for EDA and experimentation
├── src/                # Core source code
│   ├── dashboard/      # Streamlit dashboard app
│   ├── evaluation/     # Model evaluation scripts
│   ├── features/       # Feature engineering modules
│   ├── ingestion/      # Data ingestion and downloading scripts
│   ├── preprocessing/  # Text cleaning and preprocessing (e.g., clean_text.py)
│   ├── serving/        # FastAPI application (app.py) and API schemas
│   └── training/       # Model training scripts (baseline and advanced)
├── tests/              # Unit and integration tests
├── requirements.txt    # Python dependencies
└── requirements-lock.txt # Pinned dependencies
```

## Features

- **FastAPI Serving Layer**: High-performance API with endpoints for single prediction, batch prediction, and model health/versioning.
- **Explainable AI (XAI)**:
  - **LIME Integration**: Token-level contribution weights to understand why a specific prediction was made.
  - **LLM Explanations**: Natural language explanations attaching context to the model's prediction.
  - **RAG Verification**: Cross-referencing against a trusted reference index for fact-checking and validation.
- **MLflow Tracking**: Integrated experiment tracking and model registry for reproducible machine learning workflows.
- **Streamlit Dashboard**: A user-friendly web interface to interact with the API, visualize predictions, and analyze explanations.

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd FactRadar/fake-news-detection
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### 1. Start the FastAPI Backend
Start the backend server using Uvicorn. This will serve the machine learning model and expose the REST endpoints.

```bash
uvicorn src.serving.app:app --reload
```
The API will be accessible at `http://localhost:8000`. You can explore the interactive Swagger documentation at `http://localhost:8000/docs`.

### 2. Start the Streamlit Dashboard
In a separate terminal, launch the Streamlit dashboard to interact with the system via a UI.

```bash
streamlit run src/dashboard/app.py
```
*(Check the exact path to your Streamlit app if different)*

### 3. Access MLflow UI (Optional)
To view training experiments and model metrics:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

## API Endpoints Overview

The FastAPI backend provides several endpoints:

- `POST /predict`: Classify a single article as fake or real. Returns the predicted label and confidence.
- `POST /predict/batch`: Classify multiple articles in a single request.
- `POST /explain`: Get LIME-based token-level explanations for a single prediction.
- `POST /analyze`: Detailed analysis combining the base prediction, LIME explanation, LLM natural language explanation, and RAG verification.
- `GET /health`: Check if the model is loaded and the service is healthy.
- `GET /model/version`: Retrieve current model version and training metrics (Accuracy, F1, ROC-AUC, etc.).

## Model Architecture

The current system relies on a baseline **TF-IDF + Logistic Regression** pipeline serialized using `joblib`. 
- **Preprocessing**: Input text is cleaned (removing punctuation, lowercasing, etc.) using `src.preprocessing.clean_text` to prevent train/serve skew.
- **Handling Out-of-Distribution (OOD)**: The API includes heuristic checks for very short inputs (e.g., `< 3 tokens`), flagging them with a `low_confidence_ood` warning.

## Testing

Run the test suite using pytest to ensure all components are functioning correctly:

```bash
pytest tests/
```
