# Disease Prediction System

A machine learning-based application that predicts possible diseases from user-selected symptoms. The application provides an interactive interface using Streamlit and stores prediction history using SQLite.

## Features

- Symptom-based disease prediction
- Top 3 disease predictions with confidence scores
- Machine learning model comparison
- Prediction history
- Recommended doctor based on predicted disease
- Disease information and precautions
- Downloadable prediction report
- Interactive Streamlit interface

## Machine Learning

The project compares multiple classification algorithms:

- Random Forest
- Decision Tree
- K-Nearest Neighbors (KNN)

The models are evaluated using accuracy, cross-validation, and classification reports. The best-performing model is saved and used by the application.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- SQLite

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
