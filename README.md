# Old Car Price Prediction

A beginner-friendly Machine Learning + Streamlit project.

## Model
Random Forest Regression

## Dataset
`dataset/car_data.csv` contains 1,200 sample records created for this educational demo.

## Model performance on the included sample data
- MAE: ₹34,171
- R²: 0.919

## Run in VS Code

Open the `Old-Car-Price-Prediction` folder in VS Code.

### 1. Create virtual environment (optional)
```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

### 2. Install libraries
```bash
pip install -r requirements.txt
```

### 3. Start the app
```bash
streamlit run app.py
```

The browser will open the Streamlit app.

## Important
The included dataset is synthetic/sample data for a college project demo. For a real-world predictor, replace it with a reliable used-car dataset and retrain the model.
