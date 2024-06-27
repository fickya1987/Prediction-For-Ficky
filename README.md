# Sales Prediction App 📈

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url-here)

This Streamlit application simplifies sales forecasting by leveraging machine learning and time series models. 

## Key Features ✨

- **Diverse Models:** Employs Random Forest, Gradient Boosting, XGBoost, SVR, Stacking, and ARIMA for robust predictions.
- **Hyperparameter Optimization:** Fine-tunes model parameters using GridSearchCV for enhanced accuracy.
- **Intuitive Interface:** Offers a user-friendly experience with a clear data upload process and interactive forecasting period selection.
- **Performance Evaluation:** Presents a comprehensive comparison table with actual sales and predictions from each model, alongside Mean Squared Error (MSE) metrics.
- **Full Dataset ARIMA Forecast:**  Provides an additional ARIMA forecast for the entire dataset, offering a holistic perspective on sales trends.
- **Automated Preprocessing:** Handles date feature extraction, categorical encoding, missing value imputation, and feature scaling, streamlining the workflow.

## How It Works ⚙️

1. **Upload CSV:** Upload your sales data CSV file (must include `Date` and `Sales` columns).
2. **Choose Forecast Period:** Use the slider to select the number of days to predict.
3. **Get Predictions:** The app automatically trains and evaluates the models, displaying results in a comparison table and a dedicated ARIMA forecast table.

## Get Started 🚀

1. **Clone:** `git clone https://https://github.com/satyam7010/Sales-Prediction-For-Each-Date/tree/main`
2. **Install:** `pip install -r requirements.txt` 
3. **Run:** `streamlit run app.py`

## Data Format 📄

Your CSV file should be structured as follows:


