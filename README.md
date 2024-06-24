# Sales-Prediction-For-Each-Date
This user-friendly Streamlit app streamlines sales forecasting. Upload your CSV data, select a forecasting period, and get predictions from multiple machine learning models, including ARIMA. Easily compare model performance with clear metrics and visualize future sales trends.


**Documentation**

Here's an extended explanation of how the app works:

1. **Data Input:**
   - The user uploads a CSV file with at least the `Date` and `Sales` columns.
   - Additional columns can be included as potential features for prediction.

2. **Data Preparation:**
   - The app automatically transforms the data:
     - Extracts date components (year, month, day, etc.) for time-series features.
     - Encodes categorical variables into numerical format.
     - Handles missing values.
     - Scales features to ensure they have similar ranges, which is often important for machine learning algorithms.

3. **Model Training and Prediction:**
   - The app uses six different models (Random Forest, Gradient Boosting, XGBoost, SVR, Stacking, and ARIMA) for forecasting.
   - GridSearchCV is applied to each model to find the best hyperparameters, improving their predictive accuracy.
   - Predictions are made for the selected forecasting period.
   - ARIMA, specifically, uses the full dataset for its forecast, providing a potentially different perspective than other models that use the test set for evaluation.

4. **Evaluation and Presentation:**
   - The app calculates and displays the Mean Squared Error (MSE) for each model, allowing users to compare their performance.
   - A table is presented with actual sales, predictions from each model, and an ARIMA-specific table for the entire dataset's forecast.
   - The model with the lowest MSE is identified as the "best" model for this particular dataset.

5. **High-Deviation Analysis:**
   - While not included in the Streamlit app directly, the code has provisions to analyze cases where predicted sales deviate significantly from actual sales. 

**Key Advantages**

- User-friendly interface with minimal setup required.
- Flexibility to compare multiple models easily.
- Automatic data preprocessing simplifies the workflow.
- Identifies the best-performing model for quick decision-making.
- Provides a separate ARIMA forecast for the full dataset, adding another layer of analysis.



6. **How to Run:**

Prerequisites: Ensure you have Python and the required libraries installed (pandas, scikit-learn, streamlit, statsmodels, xgboost). You can install them using:

-Bash
pip install pandas scikit-learn streamlit statsmodels xgboost
Use code with caution.
content_copy
Open Your Terminal: Navigate to the directory where you've saved the app.py file.

-Run the App: Type the following command and press Enter:

-Bash
streamlit run app.py
Use code with caution.
content_copy
This will start the app, and a new browser window will open automatically. You should see the app's interface where you can upload your data and begin exploring sales predictions.
