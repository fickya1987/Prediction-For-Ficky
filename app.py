import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA
import warnings
from sklearn.exceptions import ConvergenceWarning
from datetime import timedelta

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- FUNCTIONS ---

def random_forest_prediction(X_train, X_test, y_train, y_test):
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    best_rf.fit(X_train, y_train)
    rf_pred = best_rf.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_pred)
    mse_list.append(f"random_forest_mse,{rf_mse}")
    return rf_mse, rf_pred

def gradient_boosting_prediction(X_train, X_test, y_train, y_test):
    gb = GradientBoostingRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
    }
    grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_gb = grid_search.best_estimator_
    best_gb.fit(X_train, y_train)
    gb_pred = best_gb.predict(X_test)
    gb_mse = mean_squared_error(y_test, gb_pred)
    mse_list.append(f"gradient_boosting_mse,{gb_mse}")
    return gb_mse, gb_pred

def xgboost_prediction(X_train, X_test, y_train, y_test):
    xgb = XGBRegressor()
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.15, 0.35],
    }
    grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_xgb = grid_search.best_estimator_
    best_xgb.fit(X_train, y_train)
    xgboost_pred = best_xgb.predict(X_test)
    xgboost_mse = mean_squared_error(y_test, xgboost_pred)
    mse_list.append(f"xgboost_mse,{xgboost_mse}")
    return xgboost_mse, xgboost_pred

def svr_prediction(X_train, X_test, y_train, y_test):
    svr = SVR()
    param_grid = {
        'kernel': ['linear', 'poly', 'rbf'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    }
    grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_svr = grid_search.best_estimator_
    best_svr.fit(X_train, y_train)
    svr_pred = best_svr.predict(X_test)
    svr_mse = mean_squared_error(y_test, svr_pred)
    mse_list.append(f"svr_prediction_mse,{svr_mse}")
    return svr_mse, svr_pred

def stack_prediction(X_train, X_test, y_train, y_test):
    linear_reg = LinearRegression()
    mlp_reg = MLPRegressor(max_iter=2000, random_state=42)
    svr_reg = SVR()

    mlp_param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'activation': ['tanh', 'relu'],
        'alpha': [0.0001, 0.001],
        'learning_rate': ['constant', 'adaptive']
    }
    grid_search = GridSearchCV(mlp_reg, mlp_param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_mlp = grid_search.best_estimator_

    svr_param_grid = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    }
    grid_search = GridSearchCV(svr_reg, svr_param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_svr = grid_search.best_estimator_

    stacked_model = StackingRegressor(
        estimators=[('linear', linear_reg), ('mlp', best_mlp), ('svr', best_svr)],
        final_estimator=LinearRegression()
    )
    stacked_model.fit(X_train, y_train)
    stack_pred = stacked_model.predict(X_test)
    stack_mse = mean_squared_error(y_test, stack_pred)
    mse_list.append(f"stack_mse,{stack_mse}")
    return stack_mse, stack_pred

def arima_prediction(df, forecasting_period):
    model = ARIMA(df['Sales'], order=(5, 1, 2))
    results = model.fit()
    next_month_forecast = results.get_forecast(steps=forecasting_period)
    next_month_index = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=forecasting_period)
    next_month_sales = next_month_forecast.predicted_mean.values
    rounded_sales = pd.Series(next_month_sales).round().astype(int)
    rounded_sales_df_arima = pd.DataFrame({'Date': next_month_index, 'ARIMA': rounded_sales})

    y_true = df['Sales'].values[-forecasting_period:]
    y_pred = next_month_sales
    mse = mean_squared_error(y_true, y_pred)

    return rounded_sales_df_arima, mse

def prepare_data(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    return df



# --- STREAMLIT APP ---

st.title("Sales Prediction App")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
forecasting_period = st.slider("Select the forecasting period (days):", min_value=1, max_value=60, value=20)

if uploaded_file is not None:
    try:
        # Load and prepare data
        df = prepare_data(uploaded_file)
        # Display the first 5 rows of the dataframe
        st.subheader('Data Sample:')
        st.write(df.head().to_html(index=False), unsafe_allow_html=True)
        # Split into train and test based on the provided logic
        last_month_start = df['Date'].max() - pd.DateOffset(months=2)
        train_data = df[df['Date'] < last_month_start].copy()
        test_data = df[df['Date'] >= last_month_start].copy()
        
        # Prepare features (X) and target (y)
        X_train = train_data.drop(['Sales', 'Date'], axis=1)
        y_train = train_data['Sales']
        X_test = test_data.drop(['Sales', 'Date'], axis=1)
        y_test = test_data['Sales']

        # Convert categorical columns to numeric using one-hot encoding if any
        X_train = pd.get_dummies(X_train)
        X_test = pd.get_dummies(X_test)

        # Ensure both train and test sets have the same columns after encoding
        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

        # Check for missing values
        X_train.fillna(0, inplace=True)
        X_test.fillna(0, inplace=True)

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Initialize mse_list
        mse_list = []

        # Make predictions with all models
        rf_mse, rf_pred = random_forest_prediction(X_train_scaled, X_test_scaled, y_train, y_test)
        gb_mse, gb_pred = gradient_boosting_prediction(X_train_scaled, X_test_scaled, y_train, y_test)
        xgboost_mse, xgboost_pred = xgboost_prediction(X_train_scaled, X_test_scaled, y_train, y_test)
        svr_mse, svr_pred = svr_prediction(X_train_scaled, X_test_scaled, y_train, y_test)
        stack_mse, stack_pred = stack_prediction(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # ARIMA Prediction on Full Dataset (not on the test data)
        df_arima = df.copy()
        df_arima['Date'] = pd.to_datetime(df_arima['Date'])
        df_arima.set_index('Date', inplace=True)
        rounded_sales_df_arima, arima_mse = arima_prediction(df_arima[:-forecasting_period], forecasting_period)
        
        # Prepare the predictions for the last `forecasting_period` days
        last_date_in_train = df_arima.index[-forecasting_period - 1]
        next_month_index = pd.date_range(start=last_date_in_train + pd.DateOffset(days=1), periods=forecasting_period)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Date': test_data['Date'],
            'Actual Sales': y_test.values,
            'Random Forest Predicted Sales': rf_pred,
            'Gradient Boosting Predicted Sales': gb_pred,
            'XGBoost Predicted Sales': xgboost_pred,
            'SVR Predicted Sales': svr_pred,
            'Stack Predicted Sales': stack_pred,
        })

        # Parse mse_list and calculate the average of the top 5 MSE values
        parsed_mse_list = []
        for item in mse_list:
            model_name, mse_value = item.split(',')
            parsed_mse_list.append((model_name, float(mse_value)))

        parsed_mse_list.sort(key=lambda x: x[1])
        top_5_mse = parsed_mse_list[:5]
        average_top_5_mse = sum(mse for _, mse in top_5_mse) / len(top_5_mse)

        # Print the top 5 MSE values and their average
        st.subheader("Top 5 MSE values and their corresponding models:")
        for model, mse in top_5_mse:
            st.write(f"{model}: {mse}")
        st.write(f"Average of the top 5 MSE values: {average_top_5_mse}")

        # Calculate errors
        comparison_df['Gradient Boosting Error'] = abs(comparison_df['Actual Sales'] - comparison_df['Gradient Boosting Predicted Sales'])
        comparison_df['XGBoost Error'] = abs(comparison_df['Actual Sales'] - comparison_df['XGBoost Predicted Sales'])
        comparison_df['SVR Error'] = abs(comparison_df['Actual Sales'] - comparison_df['SVR Predicted Sales'])
        comparison_df['Stack Error'] = abs(comparison_df['Actual Sales'] - comparison_df['Stack Predicted Sales'])
        comparison_df['Random Forest Error'] = abs(comparison_df['Actual Sales'] - comparison_df['Random Forest Predicted Sales'])

        # Identify the model with the smallest error for each row
        error_columns = ['Gradient Boosting Error', 'XGBoost Error', 'SVR Error', 'Stack Error', 'Random Forest Error']
        comparison_df['Closest Model'] = comparison_df[error_columns].idxmin(axis=1)

        # Map the error column names to model names
        model_name_mapping = {
            'Gradient Boosting Error': 'Gradient Boosting',
            'XGBoost Error': 'XGBoost',
            'SVR Error': 'SVR',
            'Stack Error': 'Stack',
            'Random Forest Error': 'Random Forest'
        }
        comparison_df['Closest Model'] = comparison_df['Closest Model'].map(model_name_mapping)

        # Create a dictionary to map model names to their respective error columns
        model_to_error_column = {
            'Gradient Boosting': 'Gradient Boosting Error',
            'XGBoost': 'XGBoost Error',
            'SVR': 'SVR Error',
            'Stack': 'Stack Error',
            'Random Forest': 'Random Forest Error'
        }
        comparison_df['Closest Model Error'] = comparison_df.apply(lambda row: row[model_to_error_column[row['Closest Model']]], axis=1)

        # Add a column for the percentage deviation of the Closest Model Error from the Actual Sales
        comparison_df['Percentage Deviation'] = (comparison_df['Closest Model Error'] / comparison_df['Actual Sales']) * 100

        # Display comparison dataframe
        st.subheader("Actual vs. Predicted Sales Comparison (Non-ARIMA Models)")
        st.dataframe(comparison_df)

        # Create ARIMA DataFrame
        arima_df = pd.DataFrame({
            'Date': next_month_index,
            'ARIMA Predicted Sales': rounded_sales_df_arima['ARIMA'],
        })
        st.subheader("ARIMA Predicted Sales")
        st.dataframe(arima_df)

        st.subheader("Model Mean Squared Error (MSE)")
        st.write(f"Random Forest MSE: {rf_mse}")
        st.write(f"Gradient Boosting MSE: {gb_mse}")
        st.write(f"XGBoost MSE: {xgboost_mse}")
        st.write(f"SVR MSE: {svr_mse}")
        st.write(f"Stacking MSE: {stack_mse}")
        st.write(f"ARIMA MSE: {arima_mse:.2f}")

        # Determine best model based on MSE
        best_model_name = min(mse_list, key=lambda x: float(x.split(",")[1])).split(",")[0]
        st.write(f"Best Model based on MSE: {best_model_name}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
