import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import pandas as pd
APP_DIR = Path(__file__).resolve().parent 
PROJECT_ROOT = APP_DIR.parent

# --- Set Streamlit Page Configuration (MUST be called first) ---
st.set_page_config(page_title="Real-Time Fraud Detection", layout="wide")

# --- Load Saved Model and Scaler ---
try:
    with open(PROJECT_ROOT / "models" /'best_xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(PROJECT_ROOT / "models" /'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Error: Model or Scaler file not found. Please ensure 'best_xgb_model.pkl' and 'scaler.pkl' are in the directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model/scaler: {e}")
    st.stop()

# --- Define Feature Lists (MUST match your model training columns) ---
SCALER_FIT_FEATURES = ['amount', 'oldbalanceOrg', 'oldbalanceDest', 'errorBalanceOrig', 'balanceChangeOrig', 'errorBalanceDest']
CYCLICAL_FEATURES = ['hour_sin', 'hour_cos']
CATEGORICAL_FEATURES = ['error_flag', 'isMerchantDest', 'type_CASH_OUT', 'type_TRANSFER' ]

# Full list of features used for prediction
FEATURE_ORDER = [
    'amount', 
    'oldbalanceOrg', 
    'oldbalanceDest', 
    'errorBalanceOrig', 
    'error_flag',          # Position 4
    'isMerchantDest',      # Position 5
    'balanceChangeOrig', 
    'hour_sin', 
    'hour_cos', 
    'errorBalanceDest', 
    'type_CASH_OUT', 
    'type_TRANSFER'
]
#FEATURE_ORDER = NUMERICAL_FEATURES + CYCLICAL_FEATURES + CATEGORICAL_FEATURES


def engineer_features(input_df):
    """Performs feature engineering on the single input DataFrame."""
    
    # 1. Cyclical Encoding
    hour_of_day = input_df['hour_of_day'].iloc[0]
    hours_in_day = 24
    input_df['hour_sin'] = np.sin(2 * np.pi * hour_of_day / hours_in_day)
    input_df['hour_cos'] = np.cos(2 * np.pi * hour_of_day / hours_in_day)
    
    # Store transaction type before dropping the column for the OHE logic
    #transaction_type = input_df['type'].iloc[0]

    # 2. Balance Features
   
    transaction_type = input_df['type'].iloc[0]
    input_df['balanceChangeOrig'] = input_df['newbalanceOrig'] - input_df['oldbalanceOrg']
    
    # --- CRITICAL FIX 1: errorBalanceOrig (Already done) ---
    if transaction_type in ['CASH_OUT', 'TRANSFER']:
        input_df['errorBalanceOrig'] = input_df['balanceChangeOrig'] + input_df['amount']
    elif transaction_type in ['CASH_IN', 'PAYMENT', 'DEBIT']:
        input_df['errorBalanceOrig'] = input_df['balanceChangeOrig'] - input_df['amount']
        
    # --- CRITICAL FIX 2: errorBalanceDest (NEW ISSUE) ---
    if transaction_type in ['CASH_OUT', 'TRANSFER']:
        # Only calculate destination error for types that involve a destination receiving funds
        input_df['errorBalanceDest'] = input_df['newbalanceDest'] - input_df['oldbalanceDest'] - input_df['amount']
    else:
        # For CASH_IN, PAYMENT, DEBIT, the destination error is irrelevant (or should be 0)
        # Setting this to 0 prevents the massive error outlier.
        input_df['errorBalanceDest'] = 0.0
        
    input_df['error_flag'] = (input_df['errorBalanceOrig'].abs() > 0.001).astype(int)
    
    # 3. Merchant Flag
    input_df['isMerchantDest'] = 1 if input_df['nameDest'].iloc[0].startswith('M') else 0

    # --- 4. ROBUST ONE-HOT ENCODING AND ALIGNMENT (CRITICAL FIX) ---
    
    # Get the list of ALL required OHE columns from your feature definition
    OHE_COLUMNS = [c for c in FEATURE_ORDER if c.startswith('type_')]
    
    # Create all possible OHE columns and set them to 0
    for col in OHE_COLUMNS:
        input_df[col] = 0
        
    # Set the column corresponding to the input type to 1
    target_col = f'type_{transaction_type}'
    if target_col in input_df.columns:
        input_df[target_col] = 1
    else:
        # This should not happen if the selectbox is aligned with FEATURE_ORDER
        st.warning(f"Warning: Transaction type '{transaction_type}' does not match any OHE column.")
        
    # 5. FINAL CLEANUP: Remove the original 'type' column
    # Use .loc to avoid chained assignment warnings, but .drop is fine here.
    input_df = input_df.drop(['type', 'nameDest', 'hour_of_day', 'newbalanceOrg', 'newbalanceDest'], axis=1, errors='ignore')

    # The final DataFrame is returned. We rely on the calling function to select and order.
    return input_df


def main():
    st.title("Real-Time Financial Fraud Detector")
    st.markdown("---")
    st.header("Input Transaction Details")

    # --- User Input Form ---
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Transaction Core Details")
            transaction_type = st.selectbox(
                "Transaction Type",
                ['CASH_OUT', 'TRANSFER', 'CASH_IN', 'PAYMENT', 'DEBIT']
            )
            amount = st.number_input("Amount ($)", min_value=1.0, value=1000.0)
            hour_of_day = st.slider("Hour of Day (0-23)", 0, 23, 12)

        with col2:
            st.subheader("Originator Account")
            oldbalanceOrg = st.number_input("Old Balance Originator", min_value=0.0, value=10000.0) # Changed from 0.0
            newbalanceOrig = st.number_input("New Balance Originator", min_value=0.0, value=11000.0) # Changed from 0.0
            
            
        with col3:
            st.subheader("Destination Account")
            nameDest = st.text_input("Destination ID (e.g., C12345 or M67890)", value="C12345")
            oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, value=5000.0) # Changed from 0.0
            # For CASH_IN, Destination Balance doesn't change much for the Destination (as the recipient is the originator)
            # But let's assume a generic transaction:
            newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, value=6000.0) # Changed from 0.0

            # Ensure amount is not 1.0
            
        submitted = st.form_submit_button("Check for Fraud")

    # --- Prediction Logic ---
    if submitted:
        # 1. Create Input DataFrame
        raw_data = {
            'amount': [amount],
            'oldbalanceOrg': [oldbalanceOrg],
            'newbalanceOrig': [newbalanceOrig],
            'oldbalanceDest': [oldbalanceDest],
            'newbalanceDest': [newbalanceDest],
            'type': [transaction_type],
            'nameDest': [nameDest],
            'hour_of_day': [hour_of_day]
        }
        RAW_INPUT_COLUMNS = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest','newbalanceDest', 'type', 'nameDest', 'hour_of_day']
        raw_df = pd.DataFrame(raw_data, columns=RAW_INPUT_COLUMNS)
        # raw_df = pd.DataFrame(raw_data)

        # 2. Engineer Features
        engineered_df = engineer_features(raw_df.copy())
        
        # 3. Scale Numerical Features (CRUCIAL: Ensure Feature Order)
        
        SCALER_FIT_FEATURES = ['amount', 'oldbalanceOrg', 'oldbalanceDest', 'errorBalanceOrig', 'balanceChangeOrig', 'errorBalanceDest' # 6 features
        ]
        
        # NOTE: Your original NUMERICAL_FEATURES was missing 'hour_sin' and 'hour_cos'
        try:
            # Select ONLY the 6 features the scaler was trained on
            df_to_transform = engineered_df[SCALER_FIT_FEATURES]
            scaled_features_array = scaler.transform(df_to_transform)
            scaled_df = pd.DataFrame(scaled_features_array, columns=SCALER_FIT_FEATURES)
            
        except ValueError as e:
            # If this error still happens, the scaler file is fundamentally broken.
            st.error(f"FATAL SCALING ERROR: Could not transform features. Error: {e}")
            st.stop()
       
       # --- 4. CREATE FINAL ALIGNED VECTOR (15 Features) ---
        
        final_data_dict = {}
        
        for feature in FEATURE_ORDER:
            if feature in SCALER_FIT_FEATURES:
                # Use the scaled value for the 6 known numerical features
                final_data_dict[feature] = scaled_df[feature].iloc[0]
            else:
                # Use the UNMODIFIED (unscaled) value for the rest (cyclical, binary, OHE)
                final_data_dict[feature] = engineered_df[feature].iloc[0] 
        
        # Ensure the final DataFrame is built and ordered correctly
        df_final_prediction = pd.DataFrame([final_data_dict])[FEATURE_ORDER] 

        # 6. Predict
        prediction_proba = model.predict_proba(df_final_prediction)[:, 1]
        fraud_proba = prediction_proba[0]
        
  
        # 7. Display Result
        st.markdown("---")
        st.subheader("Prediction Result")

        # --- FINAL OVERRIDE LOGIC: Check Universal Financial Consistency ---

        # We check the error_flag, which is 0 if the unscaled balance error is near zero.
        # This bypasses the complexity of comparing two different scaled 'zero' values.
        is_financially_consistent = (df_final_prediction['error_flag'].iloc[0] == 0)

        # If the model gives a high score, but the financial metrics are clean, we overrule it.
        if is_financially_consistent and fraud_proba >= 0.5:
    
            st.success(f"✅ Transaction Appears Legitimate (Model Overruled)")
            st.markdown("Reason: **Zero calculated balance error** (confirmed by `error_flag=0`), overriding the model's high alert due to temporal bias.")
            st.markdown(f"Model Score (dominated by hour): {fraud_proba * 100:.2f}%")
            st.markdown("Recommendation: **APPROVE** the transaction.")
    
        # --- Normal Model Prediction ---
        elif fraud_proba >= 0.5:
            # This now catches cases where error_flag=1 (inconsistent) AND high proba
            st.error(f"⚠️ **HIGH FRAUD ALERT**")
            st.markdown(f"The transaction has a **{fraud_proba * 100:.2f}%** probability of being fraudulent.")
            st.markdown("Recommendation: **HOLD** the transaction and initiate a manual review immediately.")
        else:
            # This is for transactions that are approved by the model (low proba)
            st.success(f"✅ Transaction Appears Legitimate")
            st.markdown(f"The transaction has a low fraud probability of {fraud_proba * 100:.2f}%.")
            st.markdown("Recommendation: **APPROVE** the transaction.")

        st.markdown("---")
        st.caption("Feature Vector used for prediction (Scaled and Aligned):")
        st.write(df_final_prediction)
                      

if __name__ == "__main__":
    main()