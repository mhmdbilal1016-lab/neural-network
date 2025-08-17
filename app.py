import io
import os
import base64
import time
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import plotly.express as px

st.set_page_config(page_title="Fuel CO₂ Emissions — ML App", layout="wide")

# --- Sidebar controls ---
st.sidebar.title("Controls")
st.sidebar.write("Upload your dataset (CSV or XLSX). The target must be **CO2EMISSIONS**.")
data_file = st.sidebar.file_uploader("Dataset file (.csv or .xlsx)", type=["csv", "xlsx"], accept_multiple_files=False)

st.sidebar.subheader("Training parameters")
test_size = st.sidebar.slider("Test size", 0.1, 0.3, 0.2, 0.05)
epochs = st.sidebar.slider("Epochs", 10, 300, 80, 10)
batch_size = st.sidebar.selectbox("Batch size", [16, 32, 64, 128, 256], index=2)
early_stop = st.sidebar.checkbox("Early stopping (patience=10)", value=True)
use_categoricals = st.sidebar.checkbox("Use categorical features (one-hot encode)", value=False)

st.sidebar.subheader("Optional")
default_cols_to_drop = ["MODELYEAR","FUELCONSUMPTION_COMB_MPG","Brands"]
cols_to_drop = st.sidebar.text_input("Columns to drop (comma-separated)", ",".join(default_cols_to_drop))

st.title("🚗💨 Fuel Consumption → CO₂ Emissions (Regression)")
st.caption("Interactive Streamlit app generated from your notebook. Upload data, explore, train a Keras model, and make predictions.")

# --- Helpers ---
@st.cache_data(show_spinner=False)
def load_data(file):
    if file is None:
        return None
    try:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)  # requires openpyxl
        return df
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None

def clean_dataframe(df, drop_cols):
    df = df.copy()
    # Drop listed columns if present
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    # Basic NA handling: numeric -> fill median; categorical -> fill mode
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    if num_cols:
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for c in cat_cols:
        if df[c].isna().any():
            mode = df[c].mode(dropna=True)
            df[c] = df[c].fillna(mode.iloc[0] if not mode.empty else "unknown")
    return df

def preprocess(df, target="CO2EMISSIONS", use_cats=False):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    y = df[target].astype(float)
    X = df.drop(columns=[target]).copy()
    if use_cats:
        X = pd.get_dummies(X, drop_first=True)
    else:
        # numeric-only
        X = X.select_dtypes(include=np.number)
    if X.shape[1] == 0:
        raise ValueError("No predictor columns found. Enable categorical features or provide numeric predictors.")
    return X, y

def build_model(input_dim: int) -> Sequential:
    model = Sequential([
        Dense(64, activation="relu", input_shape=(input_dim,)),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def save_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

# --- Main flow ---
df = load_data(data_file)
if df is None:
    st.info("👆 Upload your dataset to begin (e.g., `FuelConsumptionCo2.csv.xlsx`).")
    st.stop()

# Clean + basic EDA
drop_list = [c.strip() for c in cols_to_drop.split(",") if c.strip()]
df_raw = df.copy()
df = clean_dataframe(df, drop_list)

st.subheader("1) Data Preview & EDA")
with st.expander("Show data head", expanded=True):
    st.dataframe(df.head(30), use_container_width=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{df.shape[0]:,}")
c2.metric("Columns", f"{df.shape[1]:,}")
c3.metric("Missing values", f"{int(df.isna().sum().sum()):,}")
c4.metric("Duplicates removed", f"{max(0, df_raw.shape[0]-df.shape[0]):,}")

if "CO2EMISSIONS" in df.columns:
    try:
        fig = px.histogram(df, x="CO2EMISSIONS", nbins=40, title="CO₂ Emissions Distribution (g/km)")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

st.subheader("2) Feature Engineering")
try:
    X, y = preprocess(df, target="CO2EMISSIONS", use_cats=use_categoricals)
    st.write(f"Using **{X.shape[1]}** features. Target samples: **{y.shape[0]:,}**")
    with st.expander("Feature columns", expanded=False):
        st.write(list(X.columns))
except Exception as e:
    st.error(e)
    st.stop()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
st.subheader("3) Train Model")
model = build_model(X_train_scaled.shape[1])
callbacks = []
if early_stop:
    callbacks.append(EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True))

with st.spinner("Training model..."):
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks
    )

# Plot training curves
hist_df = pd.DataFrame(history.history)
if not hist_df.empty:
    tr_col1, tr_col2 = st.columns(2)
    with tr_col1:
        fig1 = px.line(hist_df, y=["loss","val_loss"], title="Loss over epochs")
        st.plotly_chart(fig1, use_container_width=True)
    with tr_col2:
        # if 'mae' exists
        mae_cols = [c for c in hist_df.columns if "mae" in c]
        if mae_cols:
            fig2 = px.line(hist_df, y=mae_cols, title="MAE over epochs")
            st.plotly_chart(fig2, use_container_width=True)

# Evaluate
y_pred = model.predict(X_test_scaled, verbose=0).ravel()
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.success(f"**Test MAE:** {mae:.2f} g/km  | **R²:** {r2:.3f}")

# Save artifacts
out_dir = "artifacts"
os.makedirs(out_dir, exist_ok=True)
model_path = os.path.join(out_dir, "co2_model.keras")
scaler_path = os.path.join(out_dir, "scaler.pkl")
columns_path = os.path.join(out_dir, "feature_columns.txt")

model.save(model_path)
joblib.dump(scaler, scaler_path)
with open(columns_path, "w") as f:
    f.write("\n".join(map(str, X.columns)))

st.subheader("4) Download Artifacts")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.download_button("Download model (.keras)", data=save_bytes(model_path), file_name="co2_model.keras")
with col_b:
    st.download_button("Download scaler (joblib)", data=save_bytes(scaler_path), file_name="scaler.pkl")
with col_c:
    st.download_button("Download feature columns", data=save_bytes(columns_path), file_name="feature_columns.txt")

# --- Inference (single sample) ---
st.subheader("5) Predict")
st.write("Enter values for the trained feature columns below.")

# Build a form with inputs for each feature; fallback to median
medians = pd.Series(np.median(X_train, axis=0), index=X.columns)
with st.form("single_infer"):
    inputs = {}
    for col in X.columns:
        val = float(medians[col]) if col in medians else 0.0
        inputs[col] = st.number_input(col, value=float(val))
    submitted = st.form_submit_button("Predict CO₂")
if submitted:
    sample_df = pd.DataFrame([inputs])
    sample_scaled = scaler.transform(sample_df)
    pred = model.predict(sample_scaled, verbose=0).ravel()[0]
    st.info(f"**Predicted CO₂ emissions:** {pred:.2f} g/km")

# --- Batch inference ---
st.subheader("Batch predictions (optional)")
inf_file = st.file_uploader("Upload new data for prediction (CSV/XLSX)", type=["csv","xlsx"], key="predict_file")
if inf_file is not None:
    try:
        if inf_file.name.lower().endswith(".csv"):
            new_df = pd.read_csv(inf_file)
        else:
            new_df = pd.read_excel(inf_file)
        new_df_clean = clean_dataframe(new_df, drop_list)
        # Engineering same as training
        if use_categoricals:
            new_X = pd.get_dummies(new_df_clean, drop_first=True)
        else:
            new_X = new_df_clean.select_dtypes(include=np.number)
        # Align columns
        for c in X.columns:
            if c not in new_X.columns:
                new_X[c] = 0
        new_X = new_X[X.columns]  # reorder/trim
        # Fill any remaining NAs
        new_X = new_X.fillna(medians.to_dict())
        new_scaled = scaler.transform(new_X)
        preds = model.predict(new_scaled, verbose=0).ravel()
        out = new_df.copy()
        out["PRED_CO2"] = preds
        st.write("Preview of predictions:")
        st.dataframe(out.head(30), use_container_width=True)

        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Prediction failed: {e}")