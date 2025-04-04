import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from io import BytesIO

# ----------------------------- DATA (move to top to fix NameError)
raw_data = {
    "Phosphorus": {"T": [1000, 1050, 1100, 1150, 1200], "D": [3.2e-15, 1.2e-14, 3.9e-14, 1.3e-13, 3.4e-13]},
    "Boron":      {"T": [900, 1000, 1100, 1200],          "D": [4.2e-16, 2.7e-15, 1.4e-14, 6.7e-14]},
    "Arsenic":    {"T": [900, 1000, 1100],                 "D": [1.0e-15, 3.5e-15, 1.1e-14]},
    "Antimony":   {"T": [1000, 1050, 1100, 1150],          "D": [2.1e-15, 6.3e-15, 2.2e-14, 6.1e-14]},
    "Gallium":    {"T": [900, 1000, 1100],                 "D": [2.1e-16, 1.0e-15, 5.3e-15]},
    "Nitrogen":   {"T": [800, 900, 1000, 1100, 1200],      "D": [2.9e-20, 2.7e-19, 1.9e-18, 1.2e-17, 5.0e-17]}
}

# ----------------------------- INIT SESSION STATE
if 'history' not in st.session_state:
    st.session_state.history = []

if "plot_scale" not in st.session_state:
    st.session_state.plot_scale = "Log D"
if "temp_unit" not in st.session_state:
    st.session_state.temp_unit = "Celsius (°C)"

# ----------------------------- PAGE CONFIG
st.set_page_config(page_title="ML-Enhanced Diffusion Modeling", layout="wide")
st.title("🧬 DiffuLab: ML-Enhanced Diffusion Modeling")
st.caption("Powered by Physics-Informed Neural Networks • Built with ❤️ in Streamlit")

# ----------------------------- SIDEBAR
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Silicon_Structure.svg/2560px-Silicon_Structure.svg.png", use_container_width=True)
    st.subheader("🔧 Choose Settings")

    selected_dopant = st.selectbox("🧪 Dopant", list(raw_data.keys()))
    temp_input = st.slider("🌡️ Temperature (°C)", 600, 1300, 1000)
    dopant_2 = st.selectbox("Select Dopant 2", list(raw_data.keys()), index=1, key="dopant2")

    st.markdown("---")
    st.subheader("📁 Upload New Data")
    uploaded_file = st.file_uploader("Upload Dopant CSV (T, D)", type=["csv"])
    st.markdown("---")
    st.caption("🔬 Created by Rogelio Lopez")

# ----------------------------- HANDLE UPLOAD
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        raw_data[selected_dopant] = {"T": df.iloc[:, 0].tolist(), "D": df.iloc[:, 1].tolist()}
        st.success(f"✅ Loaded data for {selected_dopant} from CSV")
    except:
        st.error("❌ Failed to read uploaded file.")

# ----------------------------- PREPARE TRAINING DATA
X, y = [], []
for dopant in raw_data:
    T_vals = np.array(raw_data[dopant]['T']) + 273.15
    D_vals = np.log10(np.array(raw_data[dopant]['D']))
    X.extend(1 / T_vals)
    y.extend(D_vals)
X = np.array(X).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# ----------------------------- TRAIN PINN
@st.cache_resource
def train_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='tanh', input_shape=(1,)),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_scaled, y, epochs=1000, verbose=0)
    return model

model = train_model()

# ----------------------------- PREDICTION & FITTING
T_predict = np.linspace(600, 1300, 300) + 273.15
invT_predict = 1 / T_predict
invT_scaled = scaler_X.transform(invT_predict.reshape(-1, 1))
logD_pred = model.predict(invT_scaled)
D_pred = 10**logD_pred.flatten()

# Arrhenius Fit
X_fit = 1 / T_predict.reshape(-1, 1)
logD_fit = np.log10(D_pred)
A = np.vstack([X_fit.flatten(), np.ones_like(X_fit.flatten())]).T
Ea_fit, logD0_fit = np.linalg.lstsq(A, logD_fit, rcond=None)[0]
Ea_eV = Ea_fit * 8.617e-5
D0_val = 10**logD0_fit

# R² Score
y_pred_for_r2 = model.predict(invT_scaled)
r2 = r2_score(logD_fit, y_pred_for_r2)
st.metric("📊 R² Score of Model", f"{r2:.4f}")

# ----------------------------- PLOT
plt.style.use("seaborn-v0_8-darkgrid")
fig, ax = plt.subplots(figsize=(12, 6))
dopant_colors = {
    "Phosphorus": "red", "Boron": "blue", "Arsenic": "green",
    "Antimony": "orange", "Gallium": "purple", "Nitrogen": "brown"
}
dopant_1 = st.selectbox("Select Dopant 1", list(raw_data.keys()), index=0, key="dopant1")

for dopant in [dopant_1, dopant_2]:
    T_vals = np.array(raw_data[dopant]['T']) + 273.15
    D_vals = np.array(raw_data[dopant]['D'])
    ax.scatter(T_vals, np.log10(D_vals), label=f"{dopant} (Literature)",
               color=dopant_colors[dopant], s=60, edgecolors='k', alpha=0.8)

# PINN prediction line
if st.session_state.plot_scale == "Linear D":
    ax.plot(T_predict, D_pred, 'k--', label='PINN Prediction')
    ax.set_ylabel("Diffusivity [cm²/s]")
    ax.set_yscale("linear")
    ax.set_ylim(1e-20, 1e-12)
else:
    ax.plot(T_predict, logD_pred, 'k--', label='PINN Prediction')
    ax.set_ylabel("log₁₀(Diffusivity [cm²/s])")

if st.session_state.temp_unit == "Celsius (°C)":
    ax.set_xlabel("Temperature (°C)")
    ticks = np.linspace(600, 1300, 8)
    ax.set_xticks(ticks + 273.15)
    ax.set_xticklabels([f"{int(t)}" for t in ticks])
else:
    ax.set_xlabel("Temperature (K)")

arrhenius_eq = f"log₁₀(D) = {logD0_fit:.2f} - {Ea_fit:.2f}/kT"
ax.text(0.05, 0.92, arrhenius_eq, transform=ax.transAxes, fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8))
ax.set_title("📈 PINN Prediction vs Literature Data")
ax.legend()
st.pyplot(fig)

# ----------------------------- DISPLAY RESULTS
st.success(f"**Arrhenius Fit:** D = {D0_val:.2e} * exp(-{Ea_eV:.3f} eV / kT)")

if D0_val > 1e-13:
    category = "🟥 High Diffusivity"
elif D0_val > 1e-17:
    category = "🟨 Moderate Diffusivity"
else:
    category = "🟦 Low Diffusivity"
st.markdown(f"### 🧠 Diffusivity Category: {category}")

# ----------------------------- DOWNLOAD OPTIONS
arrhenius_df = pd.DataFrame({
    "Dopant": [selected_dopant],
    "D0 (cm²/s)": [D0_val],
    "Ea (eV)": [Ea_eV]
})
st.download_button("⬇️ Download Arrhenius Parameters",
                   arrhenius_df.to_csv(index=False).encode(),
                   file_name=f"{selected_dopant}_arrhenius.csv")

T_user = temp_input + 273.15
invT_user = 1 / T_user
D_user = 10 ** model.predict(scaler_X.transform([[invT_user]])).flatten()[0]
st.info(f"📍 At {temp_input}°C, predicted D ≈ {D_user:.2e} cm²/s")
