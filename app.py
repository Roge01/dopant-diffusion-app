import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from io import BytesIO

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

st.set_page_config(page_title="Dopant Diffusivity PINN App", layout="wide")
st.title("🧪 Unified PINN for Dopant Diffusion in Silicon")

# Sidebar user input
with st.sidebar:
    st.header("📌 Dopant Settings")
    selected_dopant = st.selectbox("Choose a dopant", [
        "Phosphorus", "Boron", "Arsenic", "Antimony", "Gallium", "Nitrogen"])

    temp_input = st.slider("Temperature (°C)", 600, 1300, 1000)
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload New Dopant CSV", type=["csv"])

# Hardcoded dopant data
raw_data = {
    "Phosphorus": {"T": [1000, 1050, 1100, 1150, 1200], "D": [3.2e-15, 1.2e-14, 3.9e-14, 1.3e-13, 3.4e-13]},
    "Boron":      {"T": [900, 1000, 1100, 1200],          "D": [4.2e-16, 2.7e-15, 1.4e-14, 6.7e-14]},
    "Arsenic":    {"T": [900, 1000, 1100],                 "D": [1.0e-15, 3.5e-15, 1.1e-14]},
    "Antimony":   {"T": [1000, 1050, 1100, 1150],          "D": [2.1e-15, 6.3e-15, 2.2e-14, 6.1e-14]},
    "Gallium":    {"T": [900, 1000, 1100],                 "D": [2.1e-16, 1.0e-15, 5.3e-15]},
    "Nitrogen":   {"T": [800, 900, 1000, 1100, 1200],      "D": [2.9e-20, 2.7e-19, 1.9e-18, 1.2e-17, 5.0e-17]}
}

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        raw_data[selected_dopant] = {"T": df.iloc[:, 0].tolist(), "D": df.iloc[:, 1].tolist()}
        st.success(f"✅ Loaded data for {selected_dopant} from CSV")
    except:
        st.error("❌ Failed to read uploaded file.")

# Prepare data for training
X, y = [], []
for dopant in raw_data:
    T_vals = np.array(raw_data[dopant]['T']) + 273.15  # Convert to Kelvin
    D_vals = np.log10(np.array(raw_data[dopant]['D']))
    X.extend(1/T_vals)
    y.extend(D_vals)
X = np.array(X).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Define & train PINN
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='tanh', input_shape=(1,)),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_scaled, y, epochs=1000, verbose=0)

# Prediction
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

# Plotting
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['red', 'green', 'blue', 'purple', 'orange', 'black']
for i, dopant in enumerate(raw_data):
    T_lit = np.array(raw_data[dopant]['T']) + 273.15
    D_lit = np.array(raw_data[dopant]['D'])
    ax.scatter(T_lit, np.log10(D_lit), label=f"{dopant} (Lit)", color=colors[i])

ax.plot(T_predict, logD_pred, 'k--', label='PINN Prediction', linewidth=2)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("log10(Diffusivity [cm²/s])")
ax.set_title("PINN Prediction vs Literature Data")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Ea & D0 display
Ea_eV = Ea_fit * 8.617e-5  # eV
D0_val = 10**logD0_fit
st.success(f"**Arrhenius Fit:** D = {D0_val:.2e} * exp(-{Ea_eV:.3f} eV / kT)")

# Show predicted D for slider temperature
T_user = temp_input + 273.15
invT_user = 1 / T_user
D_user = 10 ** model.predict(scaler_X.transform([[invT_user]])).flatten()[0]
st.info(f"📍 At {temp_input}°C, predicted D ≈ {D_user:.2e} cm²/s")

# Download buttons
csv_data = pd.DataFrame({"T (K)": T_predict, "D (cm^2/s)": D_pred})
csv_file = csv_data.to_csv(index=False).encode()
st.download_button("⬇️ Download Prediction CSV", csv_file, file_name="pinn_diffusion.csv")

buf = BytesIO()
fig.savefig(buf, format="png")
st.download_button("⬇️ Download Plot as PNG", buf.getvalue(), file_name="diffusion_plot.png")
