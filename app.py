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

st.set_page_config(page_title="ML-Enhanced Diffusion Modeling", layout="wide")
st.title("🧬 DiffuLab: ML-Enhanced Diffusion Modeling")
st.caption("Powered by Physics-Informed Neural Networks • Built with ❤️ in Streamlit")

# Sidebar user input
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Silicon_Structure.svg/2560px-Silicon_Structure.svg.png", use_column_width=True)
    st.subheader("🔧 Choose Settings")

    selected_dopant = st.selectbox("🧪 Dopant", [
        "Phosphorus", "Boron", "Arsenic", "Antimony", "Gallium", "Nitrogen"])

    temp_input = st.slider("🌡️ Temperature (°C)", 600, 1300, 1000)

    st.subheader("🔍 Compare Two Dopants")
    dopant_1 = st.selectbox("Select Dopant 1", list(raw_data.keys()), index=0, key="dopant1")
    dopant_2 = st.selectbox("Select Dopant 2", list(raw_data.keys()), index=1, key="dopant2")

    st.markdown("---")
    st.subheader("📁 Upload New Data")
    uploaded_file = st.file_uploader("Upload Dopant CSV (T, D)", type=["csv"])

    st.markdown("---")
    st.caption("🔬 Created by Rogelio Lopez")


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

# Compute R² score
y_pred_for_r2 = model.predict(invT_scaled)
r2 = r2_score(logD_fit, y_pred_for_r2)
st.metric("📊 R² Score of Model", f"{r2:.4f}")

# Plotting
# Enhanced Plotting
plt.style.use("seaborn-v0_8-darkgrid")
fig, ax = plt.subplots(figsize=(12, 6))

dopant_colors = {
    "Phosphorus": "red",
    "Boron": "blue",
    "Arsenic": "green",
    "Antimony": "orange",
    "Gallium": "purple",
    "Nitrogen": "brown"
}

fig, ax = plt.subplots(figsize=(12, 6))

for dopant in [dopant_1, dopant_2]:
    T_vals = np.array(raw_data[dopant]['T']) + 273.15
    D_vals = np.array(raw_data[dopant]['D'])
    label = f"{dopant} (Literature)"
    color = dopant_colors[dopant]
    ax.scatter(T_vals, np.log10(D_vals), label=label, color=color, s=60, edgecolors='k', alpha=0.8)

# Plot PINN prediction
if st.session_state.plot_scale == "Linear D":
    ax.plot(T_predict, D_pred, 'k--', label='PINN Prediction', linewidth=2)
    ax.set_ylabel("Diffusivity [cm²/s]", fontsize=12)
    ax.set_yscale("linear")
    ax.set_ylim(1e-20, 1e-12)
else:
    ax.plot(T_predict, logD_pred, 'k--', label='PINN Prediction', linewidth=2)
    ax.set_ylabel("log₁₀(Diffusivity [cm²/s])", fontsize=12)
    ax.set_yscale("linear")

if st.session_state.temp_unit == "Celsius (°C)":
    ax.set_xlabel("Temperature (°C)", fontsize=12)
    ticks = np.linspace(600, 1300, 8)
    ax.set_xticks(ticks + 273.15)
    ax.set_xticklabels([f"{int(t)}" for t in ticks])
else:
    ax.set_xlabel("Temperature (K)", fontsize=12)

arrhenius_eq = f"log₁₀(D) = {logD0_fit:.2f} - {Ea_fit:.2f}/kT"
ax.text(0.05, 0.92, arrhenius_eq, transform=ax.transAxes, fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8))

ax.set_title("📈 PINN Prediction vs Literature Data", fontsize=14)
ax.legend()
ax.grid(True)
st.pyplot(fig)


ax.set_xlabel("Temperature (K)", fontsize=12)
ax.set_ylabel(r"$\log_{10}$(Diffusivity) [cm²/s]", fontsize=12)
ax.set_title("📊 PINN Prediction vs Literature Data", fontsize=14, weight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, linestyle='--', alpha=0.6)
st.pyplot(fig)
st.caption("ℹ️ **Note:** The PINN prediction (dashed line) is based on a global model trained using all dopants. It does not change when switching dopants.")


# Ea & D0 display
Ea_eV = Ea_fit * 8.617e-5  # eV
D0_val = 10**logD0_fit
st.success(f"**Arrhenius Fit:** D = {D0_val:.2e} * exp(-{Ea_eV:.3f} eV / kT)")
st.caption("This classification is based on the pre-exponential factor D₀.")

# Diffusivity category label
if D0_val > 1e-13:
    category = "🟥 High Diffusivity"
elif D0_val > 1e-17:
    category = "🟨 Moderate Diffusivity"
else:
    category = "🟦 Low Diffusivity"

st.markdown(f"### 🧠 Diffusivity Category: {category}")

# Optional: download Arrhenius params
arrhenius_df = pd.DataFrame({
    "Dopant": [selected_dopant],
    "D0 (cm²/s)": [D0_val],
    "Ea (eV)": [Ea_eV]
})
arrhenius_csv = arrhenius_df.to_csv(index=False).encode()
st.download_button("⬇️ Download Arrhenius Parameters", arrhenius_csv, file_name=f"{selected_dopant}_arrhenius.csv")


# Show predicted D for slider temperature
T_user = temp_input + 273.15
invT_user = 1 / T_user
D_user = 10 ** model.predict(scaler_X.transform([[invT_user]])).flatten()[0]
st.info(f"📍 At {temp_input}°C, predicted D ≈ {D_user:.2e} cm²/s")
st.caption("Want to know what temperature gives you a specific diffusivity? Use the tool below! 👇")

with st.expander("🔁 Inverse Prediction: Find Temperature from D"):
    logD_input = st.number_input("Enter log₁₀(Diffusivity)", value=-16.0)
    predicted_invX = model.predict(np.array([[logD_input]]))
    T_estimate_K = 1 / scaler_X.inverse_transform(predicted_invX)[0][0]
    T_estimate_C = T_estimate_K - 273.15
    st.write(f"Estimated Temperature: {T_estimate_C:.2f} °C")
    


# Download buttons
csv_data = pd.DataFrame({"T (K)": T_predict, "D (cm^2/s)": D_pred})
csv_file = csv_data.to_csv(index=False).encode()
st.download_button("⬇️ Download Prediction CSV", csv_file, file_name="pinn_diffusion.csv")

buf = BytesIO()
fig.savefig(buf, format="png")
st.download_button("⬇️ Download Plot as PNG", buf.getvalue(), file_name="diffusion_plot.png")
st.markdown("### 💾 Session Save & Load")

# Save session
if st.button("💾 Save Session to file"):
    session_dict = {
        "dopant": selected_dopant,
        "temperature_C": temp_input,
        "D_pred": D_pred.tolist(),
        "T_K": T_predict.tolist()
    }
    session_df = pd.DataFrame(session_dict)
    session_csv = session_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download Session CSV", session_csv, file_name="session_data.csv")

# Load session
uploaded_session = st.file_uploader("📤 Upload Session CSV", type=["csv"], key="session")

if uploaded_session is not None:
    try:
        loaded_df = pd.read_csv(uploaded_session)
        st.success("✅ Session loaded successfully!")

        # Plot loaded session
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(loaded_df["T_K"], np.log10(loaded_df["D_pred"]), 'b-', label="Loaded Session")
        ax2.set_xlabel("Temperature (K)")
        ax2.set_ylabel("log10(Diffusivity [cm²/s])")
        ax2.set_title("Loaded Session Prediction")
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"❌ Failed to load session: {e}")

# -- EXPORT .xyz FORMAT FOR LAMMPS/OVITO --

st.markdown("### 🧬 Export for LAMMPS / OVITO")

xyz_export = st.button("📤 Generate .xyz File")

if xyz_export:
    num_atoms = 100  # You can adjust this
    z_positions = np.linspace(0, 100, num_atoms)  # arbitrary length scale (Angstroms)
    concentrations = D_pred[:num_atoms] / np.max(D_pred)  # normalize for illustration

    # Create .xyz content
    xyz_lines = [f"{num_atoms}", "Dopant profile exported from PINN"]
    for i in range(num_atoms):
        xyz_lines.append(f"{selected_dopant[0]} 0.0 0.0 {z_positions[i]:.3f} {concentrations[i]:.4f}")

    xyz_text = "\n".join(xyz_lines)
    st.code(xyz_text, language="xyz")

    # Download
    xyz_bytes = xyz_text.encode()
    st.download_button("⬇️ Download .xyz for OVITO", xyz_bytes, file_name=f"{selected_dopant}_profile.xyz")

# -- SESSION HISTORY MANAGEMENT --

# Save to session
if st.button("💾 Save Session", key="save_session"):
    st.session_state.history.append({
        "dopant": selected_dopant,
        "temperature_C": temp_input,
        "D_predicted": D_user,
        "Ea (eV)": Ea_eV,
        "D0 (cm²/s)": D0_val
    })
    st.success("✅ Session saved!")

# View previous saved sessions
if st.session_state.history:
    st.markdown("### 🧾 Previous Sessions")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)

    # Download session history
    hist_csv = history_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download Session History", hist_csv, file_name="session_history.csv")

