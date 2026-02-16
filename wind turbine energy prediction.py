# Wind Turbine Energy Prediction - Complete Pipeline
# This script contains the full ML pipeline from data loading to model saving

# Step 1: Importing Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import warnings
warnings.filterwarnings('ignore')

# Step 2: Load and Analyze the Dataset
path = "data/dataset/train.csv"
df = pd.read_csv(path)

# Rename columns for better understanding
df.rename(columns={
    'tracking_id': 'ID',
    'datetime': 'DateTime',
    'wind_speed(m/s)': 'WindSpeed',
    'atmospheric_temperature(°C)': 'AtmosphericTemp',
    'shaft_temperature(°C)': 'ShaftTemp',
    'blades_angle(°)': 'BladesAngle',
    'gearbox_temperature(°C)': 'GearboxTemp',
    'engine_temperature(°C)': 'EngineTemp',
    'motor_torque(N-m)': 'MotorTorque',
    'generator_temperature(°C)': 'GeneratorTemp',
    'atmospheric_pressure(Pascal)': 'AtmosphericPressure',
    'area_temperature(°C)': 'AreaTemp',
    'windmill_body_temperature(°C)': 'WindmillBodyTemp',
    'wind_direction(°)': 'WindDirection',
    'resistance(ohm)': 'Resistance',
    'rotor_torque(N-m)': 'RotorTorque',
    'turbine_status': 'TurbineStatus',
    'cloud_level': 'CloudLevel',
    'blade_length(m)': 'BladeLength',
    'blade_breadth(m)': 'BladeBreadth',
    'windmill_height(m)': 'WindmillHeight',
    'windmill_generated_power(kW/h)': 'Power_kWh'
}, inplace=True)

print(f"Dataset Shape: {df.shape}")
print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
print(df.head())

# Step 3: Data Description
print("\n" + "=" * 50)
print("DATASET INFO")
print("=" * 50)
df.info()

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

# Step 4: Correlation Analysis
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

print("\n" + "=" * 50)
print("Correlation with Power_kWh (Target):")
print("=" * 50)
power_corr = corr['Power_kWh'].drop('Power_kWh').sort_values(ascending=False)
for feature, value in power_corr.items():
    print(f"{feature:>25s} : {value:+.4f}")

# Heatmap
plt.figure(figsize=(16, 12))
ax = sns.heatmap(corr, vmin=-1, vmax=1, annot=True, fmt='.2f',
                 cmap='RdBu_r', linewidths=0.5, square=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title("Correlation Heatmap", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.show()

# Step 5: Splitting Data into Independent (X) and Dependent (y) Variables
df_clean = df[['WindSpeed', 'MotorTorque', 'RotorTorque', 'Power_kWh']].dropna()

y = df_clean['Power_kWh']
X = df_clean[['WindSpeed', 'MotorTorque', 'RotorTorque']]

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=0)

print(f"\nTraining samples: {len(train_X)}")
print(f"Testing samples : {len(val_X)}")

# Step 6: Model Building - Random Forest Regressor
forest_model = RandomForestRegressor(
    n_estimators=750,
    max_depth=4,
    max_leaf_nodes=500,
    random_state=1
)

print("\nTraining the Random Forest Regressor model...")
forest_model.fit(train_X, train_y)
print("Model trained successfully!")

# Step 7: Model Evaluation
power_preds = forest_model.predict(val_X)

mae = mean_absolute_error(val_y, power_preds)
r2 = r2_score(val_y, power_preds)

print("\n" + "=" * 50)
print("MODEL EVALUATION RESULTS")
print("=" * 50)
print(f"Mean Absolute Error (MAE) : {mae:.4f}")
print(f"R² Score                  : {r2:.4f}")

# Step 8: Save the Model
joblib.dump(forest_model, "power_prediction.sav")
print("\nModel saved as 'power_prediction.sav'")
