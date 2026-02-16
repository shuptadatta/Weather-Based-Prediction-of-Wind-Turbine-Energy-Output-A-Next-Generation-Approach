# Test the saved Wind Mill Power Prediction model
# This script loads the saved model and runs test predictions

import joblib
import pandas as pd

# Load the saved model
model = joblib.load("power_prediction.sav")
print("Model loaded successfully!")
print(f"Model type: {type(model).__name__}")
print(f"Number of estimators: {model.n_estimators}")
print(f"Max depth: {model.max_depth}")

# Test predictions with sample inputs
# Features: [WindSpeed (m/s), MotorTorque (N-m), RotorTorque (N-m)]
test_cases = [
    {"WindSpeed": 94.82, "MotorTorque": 2563.12, "RotorTorque": 42.08},
    {"WindSpeed": 241.83, "MotorTorque": 2372.38, "RotorTorque": 107.89},
    {"WindSpeed": 10.72, "MotorTorque": 781.70, "RotorTorque": 13.39},
    {"WindSpeed": 50.00, "MotorTorque": 1500.00, "RotorTorque": 60.00},
    {"WindSpeed": 300.00, "MotorTorque": 2800.00, "RotorTorque": 120.00},
]

print("\n" + "=" * 65)
print(f"{'WindSpeed':>12} {'MotorTorque':>14} {'RotorTorque':>14} {'Predicted kWh':>16}")
print("=" * 65)

for case in test_cases:
    x_test = pd.DataFrame([[case["WindSpeed"], case["MotorTorque"], case["RotorTorque"]]],
                          columns=["WindSpeed", "MotorTorque", "RotorTorque"])
    prediction = model.predict(x_test)[0]
    print(f"{case['WindSpeed']:>12.2f} {case['MotorTorque']:>14.2f} {case['RotorTorque']:>14.2f} {prediction:>16.4f}")

print("=" * 65)
print("\nAll test predictions completed successfully!")
