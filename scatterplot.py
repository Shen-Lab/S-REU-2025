import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("predicted_outputs.csv")

# Drop attempts with missing or zero predictions
df = df[df["PredictedLen"] > 0]

# Extract values
attempts = df["Attempt"]
predicted_lengths = df["PredictedLen"]
expected_length = df["ExpectedLen"].iloc[0]  # Assume same across all attempts
avg_length = predicted_lengths.mean()

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(attempts, predicted_lengths, color='blue', label='Predicted Length')
plt.axhline(y=avg_length, color='red', linestyle='--', label=f'Avg Predicted = {avg_length:.2f}')
plt.axhline(y=expected_length, color='green', linestyle='--', label=f'Expected = {expected_length}')

# Labels and formatting
plt.title("Predicted Amino Acid Lengths vs Attempts")
plt.xlabel("Attempts")
plt.ylabel("Predicted Sequence Length")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
