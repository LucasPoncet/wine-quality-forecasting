import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Metrics from the reports
metrics = [
    "Accuracy",
    "F1 Score",
    "Precision Class 0",
    "Recall Class 0",
    "Precision Class 1",
    "Recall Class 1",
]
mlp_scores = [0.8181, 0.7716, 0.92, 0.79, 0.69, 0.87]
lgbm_scores = [0.7763, 0.7147, 0.87, 0.77, 0.65, 0.79]

# Create DataFrame for easier plotting
df = pd.DataFrame({"Metric": metrics, "MLP": mlp_scores, "LGBM": lgbm_scores})

# Plotting
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width / 2, df["MLP"], width, label="MLP")
bars2 = ax.bar(x + width / 2, df["LGBM"], width, label="LGBM")

ax.set_ylabel("Score")
ax.set_title("MLP vs LGBM – Test Metrics Comparison")
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=30)
ax.legend()
ax.set_ylim(0.5, 1.0)

plt.tight_layout()
plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.show()
