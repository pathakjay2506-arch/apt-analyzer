# Run this in your terminal (with venv active) to generate a test CSV
# Save this as make_test_data.py then run: python make_test_data.py
import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

# Simulate network flow features
data = {
    "duration":     np.random.exponential(1, n),
    "src_bytes":    np.random.exponential(5000, n),
    "dst_bytes":    np.random.exponential(3000, n),
    "src_pkts":     np.random.randint(1, 100, n),
    "dst_pkts":     np.random.randint(1, 100, n),
    "flow_iat_mean":np.random.exponential(50, n),
    "flow_iat_std": np.random.exponential(20, n),
    "pkt_len_mean": np.random.exponential(500, n),
}

# 850 normal, 80 DoS, 40 PortScan, 20 Bot, 10 Infiltration
labels = (
    ["BENIGN"] * 850 +
    ["DoS Hulk"] * 80 +
    ["PortScan"] * 40 +
    ["Bot"] * 20 +
    ["Infilteration"] * 10
)
np.random.shuffle(labels)
data["Label"] = labels

df = pd.DataFrame(data)
df.to_csv("test_traffic.csv", index=False)
print(f"Created test_traffic.csv with {len(df)} rows")