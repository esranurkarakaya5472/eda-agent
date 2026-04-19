"""Generates a realistic test CSV for EDA Agent demo."""
import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

df = pd.DataFrame({
    "customer_id":    [f"C{i:04d}" for i in range(n)],
    "age":             np.random.randint(18, 72, n).astype(float),
    "salary":          np.random.exponential(scale=6000, size=n).round(2),
    "tenure_months":   np.random.randint(1, 120, n).astype(float),
    "num_products":    np.random.randint(1, 5, n).astype(float),
    "credit_score":    np.random.normal(loc=650, scale=80, size=n).clip(300, 850).round(0),
    "account_balance": np.random.exponential(scale=15000, size=n).round(2),
    "city":            np.random.choice(["Istanbul", "Ankara", "Izmir", "Bursa", "Antalya"], n,
                                        p=[0.40, 0.25, 0.15, 0.10, 0.10]),
    "gender":          np.random.choice(["Male", "Female"], n),
    "has_credit_card": np.random.choice([0, 1], n, p=[0.35, 0.65]),
    "is_active":       np.random.choice([0, 1], n, p=[0.20, 0.80]),
    "churn":           np.random.choice([0, 1], n, p=[0.75, 0.25]),
})

# Aşırı değerler (outlier) ekle
df.loc[0,  "salary"]          = 999_999.0
df.loc[1,  "account_balance"] = 2_500_000.0
df.loc[2,  "credit_score"]    = 5.0

# Eksik değerler ekle (%10-15 arası)
for col, rate in [("age", 0.08), ("salary", 0.12), ("city", 0.10), ("credit_score", 0.06)]:
    mask = np.random.choice([True, False], n, p=[rate, 1 - rate])
    df.loc[mask, col] = np.nan

# Sabit/faydasız bir sütun (low variance test)
df["constant_col"] = 1

df.to_csv("data/sample_customers.csv", index=False)
print(f"✅ data/sample_customers.csv oluşturuldu — {n} satır, {df.shape[1]} sütun")
print(f"   Eksik değer olan sütunlar: {df.columns[df.isna().any()].tolist()}")
print(f"   Hedef sütun: churn")
