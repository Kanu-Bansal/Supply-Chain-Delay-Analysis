import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. DATA LOAD
orders = pd.read_csv("Supply Chain Delay Prediction\olist_orders_dataset.csv")
items = pd.read_csv("Supply Chain Delay Prediction\olist_order_items_dataset.csv")
customers = pd.read_csv("Supply Chain Delay Prediction\olist_customers_dataset.csv")
sellers = pd.read_csv("Supply Chain Delay Prediction\olist_sellers_dataset.csv")

# Understand Data
print("Item Information:",items.info())
print("Item Shape:",items.shape)
print("Item Describe:",items.describe())
print("Item Columns:",items.columns)
print("Orders Information:",orders.info())
print("Orders Shape:",orders.shape)
print("Orders Describe:",orders.describe())
print("Orders Columns:",orders.columns)
print("Customers Information:",customers.info())
print("Customers Shape:",customers.shape)
print("Customers Describe:",customers.describe())
print("Customers Columns:",customers.columns)
print("Sellers Information:",sellers.info())
print("Sellers Shape:",sellers.shape)
print("Sellers Describe:",sellers.describe())
print("Sellers Columns:",sellers.columns)

# 2. DATA MERGE
df = orders.merge(customers, on="customer_id", how="left")
df = df.merge(items, on="order_id", how="left")
df = df.merge(sellers, on="seller_id", how="left")
print("After Merge:", df.shape)

# 3. DATA CLEANING
# Remove undelivered orders
df = df[df["order_delivered_customer_date"].notna()]

# Remove missing sellers
df = df[df["seller_id"].notna()]

# Drop unnecessary columns
drop_cols = [
    "customer_zip_code_prefix",
    "seller_zip_code_prefix",
    "order_approved_at",
    "order_delivered_carrier_date",
    "shipping_limit_date",
    "product_id"
]
df = df.drop(columns=drop_cols)

# Convert dates
df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"])
df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"])

# 4. REMOVE DUPLICATES (ORDER LEVEL)
df = df.sort_values("order_purchase_timestamp") \
.drop_duplicates(subset="order_id", keep="first")
print("After Cleaning:", df.shape)

# 5. FEATURE ENGINEERING
# Delay Days
df["Delay_Days"] = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.days

# Delay Flag
df["Is_Delayed"] = df["Delay_Days"].apply(lambda x: 1 if x > 0 else 0)

# Delivery Time
df["Delivery_Time"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days

# Vendor Delay Rate
vendor_delay = df.groupby("seller_id")["Is_Delayed"].mean()
df["Vendor_Delay_Rate"] = df["seller_id"].map(vendor_delay)

# Risk Score
df["Risk_Score"] = (df["Delivery_Time"] * 0.3 +df["Vendor_Delay_Rate"] * 100 * 0.7)

# Risk Level
def risk_level(score):
    if score < 20:
        return "Low"
    elif score < 50:
        return "Medium"
    else:
        return "High"

df["Risk_Level"] = df["Risk_Score"].apply(risk_level)

# 6. BASIC ANALYSIS
print("Delayed vs On-time:", df["Is_Delayed"].value_counts())
print("Top States by Delay:",df.groupby("customer_state")["Delay_Days"].mean().sort_values(ascending=False).head(10))
print("Top Worst Vendors:",df.groupby("seller_id")["Delay_Days"].mean().sort_values(ascending=False).head(10))
print("Delay Statistics:", df["Delay_Days"].describe())

# 7. MODEL BUILDING
X = df[["Delivery_Time", "Vendor_Delay_Rate"]]
y = df["Is_Delayed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Model Trained Successfully!")

# 8. MODEL EVALUATION
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print("Classification Report:", classification_report(y_test, pred))

# 9. SAVE FINAL DATA (FOR POWER BI)
df.to_csv("final_supply_chain_data.csv", index=False)
print("Final dataset saved successfully!")
