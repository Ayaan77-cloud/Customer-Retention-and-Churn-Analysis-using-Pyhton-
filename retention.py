# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 2. LOAD DATA
# ==============================
df = pd.read_csv("customer_retention.csv")

# Convert to datetime
df['OrderDate'] = pd.to_datetime(df['OrderDate'])

# ==============================
# 3. DATA CLEANING
# ==============================
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# ==============================
# 4. RFM ANALYSIS
# ==============================
snapshot_date = df['OrderDate'].max()

rfm = df.groupby('CustomerID').agg({
    'OrderDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceID': 'count',
    'Amount': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

# ==============================
# 5. RFM SEGMENTATION
# ==============================
rfm['R_score'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1])
rfm['F_score'] = pd.qcut(rfm['Frequency'], 4, labels=[1,2,3,4])
rfm['M_score'] = pd.qcut(rfm['Monetary'], 4, labels=[1,2,3,4])

rfm['RFM_Score'] = rfm[['R_score','F_score','M_score']].sum(axis=1)

def segment(x):
    if x >= 10:
        return "High Value"
    elif x >= 6:
        return "Medium Value"
    else:
        return "Low Value"

rfm['Segment'] = rfm['RFM_Score'].apply(segment)

# ==============================
# 6. CHURN ANALYSIS
# ==============================
rfm['Churn'] = rfm['Recency'].apply(lambda x: 1 if x > 90 else 0)

# ==============================
# 7. VISUALIZATION (MATPLOTLIB ONLY)
# ==============================

# ------------------------------
# GRAPH 1: CUSTOMER SEGMENTS
# ------------------------------
segment_counts = rfm['Segment'].value_counts()

plt.figure()
plt.bar(segment_counts.index, segment_counts.values)
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")
plt.title("Customer Segmentation")

for i, v in enumerate(segment_counts.values):
    plt.text(i, v + 1, str(v), ha='center')

plt.show()


# ------------------------------
# GRAPH 2: CHURN ANALYSIS
# ------------------------------
churn_counts = rfm['Churn'].value_counts()

plt.figure()
plt.bar(['Active','Churned'], churn_counts.values)
plt.xlabel("Customer Status")
plt.ylabel("Number of Customers")
plt.title("Churn Analysis")

for i, v in enumerate(churn_counts.values):
    plt.text(i, v + 1, str(v), ha='center')

plt.show()


# ------------------------------
# GRAPH 3: CORRELATION HEATMAP (MATPLOTLIB)
# ------------------------------
corr = rfm[['Recency','Frequency','Monetary']].corr()

plt.figure()
plt.imshow(corr)

plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap")

# Add values inside heatmap
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j, i, round(corr.iloc[i, j], 2),
                 ha='center', va='center', color='white')

plt.show()


# ------------------------------
# GRAPH 4: COHORT ANALYSIS
# ------------------------------
df['OrderMonth'] = df['OrderDate'].dt.to_period('M')
df['CohortMonth'] = df.groupby('CustomerID')['OrderDate'].transform('min').dt.to_period('M')

cohort_data = df.groupby(['CohortMonth', 'OrderMonth']).agg({'CustomerID':'nunique'}).reset_index()

cohort_pivot = cohort_data.pivot(index='CohortMonth', columns='OrderMonth', values='CustomerID')

plt.figure()
plt.imshow(cohort_pivot.fillna(0))

plt.colorbar()
plt.title("Cohort Analysis")

plt.xticks(range(len(cohort_pivot.columns)), cohort_pivot.columns, rotation=45)
plt.yticks(range(len(cohort_pivot.index)), cohort_pivot.index)

plt.show()

