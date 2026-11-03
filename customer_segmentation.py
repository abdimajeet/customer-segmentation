"""3. Customer Segmentation: K-Means clustering on the UCI Wholesale Customers dataset.

Problem:  group a wholesale distributor's customers by what they buy, with no labels.
Dataset:  Wholesale Customers (UCI), 440 clients in Portugal, annual spending in 6
          product categories, plus their channel (hotel/restaurant/café or retail).
Model:    K-Means. k is chosen with the elbow method and the silhouette score.

Run:  python customer_segmentation.py
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from common import dataset, r, save_result

SPEND = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
NICE = {"Fresh": "fresh food", "Milk": "milk", "Grocery": "grocery", "Frozen": "frozen food",
        "Detergents_Paper": "detergents & paper", "Delicassen": "delicatessen"}

# ---- load ----
df = pd.read_csv(dataset("wholesale") / "Wholesale customers data.csv")
df["channel_name"] = df.Channel.map({1: "HoReCa", 2: "Retail"})

# ---- EDA ----
skew = df[SPEND].skew().sort_values(ascending=False)
corr = df[SPEND].corr()
pair = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().idxmax()

# ---- preprocessing ----
# Spending is heavily right-skewed (a few huge clients), so take log(1 + x), then standardise.
X = StandardScaler().fit_transform(np.log1p(df[SPEND]))

# ---- choose k ----
ks = list(range(1, 11))
inertia = [KMeans(k, n_init=10, random_state=42).fit(X).inertia_ for k in ks]
sil = {k: silhouette_score(X, KMeans(k, n_init=10, random_state=42).fit_predict(X)) for k in range(2, 9)}
best_k = max(sil, key=sil.get)

# ---- final model ----
km = KMeans(best_k, n_init=10, random_state=42).fit(X)
df["segment"] = km.labels_
profile = df.groupby("segment")[SPEND].median()
retail_share = df.groupby("segment").Channel.apply(lambda c: (c == 2).mean())
sizes = df.segment.value_counts().sort_index()

def describe(s):
    top = profile.loc[s].sort_values(ascending=False).index[:2]
    return f"segment {s + 1} ({sizes[s]} customers, {retail_share[s]:.0%} retail) spends most on {NICE[top[0]]} and {NICE[top[1]]}"

agree = max((df.segment == df.Channel - 1).mean(), (df.segment == 2 - df.Channel).mean()) if best_k == 2 else None

print(f"skew: {skew.round(1).to_dict()}")
print(f"silhouette by k: { {k: round(v, 3) for k, v in sil.items()} } -> k = {best_k}")
print(profile.round(0))

save_result("customer-segments", {
    "cfg": {"model": "K-Means", "dataset": f"Wholesale Customers, {len(df)} clients",
            "features": "6 spending categories, log + scaled", "k": f"{best_k} (best silhouette)", "init": "k-means++, 10 restarts"},
    "metric": {"label": "Silhouette score", "value": f"{sil[best_k]:.3f}"},
    "metrics": [{"label": "Segments", "value": str(best_k)}],
    "curve": {"xLabel": "number of clusters k", "yLabel": "inertia (within-cluster spread)", "x": ks,
              "train": r(inertia, 1), "val": None, "trainLabel": "inertia (elbow method)", "valLabel": None},
    "findings": [
        f"Spending is very skewed (skewness up to {skew.iloc[0]:.1f} for {NICE[skew.index[0]]}), so I log-transformed it before clustering. That way a few very large clients don't dominate the distances.",
        f"{NICE[pair[0]].capitalize()} and {NICE[pair[1]]} spending are strongly linked (correlation {corr.loc[pair]:.2f}): customers who buy one buy the other.",
        f"The elbow chart has no sharp bend, so I chose k with the silhouette score instead: it is highest at k = {best_k} ({sil[best_k]:.2f}).",
        *[describe(s)[0].upper() + describe(s)[1:] + "." for s in range(best_k)],
        *([f"Without ever seeing the channel label, the clusters match hotel/restaurant vs retail for {agree:.0%} of customers."] if agree else []),
    ],
})
