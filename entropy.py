import numpy as np
import pandas as pd

df = pd.read_csv(r"C:\Users\UserS2025\OneDrive - IBB Institut für Berufliche Bildung AG - Viona\DL_DeepLearning\Aufgabe\radfahren_orig.csv")
target = "Radfahren"
features = [col for col in df.columns if col != target]

def entropy(series):
    probs = series.value_counts(normalize=True)
    probs_nz = probs[probs > 0]
    return -np.sum(probs_nz * np.log2(probs_nz))

def gini(series):
    probs = series.value_counts(normalize=True)
    return 1 - np.sum(probs ** 2)

# Parent metrics (before any split)
parent_entropy = entropy(df[target])
parent_gini    = gini(df[target])
print(f"Parent entropy ({target}): {parent_entropy:.3f}")
print(f"Parent gini   ({target}): {parent_gini:.3f}")
print("-" * 55)

results = []
for feature in features:
    groups = df.groupby(feature)[target]

    weighted_entropy = sum(
        (len(group) / len(df)) * entropy(group)
        for _, group in groups
    )
    weighted_gini = sum(
        (len(group) / len(df)) * gini(group)
        for _, group in groups
    )

    info_gain      = parent_entropy - weighted_entropy
    gini_reduction = parent_gini    - weighted_gini

    results.append({
        "Feature":          feature,
        "Wtd Entropy":      round(weighted_entropy, 3),
        "Info Gain":        round(info_gain, 3),
        "Wtd Gini":         round(weighted_gini, 3),
        "Gini Reduction":   round(gini_reduction, 3),
    })

    print(f"Feature: {feature}")
    print(f"  Weighted entropy after split : {weighted_entropy:.3f}")
    print(f"  Information gain             : {info_gain:.3f}")
    print(f"  Weighted gini after split    : {weighted_gini:.3f}")
    print(f"  Gini reduction               : {gini_reduction:.3f}")
    print()

# Summary table sorted by information gain
results_df = pd.DataFrame(results).sort_values("Info Gain", ascending=False)
print("-" * 55)
print("Summary (sorted by Information Gain):")
print(results_df.to_string(index=False))
print()

# Best feature by each metric
best_entropy = results_df.iloc[0]["Feature"]
best_gini    = results_df.sort_values("Gini Reduction", ascending=False).iloc[0]["Feature"]
print(f"Best split by Information Gain : {best_entropy}")
print(f"Best split by Gini Reduction   : {best_gini}")