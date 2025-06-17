import pickle
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.metrics import adjusted_rand_score, silhouette_score

# --- Load saved clustering results ---
with open("./d2v_workflow/clustering_results.pkl", "rb") as f:
    data = pickle.load(f)
paths = data["paths"]
predicted_labels = data["predicted_labels"]
vectors = np.array(data["vectors"])

# --- Convert cluster names to integer IDs ---
label_map = {name: idx for idx, name in enumerate(sorted(set(predicted_labels)))}
label_ids = [label_map[name] for name in predicted_labels]

# --- Optional: Extract true labels from filenames ---
def extract_true_labels(paths):
    labels = []
    for path in paths:
        name = Path(path).stem
        true_label = name.split("_")[0]  # assumes format like "sports_doc1.txt"
        labels.append(true_label)
    return labels

# --- Purity Score ---
def purity_score(y_true, y_pred):
    contingency_matrix = {}
    for true, pred in zip(y_true, y_pred):
        contingency_matrix.setdefault(pred, []).append(true)
    total = sum(
        Counter(group).most_common(1)[0][1] for group in contingency_matrix.values()
    )
    return total / len(y_true)

# --- Evaluation ---
true_labels = extract_true_labels(paths)
purity = purity_score(true_labels, predicted_labels)
ari = adjusted_rand_score(true_labels, predicted_labels)
sil_score = silhouette_score(vectors, label_ids) if len(set(label_ids)) > 1 else -1

print(
    f"Silhouette Score: {sil_score:.4f}"
    if sil_score != -1
    else "Silhouette Score: Not applicable (only 1 cluster)"
)
print(f"Purity Score: {purity:.4f}")
print(f"Adjusted Rand Index: {ari:.4f}")

# --- UMAP Visualization ---
n_neighbors = min(15, max(2, len(vectors) - 1))  # Adaptive n_neighbors
min_dist = 0.1  # Controls how tightly UMAP packs points together

reducer = umap.UMAP(
    n_components=2, 
    n_neighbors=n_neighbors, 
    min_dist=min_dist,
    random_state=42,
    metric='cosine'  # Often works well for document vectors
)
reduced = reducer.fit_transform(vectors)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=label_ids, cmap="tab10", s=40)
plt.title("Document Clusters (UMAP Projection)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.colorbar(scatter, label="Cluster ID")
plt.grid(True)
plt.tight_layout()
plt.savefig("clusters_umap.png")
plt.show()