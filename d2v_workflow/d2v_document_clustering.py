import json
import os
import pickle
import shutil
from collections import Counter
from pathlib import Path

import nltk
import numpy as np
from docx import Document
from gensim.models import Doc2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pdfminer.high_level import extract_text as extract_pdf_text
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")
nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))

# ======= CONFIG =======
MODEL_PATH = "./d2v_workflow/doc2vec_agnews.bin"
CLUSTER_STATE_PATH = "./d2v_workflow/cluster_state.json"
DOCUMENTS_DIR = "documents"
CLUSTERED_OUTPUT = "d2v_clusters"
SIMILARITY_THRESHOLD = 0.75
PICKLE_OUTPUT = "./d2v_workflow/clustering_results.pkl"
# ======================

# Persistent token storage for each cluster
cluster_tokens = {}


def load_clusters(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        return {name: np.array(vec) for name, vec in data.items()}
    return {}


def save_clusters(cluster_data, path):
    with open(path, "w") as f:
        json.dump({k: v.tolist() for k, v in cluster_data.items()}, f)


def extract_text(file_path):
    if file_path.suffix.lower() == ".pdf":
        return extract_pdf_text(str(file_path))
    elif file_path.suffix.lower() == ".docx":
        doc = Document(str(file_path))
        return "\n".join([para.text for para in doc.paragraphs])
    elif file_path.suffix.lower() == ".txt":
        return file_path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha() and t not in STOP_WORDS]


def assign_cluster(vector, tokens, cluster_centroids):
    if not cluster_centroids:
        name = "Cluster_1"
        cluster_centroids[name] = vector
        cluster_tokens[name] = tokens
        return name

    similarities = {
        name: cosine_similarity([vector], [centroid])[0][0]
        for name, centroid in cluster_centroids.items()
    }

    best_match = max(similarities, key=similarities.get)
    if similarities[best_match] >= SIMILARITY_THRESHOLD:
        cluster_tokens.setdefault(best_match, []).extend(tokens)
        return best_match
    else:
        new_name = f"Cluster_{len(cluster_centroids) + 1}"
        cluster_centroids[new_name] = vector
        cluster_tokens[new_name] = tokens
        return new_name


def rename_clusters(cluster_tokens):
    renamed = {}
    for cluster_name, tokens in cluster_tokens.items():
        counter = Counter(t for t in tokens if len(t) > 3)
        top_words = [word for word, _ in counter.most_common(3)]
        readable_name = "_".join(top_words).title() if top_words else cluster_name
        renamed[cluster_name] = readable_name
    return renamed


def copy_to_cluster(file_path, cluster_name, name_map):
    true_name = name_map.get(cluster_name, cluster_name)
    dest_folder = Path(CLUSTERED_OUTPUT) / true_name
    dest_folder.mkdir(parents=True, exist_ok=True)
    dest_file = dest_folder / file_path.name
    shutil.copy2(file_path, dest_file)


def save_for_evaluation(file_cluster_map, renamed_clusters, file_vectors):
    data_to_save = {
        "paths": [str(p) for p in file_cluster_map.keys()],
        "predicted_labels": [
            renamed_clusters.get(c, c) for c in file_cluster_map.values()
        ],
        "vectors": file_vectors,
    }
    with open(PICKLE_OUTPUT, "wb") as f:
        pickle.dump(data_to_save, f)
    print(f"📦 Saved evaluation data to {PICKLE_OUTPUT}")


def process_documents():
    model = Doc2Vec.load(MODEL_PATH)
    cluster_centroids = load_clusters(CLUSTER_STATE_PATH)
    cluster_tokens.clear()
    file_cluster_map = {}
    file_vectors = []

    for file_path in sorted(Path(DOCUMENTS_DIR).glob("*")):
        print(f"\nProcessing: {file_path.name}")
        try:
            text = extract_text(file_path)
            tokens = preprocess_text(text)
            if not tokens:
                print("No valid content found. Skipping.")
                continue

            vector = model.infer_vector(tokens, epochs=20)
            cluster_name = assign_cluster(vector, tokens, cluster_centroids)
            file_cluster_map[file_path] = cluster_name
            file_vectors.append(vector)
            print(f"Assigned to temporary cluster: {cluster_name}")
        except Exception as e:
            print(f"Failed to process {file_path.name}: {e}")

    # Rename clusters for interpretability
    renamed_clusters = rename_clusters(cluster_tokens)

    # Copy files into cluster folders
    for file_path, old_cluster in file_cluster_map.items():
        copy_to_cluster(file_path, old_cluster, renamed_clusters)

    # Save cluster state and evaluation data
    save_clusters(cluster_centroids, CLUSTER_STATE_PATH)
    save_for_evaluation(file_cluster_map, renamed_clusters, file_vectors)

    print("\nAll documents processed, clustered, and saved.")
    print(f"Original files remain in {DOCUMENTS_DIR}")
    print(f"Clustered copies created in {CLUSTERED_OUTPUT}")


# Entry point
if __name__ == "__main__":
    Path(CLUSTERED_OUTPUT).mkdir(exist_ok=True)
    Path(DOCUMENTS_DIR).mkdir(exist_ok=True)
    process_documents()
