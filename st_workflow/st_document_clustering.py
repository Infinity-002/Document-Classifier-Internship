import re
import shutil
from collections import Counter
from pathlib import Path

import nltk
import numpy as np
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from hdbscan import HDBSCAN
from pdfminer.high_level import extract_text as extract_pdf_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from umap import UMAP

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


class DocumentClustering:
    SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".html", ".htm"}

    def __init__(self, input_folder, output_folder, model_name="all-MiniLM-L6-v2"):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)

        # Try different models based on use case
        self.model = SentenceTransformer(model_name)

        self.file_paths = [
            p
            for p in self.input_folder.iterdir()
            if p.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]
        self.embeddings = []  # Keep as list initially
        self.raw_texts = []
        self.processed_texts = []
        self.labels = []
        self.document_metadata = []

    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing"""
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove special characters but keep punctuation for sentence structure
        text = re.sub(r"[^\w\s.,!?;:-]", "", text)
        # Remove very short lines that are likely noise
        lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 20]
        return "\n".join(lines)

    def extract_text(self, path: Path) -> tuple[str, dict]:
        """Extract text and metadata"""
        metadata = {
            "filename": path.name,
            "size": path.stat().st_size,
            "extension": path.suffix.lower(),
        }

        try:
            suffix = path.suffix.lower()
            if suffix == ".txt":
                text = path.read_text(encoding="utf-8")
            elif suffix == ".pdf":
                text = extract_pdf_text(str(path))
            elif suffix == ".docx":
                doc = DocxDocument(path)
                text = "\n".join(p.text for p in doc.paragraphs)
            elif suffix in {".html", ".htm"}:
                html = path.read_text(encoding="utf-8")
                soup = BeautifulSoup(html, "lxml")
                text = soup.get_text()

            # Add basic text statistics to metadata
            if text:
                metadata.update(
                    {
                        "word_count": len(text.split()),
                        "char_count": len(text),
                        "sentence_count": len(nltk.sent_tokenize(text)),
                    }
                )

            return text, metadata
        except Exception as e:
            print(f"Error reading {path.name}: {e}")
            return "", metadata

    def chunk_document(
        self, text: str, chunk_size: int = 512, overlap: int = 50
    ) -> list[str]:
        """Split document into overlapping chunks for better representation"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            if len(chunk.split()) > 20:  # Only keep substantial chunks
                chunks.append(chunk)

        return chunks if chunks else [text]

    def embed_documents(self):
        """Enhanced document embedding with multiple strategies"""
        print("Embedding documents...")

        for path in self.file_paths:
            text, metadata = self.extract_text(path)
            if not text.strip():
                print(f"No usable text in: {path.name}")
                continue

            processed_text = self.preprocess_text(text)
            if len(processed_text.split()) < 10:
                print(f"Skipping too short document: {path.name}")
                continue

            # Include filename in the text for context
            full_text = (
                f"{path.stem.replace('_', ' ').replace('-', ' ')}\n\n{processed_text}"
            )

            chunks = self.chunk_document(full_text)
            chunk_embeddings = self.model.encode(chunks, convert_to_tensor=False)

            weights = np.exp(-0.1 * np.arange(len(chunks)))
            weights = weights / np.sum(weights)
            embedding = np.average(chunk_embeddings, axis=0, weights=weights)

            self.embeddings.append(embedding)
            self.raw_texts.append(text)
            self.processed_texts.append(processed_text)
            self.document_metadata.append(metadata)

        if self.embeddings:
            self.embeddings = np.array(self.embeddings)
            print(f"Generated embeddings for {len(self.embeddings)} documents")
        else:
            print("No embeddings generated")
            self.embeddings = np.array([])

    def reduce_dimensionality(self, n_components=50):
        """Reduce dimensionality before clustering"""
        if len(self.embeddings) <= n_components:
            return self.embeddings

        print("🔄 Reducing dimensionality with UMAP")

        reducer = UMAP(
            n_components=n_components,
            random_state=42,
            n_neighbors=min(15, len(self.embeddings) - 1),
        )

        reduced_embeddings = reducer.fit_transform(self.embeddings)
        return reduced_embeddings

    def evaluate_clustering(self, embeddings, labels):
        """Comprehensive clustering evaluation"""
        if len(set(labels)) <= 1:
            return {
                "silhouette": -1,
                "calinski_harabasz": 0,
                "davies_bouldin": float("inf"),
            }

        try:
            # Filter out noise points for evaluation
            mask = labels != -1
            if np.sum(mask) <= 1:
                return {
                    "silhouette": -1,
                    "calinski_harabasz": 0,
                    "davies_bouldin": float("inf"),
                }

            filtered_embeddings = embeddings[mask]
            filtered_labels = labels[mask]

            silhouette = silhouette_score(filtered_embeddings, filtered_labels)
            calinski = calinski_harabasz_score(filtered_embeddings, filtered_labels)
            davies_bouldin = davies_bouldin_score(filtered_embeddings, filtered_labels)

            return {
                "silhouette": silhouette,
                "calinski_harabasz": calinski,
                "davies_bouldin": davies_bouldin,
            }
        except:
            return {
                "silhouette": -1,
                "calinski_harabasz": 0,
                "davies_bouldin": float("inf"),
            }

    def cluster_with_multiple_methods(self, embeddings):
        """Try multiple clustering methods and select the best"""
        methods = {}

        # HDBSCAN with different parameters
        for min_cluster_size in [2, 3, max(2, len(embeddings) // 10)]:
            try:
                clusterer = HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=max(1, min_cluster_size - 1),
                    metric="euclidean",
                    cluster_selection_epsilon=0.1,
                    prediction_data=True,
                )
                labels = clusterer.fit_predict(embeddings)
                score = self.evaluate_clustering(embeddings, labels)
                methods[f"hdbscan_mcs{min_cluster_size}"] = {
                    "labels": labels,
                    "score": score,
                    "clusterer": clusterer,
                }
            except Exception as e:
                print(f"HDBSCAN with min_cluster_size={min_cluster_size} failed: {e}")

        # Select best method parameters on silhouette score
        best_method = max(methods.items(), key=lambda x: x[1]["score"]["silhouette"])

        print(
            f"HDBSCAN with silhouette score: {best_method[1]['score']['silhouette']:.4f}"
        )

        return best_method[1]["labels"], best_method[1]["clusterer"]

    def generate_cluster_names(self, cluster_labels):
        """Generate names for clusters"""
        cluster_names = {}
        cluster_texts = {}

        # Group texts by cluster
        for text, label in zip(self.processed_texts, cluster_labels):
            if label == -1:
                continue
            cluster_texts.setdefault(label, []).append(text)

        for label in cluster_texts.keys():
            cluster_names[label] = f"cluster_{label}"

        return cluster_names

    def cluster_and_save(self):
        """Save documents to cluster folders with meaningful names"""
        cluster_names = self.generate_cluster_names(self.labels)
        cluster_stats = Counter(self.labels)

        print("\nCreating cluster folders...")
        for i, (label, filename) in enumerate(
            zip(self.labels, [meta["filename"] for meta in self.document_metadata])
        ):
            if label == -1:
                folder_name = "unclustered"
            else:
                cluster_name = cluster_names.get(label, f"cluster_{label}")
                folder_name = f"{cluster_name}_({cluster_stats[label]}_docs)"

            target_folder = self.output_folder / folder_name
            target_folder.mkdir(parents=True, exist_ok=True)

            source_file = self.input_folder / filename
            if source_file.exists():
                shutil.copy(source_file, target_folder / filename)

        print(f"Documents organized into {len(set(self.labels))} clusters")

    def print_cluster_summary(self):
        """Print detailed cluster summary"""
        cluster_names = self.generate_cluster_names(self.labels)
        cluster_stats = Counter(self.labels)

        print("\nCluster Summary:")
        print("=" * 50)

        for label in sorted(set(self.labels)):
            if label == -1:
                print(f"🔸 Unclustered: {cluster_stats[label]} documents")
                continue

            cluster_name = cluster_names.get(label, f"cluster_{label}")
            print(f"🔹 {cluster_name}: {cluster_stats[label]} documents")

            # Show sample documents
            cluster_docs = [
                meta["filename"]
                for meta, l in zip(self.document_metadata, self.labels)
                if l == label
            ]
            sample_docs = cluster_docs[:3]
            for doc in sample_docs:
                print(f"   - {doc}")
            if len(cluster_docs) > 3:
                print(f"   ... and {len(cluster_docs) - 3} more")
            print()

    def run(
        self,
        use_dimensionality_reduction=True,
        dim_reduction_method="umap",
    ):
        """Enhanced run method with more options"""
        print("Starting document clustering pipeline...")

        # Step 1: Embed documents
        self.embed_documents()

        if len(self.embeddings) == 0:
            print("No embeddings generated. Check file content.")
            return

        embeddings_for_clustering = self.embeddings
        if use_dimensionality_reduction and len(self.embeddings) > 50:
            embeddings_for_clustering = self.reduce_dimensionality(
                method=dim_reduction_method,
                n_components=min(50, len(self.embeddings) // 2),
            )

        print("Clustering...")
        self.labels, self.best_clusterer = self.cluster_with_multiple_methods(
            embeddings_for_clustering
        )

        self.cluster_and_save()
        self.print_cluster_summary()

        print("Clustering pipeline completed!")
