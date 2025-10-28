import pickle
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP

warnings.filterwarnings("ignore")


class ClusteringEvaluationDashboard:
    def __init__(self, embeddings, labels, document_metadata, raw_texts):
        self.embeddings = embeddings
        self.labels = labels
        self.document_metadata = document_metadata
        self.raw_texts = raw_texts
        self.n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    def compute_all_metrics(self):
        """Compute comprehensive clustering metrics"""
        metrics = {}

        # Filter out noise points for most metrics
        mask = self.labels != -1
        if np.sum(mask) <= 1:
            return {"error": "Insufficient non-noise points for evaluation"}

        filtered_embeddings = self.embeddings[mask]
        filtered_labels = self.labels[mask]

        # Internal validation metrics
        try:
            metrics["silhouette_score"] = silhouette_score(
                filtered_embeddings, filtered_labels
            )
        except:
            metrics["silhouette_score"] = -1

        try:
            metrics["calinski_harabasz_score"] = calinski_harabasz_score(
                filtered_embeddings, filtered_labels
            )
        except:
            metrics["calinski_harabasz_score"] = 0

        try:
            metrics["davies_bouldin_score"] = davies_bouldin_score(
                filtered_embeddings, filtered_labels
            )
        except:
            metrics["davies_bouldin_score"] = float("inf")

        # Cluster-specific metrics
        metrics["n_clusters"] = self.n_clusters
        metrics["n_noise_points"] = np.sum(self.labels == -1)
        metrics["noise_ratio"] = metrics["n_noise_points"] / len(self.labels)

        # Cluster balance metrics
        cluster_sizes = Counter(filtered_labels)
        sizes = list(cluster_sizes.values())
        if sizes:
            metrics["cluster_size_std"] = np.std(sizes)
            metrics["cluster_size_ratio"] = (
                max(sizes) / min(sizes) if min(sizes) > 0 else float("inf")
            )
            metrics["avg_cluster_size"] = np.mean(sizes)

        # Intra/Inter cluster distances
        intra_distances, inter_distances = self.compute_cluster_distances()
        metrics["avg_intra_cluster_distance"] = (
            np.mean(intra_distances) if intra_distances else 0
        )
        metrics["avg_inter_cluster_distance"] = (
            np.mean(inter_distances) if inter_distances else 0
        )
        metrics["separation_ratio"] = (
            (
                metrics["avg_inter_cluster_distance"]
                / metrics["avg_intra_cluster_distance"]
            )
            if metrics["avg_intra_cluster_distance"] > 0
            else 0
        )

        return metrics

    def compute_cluster_distances(self):
        """Compute intra and inter cluster distances"""
        intra_distances = []
        inter_distances = []

        # Group documents by cluster
        cluster_groups = {}
        for i, label in enumerate(self.labels):
            if label != -1:
                cluster_groups.setdefault(label, []).append(i)

        # Intra-cluster distances
        for cluster_indices in cluster_groups.values():
            if len(cluster_indices) > 1:
                cluster_embeddings = self.embeddings[cluster_indices]
                distances = cdist(
                    cluster_embeddings, cluster_embeddings, metric="cosine"
                )
                # Get upper triangle (excluding diagonal)
                upper_tri = distances[np.triu_indices_from(distances, k=1)]
                intra_distances.extend(upper_tri)

        # Inter-cluster distances (centroids)
        cluster_centroids = {}
        for label, indices in cluster_groups.items():
            cluster_centroids[label] = np.mean(self.embeddings[indices], axis=0)

        centroids = list(cluster_centroids.values())
        if len(centroids) > 1:
            centroid_distances = cdist(centroids, centroids, metric="cosine")
            upper_tri = centroid_distances[
                np.triu_indices_from(centroid_distances, k=1)
            ]
            inter_distances.extend(upper_tri)

        return intra_distances, inter_distances

    def plot_comprehensive_dashboard(self, save_path=None):
        """Create a clean and optimized evaluation dashboard"""
        fig = plt.figure(figsize=(18, 14))
        fig.suptitle(
            "Document Clustering Evaluation Dashboard",
            fontsize=18,
            fontweight="bold",
            y=0.96,
        )

        # Calculate metrics
        metrics = self.compute_all_metrics()
        if "error" in metrics:
            print(f"Error: {metrics['error']}")
            return

        # Create a 3x3 grid layout with more space
        gs = fig.add_gridspec(
            3, 3, hspace=0.35, wspace=0.25, left=0.08, right=0.92, top=0.92, bottom=0.08
        )

        # 1. Cluster Size Distribution (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_cluster_size_distribution(ax1)

        # 2. Distance Distributions (top-center)
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_distance_distributions(ax2)

        # 3. Document Type Distribution (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        self.plot_document_type_distribution(ax3)

        # 4. UMAP Visualization (middle-left)
        ax4 = fig.add_subplot(gs[1, 0])
        self.plot_umap_embeddings(ax4)

        # 5. Cluster Coherence (middle-right)
        ax5 = fig.add_subplot(gs[1, 2])
        self.plot_cluster_coherence(ax5)

        # 6. Similarity Heatmap (bottom-left)
        ax6 = fig.add_subplot(gs[2, 0])
        self.plot_similarity_heatmap(ax6)

        # 7. Individual Cluster Quality (bottom-center)
        ax7 = fig.add_subplot(gs[2, 1])
        self.plot_individual_cluster_quality(ax7)

        # 8. Summary Statistics (bottom-right)
        ax8 = fig.add_subplot(gs[2, 2])
        self.plot_summary_statistics(ax8, metrics)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")

        plt.show()
        return metrics

    def plot_cluster_size_distribution(self, ax):
        """Plot cluster size distribution with improved formatting"""
        cluster_sizes = Counter(self.labels)
        if -1 in cluster_sizes:
            del cluster_sizes[-1]  # Remove noise

        if not cluster_sizes:
            ax.text(
                0.5,
                0.5,
                "No clusters found",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title("Cluster Size Distribution", fontweight="bold", fontsize=11)
            return

        clusters, sizes = zip(*sorted(cluster_sizes.items()))
        bars = ax.bar(
            range(len(clusters)),
            sizes,
            color="steelblue",
            alpha=0.8,
            edgecolor="navy",
            linewidth=1,
        )
        ax.set_xlabel("Cluster ID", fontsize=10)
        ax.set_ylabel("Number of Documents", fontsize=10)
        ax.set_title("Cluster Size Distribution", fontweight="bold", fontsize=11)
        ax.set_xticks(range(len(clusters)))
        ax.set_xticklabels(clusters, fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars with better positioning
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + max(sizes) * 0.01,
                str(size),
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
            )

    def plot_distance_distributions(self, ax):
        """Plot intra vs inter cluster distance distributions with cleaner styling"""
        intra_distances, inter_distances = self.compute_cluster_distances()

        if not intra_distances and not inter_distances:
            ax.text(
                0.5,
                0.5,
                "Insufficient data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title("Distance Distributions", fontweight="bold", fontsize=11)
            return

        if intra_distances:
            ax.hist(
                intra_distances,
                alpha=0.7,
                label=f"Intra-cluster (n={len(intra_distances)})",
                bins=min(15, len(intra_distances) // 2 + 1),
                color="coral",
                edgecolor="darkred",
            )
        if inter_distances:
            ax.hist(
                inter_distances,
                alpha=0.7,
                label=f"Inter-cluster (n={len(inter_distances)})",
                bins=min(15, len(inter_distances) // 2 + 1),
                color="lightgreen",
                edgecolor="darkgreen",
            )

        ax.set_xlabel("Cosine Distance", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title("Intra vs Inter-cluster Distances", fontweight="bold", fontsize=11)
        ax.legend(fontsize=9)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.3)

    def plot_document_type_distribution(self, ax):
        """Plot document type distribution with better formatting"""
        if not self.document_metadata:
            ax.text(
                0.5,
                0.5,
                "No metadata available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title("Document Type Distribution", fontweight="bold", fontsize=11)
            return

        # Get file extensions
        extensions = [
            meta.get("extension", "unknown") for meta in self.document_metadata
        ]
        unique_extensions = list(set(extensions))

        # Create matrix: clusters x file types
        cluster_ids = sorted([c for c in set(self.labels) if c != -1])
        if not cluster_ids:
            ax.text(
                0.5,
                0.5,
                "No clusters found",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title("Document Type Distribution", fontweight="bold", fontsize=11)
            return

        matrix = np.zeros((len(cluster_ids), len(unique_extensions)))

        for i, cluster_id in enumerate(cluster_ids):
            cluster_mask = self.labels == cluster_id
            cluster_extensions = [
                ext for ext, mask in zip(extensions, cluster_mask) if mask
            ]
            ext_counts = Counter(cluster_extensions)

            for j, ext in enumerate(unique_extensions):
                matrix[i, j] = ext_counts.get(ext, 0)

        # Normalize by cluster size
        cluster_sizes = [np.sum(matrix[i, :]) for i in range(len(cluster_ids))]
        for i in range(len(cluster_ids)):
            if cluster_sizes[i] > 0:
                matrix[i, :] = matrix[i, :] / cluster_sizes[i]

        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            xticklabels=[
                ext[:8] for ext in unique_extensions
            ],  # Truncate long extensions
            yticklabels=[f"C{cid}" for cid in cluster_ids],
            cmap="Blues",
            ax=ax,
            cbar_kws={"label": "Proportion"},
        )
        ax.set_title(
            "Document Type Distribution\n(Proportion per Cluster)",
            fontweight="bold",
            fontsize=11,
        )
        ax.set_xlabel("File Extension", fontsize=10)
        ax.set_ylabel("Cluster ID", fontsize=10)
        ax.tick_params(labelsize=9)

    def plot_umap_embeddings(self, ax):
        """Plot UMAP embedding visualization"""
        try:
            if len(self.embeddings) < 4:
                ax.text(
                    0.5,
                    0.5,
                    "Too few documents",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("UMAP Visualization", fontweight="bold")
                return

            # Use UMAP for visualization
            umap_reducer = UMAP(
                n_components=2,
                random_state=42,
                n_neighbors=min(15, len(self.embeddings) - 1),
            )
            embeddings_2d = umap_reducer.fit_transform(self.embeddings)

            # Create scatter plot with better colors
            unique_labels = sorted(set(self.labels))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

            for i, label in enumerate(unique_labels):
                mask = self.labels == label
                if label == -1:
                    ax.scatter(
                        embeddings_2d[mask, 0],
                        embeddings_2d[mask, 1],
                        c="gray",
                        alpha=0.6,
                        s=30,
                        label=f"Noise ({np.sum(mask)})",
                    )
                else:
                    ax.scatter(
                        embeddings_2d[mask, 0],
                        embeddings_2d[mask, 1],
                        c=[colors[i]],
                        alpha=0.7,
                        s=50,
                        label=f"Cluster {label} ({np.sum(mask)})",
                    )

            ax.set_title("UMAP Visualization", fontweight="bold")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
            ax.grid(alpha=0.3)

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"UMAP error: {str(e)[:50]}...",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("UMAP Visualization", fontweight="bold")

    def plot_cluster_coherence(self, ax):
        """Plot cluster coherence scores with improved styling"""
        coherence_scores = []
        cluster_ids = []

        for cluster_id in set(self.labels):
            if cluster_id == -1:
                continue

            cluster_mask = self.labels == cluster_id
            cluster_embeddings = self.embeddings[cluster_mask]

            if len(cluster_embeddings) > 1:
                # Calculate average pairwise similarity within cluster
                similarities = cosine_similarity(cluster_embeddings)
                # Get upper triangle excluding diagonal
                upper_tri = similarities[np.triu_indices_from(similarities, k=1)]
                coherence = np.mean(upper_tri)
                coherence_scores.append(coherence)
                cluster_ids.append(cluster_id)

        if not coherence_scores:
            ax.text(
                0.5,
                0.5,
                "No clusters found",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title("Cluster Coherence", fontweight="bold", fontsize=11)
            return

        # Color coding based on coherence quality
        colors = []
        for score in coherence_scores:
            if score > 0.7:
                colors.append("#2E8B57")  # Sea Green
            elif score > 0.5:
                colors.append("#FF8C00")  # Dark Orange
            else:
                colors.append("#DC143C")  # Crimson

        bars = ax.bar(
            range(len(cluster_ids)),
            coherence_scores,
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )
        ax.set_xlabel("Cluster ID", fontsize=10)
        ax.set_ylabel("Avg Intra-cluster Similarity", fontsize=10)
        ax.set_title("Cluster Coherence Scores", fontweight="bold", fontsize=11)
        ax.set_xticks(range(len(cluster_ids)))
        ax.set_xticklabels([f"C{cid}" for cid in cluster_ids], fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels with better positioning
        for bar, score in zip(bars, coherence_scores):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.005,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=8,
            )

        # Add horizontal reference lines
        ax.axhline(y=0.7, color="green", linestyle="--", alpha=0.5, linewidth=1)
        ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5, linewidth=1)

    def plot_similarity_heatmap(self, ax):
        """Plot within-cluster similarity heatmap with better formatting"""
        # Sample a subset if too many documents
        max_docs = 25
        if len(self.embeddings) > max_docs:
            indices = np.random.choice(len(self.embeddings), max_docs, replace=False)
            sample_embeddings = self.embeddings[indices]
            sample_labels = self.labels[indices]
        else:
            sample_embeddings = self.embeddings
            sample_labels = self.labels

        # Compute similarity matrix
        similarity_matrix = cosine_similarity(sample_embeddings)

        # Sort by cluster labels
        sorted_indices = np.argsort(sample_labels)
        sorted_similarity = similarity_matrix[sorted_indices][:, sorted_indices]
        sorted_labels = sample_labels[sorted_indices]

        # Create heatmap with better formatting
        im = sns.heatmap(
            sorted_similarity,
            cmap="viridis",
            square=True,
            cbar_kws={"label": "Cosine Similarity", "shrink": 0.8},
            ax=ax,
            xticklabels=False,
            yticklabels=False,
        )

        ax.set_title(
            "Document Similarity Matrix\n(Sorted by Cluster)",
            fontweight="bold",
            fontsize=11,
        )
        ax.set_xlabel("Document Index", fontsize=10)
        ax.set_ylabel("Document Index", fontsize=10)

        # Add cluster boundaries
        prev_label = sorted_labels[0]
        boundary_pos = 0
        for i, label in enumerate(sorted_labels):
            if label != prev_label:
                ax.axhline(y=i, color="white", linewidth=2)
                ax.axvline(x=i, color="white", linewidth=2)
                prev_label = label

    def plot_individual_cluster_quality(self, ax):
        """Plot individual cluster quality scores with improved visualization"""
        cluster_scores = {}

        for cluster_id in set(self.labels):
            if cluster_id == -1:
                continue

            cluster_mask = self.labels == cluster_id
            cluster_embeddings = self.embeddings[cluster_mask]

            if len(cluster_embeddings) < 2:
                continue

            # Intra-cluster compactness (average distance to centroid)
            centroid = np.mean(cluster_embeddings, axis=0)
            distances_to_centroid = [
                cosine_similarity([emb], [centroid])[0, 0] for emb in cluster_embeddings
            ]
            compactness = np.mean(distances_to_centroid)
            cluster_scores[cluster_id] = compactness

        if not cluster_scores:
            ax.text(
                0.5,
                0.5,
                "No clusters found",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title("Individual Cluster Quality", fontweight="bold", fontsize=11)
            return

        clusters, scores = zip(*sorted(cluster_scores.items()))

        # Color coding based on quality
        colors = []
        for score in scores:
            if score > 0.7:
                colors.append("#2E8B57")  # Sea Green
            elif score > 0.5:
                colors.append("#FF8C00")  # Dark Orange
            else:
                colors.append("#DC143C")  # Crimson

        bars = ax.bar(
            range(len(clusters)),
            scores,
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )
        ax.set_xlabel("Cluster ID", fontsize=10)
        ax.set_ylabel("Avg Similarity to Centroid", fontsize=10)
        ax.set_title("Individual Cluster Quality", fontweight="bold", fontsize=11)
        ax.set_xticks(range(len(clusters)))
        ax.set_xticklabels([f"C{cid}" for cid in clusters], fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels with better positioning
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.005,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=8,
            )

        # Add reference lines
        ax.axhline(y=0.7, color="green", linestyle="--", alpha=0.5, linewidth=1)
        ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5, linewidth=1)

    def plot_summary_statistics(self, ax, metrics):
        """Plot summary statistics with improved table formatting"""
        ax.axis("off")

        # Prepare data for table with better formatting
        table_data = [
            [
                "Silhouette Score",
                f"{metrics.get('silhouette_score', 0):.4f}",
                "Excellent"
                if metrics.get("silhouette_score", 0) > 0.7
                else "Good"
                if metrics.get("silhouette_score", 0) > 0.5
                else "Fair"
                if metrics.get("silhouette_score", 0) > 0.25
                else "Poor",
            ],
            [
                "Davies-Bouldin",
                f"{metrics.get('davies_bouldin_score', 0):.4f}",
                "Excellent"
                if metrics.get("davies_bouldin_score", float("inf")) < 0.5
                else "Good"
                if metrics.get("davies_bouldin_score", float("inf")) < 1.0
                else "Fair"
                if metrics.get("davies_bouldin_score", float("inf")) < 1.5
                else "Poor",
            ],
            ["# Clusters", f"{metrics.get('n_clusters', 0)}", ""],
            [
                "Noise Ratio",
                f"{metrics.get('noise_ratio', 0):.1%}",
                "Good"
                if metrics.get("noise_ratio", 0) < 0.1
                else "Fair"
                if metrics.get("noise_ratio", 0) < 0.3
                else "High",
            ],
            [
                "Separation Ratio",
                f"{metrics.get('separation_ratio', 0):.3f}",
                "Good"
                if metrics.get("separation_ratio", 0) > 2.0
                else "Fair"
                if metrics.get("separation_ratio", 0) > 1.0
                else "Poor",
            ],
            ["Avg Cluster Size", f"{metrics.get('avg_cluster_size', 0):.1f}", ""],
        ]

        # Create table with improved styling
        table = ax.table(
            cellText=table_data,
            colLabels=["Metric", "Value", "Quality"],
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)

        # Enhanced table styling
        for i in range(len(table_data) + 1):  # +1 for header
            for j in range(3):
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_facecolor("#2E4057")
                    cell.set_text_props(weight="bold", color="white")
                    cell.set_edgecolor("white")
                    cell.set_linewidth(2)
                else:
                    cell.set_edgecolor("gray")
                    cell.set_linewidth(1)
                    if j == 2:  # Quality column
                        quality = table_data[i - 1][2]
                        if quality == "Excellent":
                            cell.set_facecolor("#90EE90")
                        elif quality == "Good":
                            cell.set_facecolor("#98FB98")
                        elif quality == "Fair":
                            cell.set_facecolor("#FFFFE0")
                        elif quality in ["Poor", "High"]:
                            cell.set_facecolor("#FFB6C1")
                        else:
                            cell.set_facecolor("white")
                    else:
                        cell.set_facecolor("#F8F8FF" if i % 2 == 0 else "white")

        ax.set_title("Summary Statistics", fontweight="bold", fontsize=11, pad=15)

    def generate_text_report(self, metrics):
        """Generate a text-based evaluation report"""
        report = []
        report.append("=" * 60)
        report.append("DOCUMENT CLUSTERING EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"Total Documents: {len(self.labels)}")
        report.append(f"Number of Clusters: {metrics.get('n_clusters', 0)}")
        report.append(
            f"Noise Points: {metrics.get('n_noise_points', 0)} ({metrics.get('noise_ratio', 0):.1%})"
        )
        report.append("")

        report.append("QUALITY METRICS:")
        report.append("-" * 20)
        report.append(f"Silhouette Score: {metrics.get('silhouette_score', 0):.4f}")
        report.append(
            "  → Interpretation: "
            + (
                "Excellent (>0.7)"
                if metrics.get("silhouette_score", 0) > 0.7
                else "Good (0.5-0.7)"
                if metrics.get("silhouette_score", 0) > 0.5
                else "Fair (0.25-0.5)"
                if metrics.get("silhouette_score", 0) > 0.25
                else "Poor (<0.25)"
            )
        )

        report.append(
            f"Davies-Bouldin Score: {metrics.get('davies_bouldin_score', 0):.4f}"
        )
        report.append(
            "  → Interpretation: "
            + (
                "Excellent (<0.5)"
                if metrics.get("davies_bouldin_score", float("inf")) < 0.5
                else "Good (0.5-1.0)"
                if metrics.get("davies_bouldin_score", float("inf")) < 1.0
                else "Fair (1.0-1.5)"
                if metrics.get("davies_bouldin_score", float("inf")) < 1.5
                else "Poor (>1.5)"
            )
        )

        return "\n".join(report)


def evaluate_clustering_system(pickle_path, output_folder):
    """
    Main function to evaluate the clustering system

    Args:
        pickle_path (str): Path to the clustering_results.pkl file.
        output_folder (str): Path to the output folder.
    """
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    vectors = data["vectors"]
    predicted_labels = data["predicted_labels"]

    # Convert string labels to integer labels
    unique_labels = sorted(list(set(predicted_labels)))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_map[label] for label in predicted_labels])

    # Create dummy metadata and raw_texts for now
    # You can enhance this later to load actual metadata and texts
    document_metadata = [
        {"extension": path.split(".")[-1]} for path in data["paths"]
    ]
    raw_texts = [""] * len(data["paths"])

    # Create evaluation dashboard
    evaluator = ClusteringEvaluationDashboard(
        embeddings=np.array(vectors),
        labels=numeric_labels,
        document_metadata=document_metadata,
        raw_texts=raw_texts,
    )

    # Generate comprehensive evaluation
    print("Generating evaluation dashboard...")
    metrics = evaluator.plot_comprehensive_dashboard(
        save_path=f"{output_folder}/d2v_clustering_evaluation.png"
    )

    # Generate text report
    report = evaluator.generate_text_report(metrics)
    print("\n" + report)

    # Save report
    with open(
        f"{output_folder}/d2v_evaluation_report.txt", "w", encoding="utf-8"
    ) as f:
        f.write(report)

    return metrics, evaluator


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate Doc2Vec clustering results."
    )
    parser.add_argument(
        "--pickle_path",
        type=str,
        default="./d2v_workflow/clustering_results.pkl",
        help="Path to the clustering_results.pkl file.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="d2v_clusters",
        help="Path to the output folder for saving the evaluation report and plot.",
    )
    args = parser.parse_args()

    evaluate_clustering_system(args.pickle_path, args.output_folder)