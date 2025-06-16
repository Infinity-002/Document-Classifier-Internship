import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, silhouette_samples
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from collections import Counter
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

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
            return {'error': 'Insufficient non-noise points for evaluation'}
        
        filtered_embeddings = self.embeddings[mask]
        filtered_labels = self.labels[mask]
        
        # Internal validation metrics
        try:
            metrics['silhouette_score'] = silhouette_score(filtered_embeddings, filtered_labels)
        except:
            metrics['silhouette_score'] = -1
            
        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(filtered_embeddings, filtered_labels)
        except:
            metrics['calinski_harabasz_score'] = 0
            
        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(filtered_embeddings, filtered_labels)
        except:
            metrics['davies_bouldin_score'] = float('inf')
        
        # Cluster-specific metrics
        metrics['n_clusters'] = self.n_clusters
        metrics['n_noise_points'] = np.sum(self.labels == -1)
        metrics['noise_ratio'] = metrics['n_noise_points'] / len(self.labels)
        
        # Cluster balance metrics
        cluster_sizes = Counter(filtered_labels)
        sizes = list(cluster_sizes.values())
        if sizes:
            metrics['cluster_size_std'] = np.std(sizes)
            metrics['cluster_size_ratio'] = max(sizes) / min(sizes) if min(sizes) > 0 else float('inf')
            metrics['avg_cluster_size'] = np.mean(sizes)
        
        # Intra/Inter cluster distances
        intra_distances, inter_distances = self.compute_cluster_distances()
        metrics['avg_intra_cluster_distance'] = np.mean(intra_distances) if intra_distances else 0
        metrics['avg_inter_cluster_distance'] = np.mean(inter_distances) if inter_distances else 0
        metrics['separation_ratio'] = (metrics['avg_inter_cluster_distance'] / 
                                     metrics['avg_intra_cluster_distance']) if metrics['avg_intra_cluster_distance'] > 0 else 0
        
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
                distances = cdist(cluster_embeddings, cluster_embeddings, metric='cosine')
                # Get upper triangle (excluding diagonal)
                upper_tri = distances[np.triu_indices_from(distances, k=1)]
                intra_distances.extend(upper_tri)
        
        # Inter-cluster distances (centroids)
        cluster_centroids = {}
        for label, indices in cluster_groups.items():
            cluster_centroids[label] = np.mean(self.embeddings[indices], axis=0)
        
        centroids = list(cluster_centroids.values())
        if len(centroids) > 1:
            centroid_distances = cdist(centroids, centroids, metric='cosine')
            upper_tri = centroid_distances[np.triu_indices_from(centroid_distances, k=1)]
            inter_distances.extend(upper_tri)
        
        return intra_distances, inter_distances
    
    def plot_comprehensive_dashboard(self, save_path=None):
        """Create a comprehensive evaluation dashboard"""
        fig = plt.figure(figsize=(20, 16))
        
        # Calculate metrics
        metrics = self.compute_all_metrics()
        if 'error' in metrics:
            print(f"Error: {metrics['error']}")
            return
        
        # 1. Silhouette Analysis
        plt.subplot(3, 4, 1)
        self.plot_silhouette_analysis()
        
        # 2. Cluster Size Distribution
        plt.subplot(3, 4, 2)
        self.plot_cluster_size_distribution()
        
        # 3. Distance Distributions
        plt.subplot(3, 4, 3)
        self.plot_distance_distributions()
        
        # 4. Metrics Summary
        plt.subplot(3, 4, 4)
        self.plot_metrics_summary(metrics)
        
        # 5. 2D Embedding Visualization
        plt.subplot(3, 4, 5)
        self.plot_2d_embeddings()
        
        # 6. Cluster Purity (if you have document types)
        plt.subplot(3, 4, 6)
        self.plot_cluster_purity()
        
        # 7. Within-cluster similarity heatmap
        plt.subplot(3, 4, 7)
        self.plot_similarity_heatmap()
        
        # 8. Cluster stability
        plt.subplot(3, 4, 8)
        self.plot_cluster_coherence()
        
        # 9. Document type distribution per cluster
        plt.subplot(3, 4, 9)
        self.plot_document_type_distribution()
        
        # 10. Nearest neighbor analysis
        plt.subplot(3, 4, 10)
        self.plot_nearest_neighbor_analysis()
        
        # 11. Cluster quality scores
        plt.subplot(3, 4, 11)
        self.plot_cluster_quality_scores()
        
        # 12. Summary statistics table
        plt.subplot(3, 4, 12)
        self.plot_summary_table(metrics)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return metrics
    
    def plot_silhouette_analysis(self):
        """Plot silhouette analysis"""
        if len(set(self.labels)) <= 1:
            plt.text(0.5, 0.5, 'No clusters found', ha='center', va='center')
            plt.title('Silhouette Analysis')
            return
        
        # Filter noise points
        mask = self.labels != -1
        if np.sum(mask) <= 1:
            plt.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            plt.title('Silhouette Analysis')
            return
        
        sample_silhouette_values = silhouette_samples(self.embeddings[mask], self.labels[mask])
        
        y_lower = 10
        for i, cluster_label in enumerate(sorted(set(self.labels[mask]))):
            cluster_silhouette_values = sample_silhouette_values[self.labels[mask] == cluster_label]
            cluster_silhouette_values.sort()
            
            size_cluster_i = cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.nipy_spectral(float(i) / len(set(self.labels[mask])))
            plt.fill_betweenx(np.arange(y_lower, y_upper),
                            0, cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)
            
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster_label))
            y_lower = y_upper + 10
        
        plt.axvline(x=silhouette_score(self.embeddings[mask], self.labels[mask]), 
                   color="red", linestyle="--", label='Average Score')
        plt.title('Silhouette Analysis')
        plt.xlabel('Silhouette Coefficient Values')
        plt.ylabel('Cluster Label')
    
    def plot_cluster_size_distribution(self):
        """Plot cluster size distribution"""
        cluster_sizes = Counter(self.labels)
        if -1 in cluster_sizes:
            del cluster_sizes[-1]  # Remove noise
        
        if not cluster_sizes:
            plt.text(0.5, 0.5, 'No clusters found', ha='center', va='center')
            plt.title('Cluster Size Distribution')
            return
        
        clusters, sizes = zip(*sorted(cluster_sizes.items()))
        plt.bar(range(len(clusters)), sizes, color='skyblue', alpha=0.7)
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Documents')
        plt.title('Cluster Size Distribution')
        plt.xticks(range(len(clusters)), clusters)
        
        # Add value labels on bars
        for i, v in enumerate(sizes):
            plt.text(i, v + 0.1, str(v), ha='center')
    
    def plot_distance_distributions(self):
        """Plot intra vs inter cluster distance distributions"""
        intra_distances, inter_distances = self.compute_cluster_distances()
        
        if not intra_distances and not inter_distances:
            plt.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            plt.title('Distance Distributions')
            return
        
        if intra_distances:
            plt.hist(intra_distances, alpha=0.7, label='Intra-cluster', bins=20, color='red')
        if inter_distances:
            plt.hist(inter_distances, alpha=0.7, label='Inter-cluster', bins=20, color='blue')
        
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.title('Intra vs Inter-cluster Distances')
        plt.legend()
    
    def plot_metrics_summary(self, metrics):
        """Plot key metrics as a bar chart"""
        metric_names = ['Silhouette\nScore', 'Calinski\nHarabasz', 'Davies\nBouldin', 'Separation\nRatio']
        
        # Normalize scores for visualization (all between 0-1, higher is better)
        values = [
            max(0, metrics.get('silhouette_score', 0)),  # -1 to 1 -> 0 to 1
            min(1, metrics.get('calinski_harabasz_score', 0) / 1000),  # Normalize by 1000
            max(0, 1 - min(1, metrics.get('davies_bouldin_score', float('inf')))),  # Invert (lower is better)
            min(1, metrics.get('separation_ratio', 0) / 10)  # Normalize by 10
        ]
        
        colors = ['green' if v > 0.5 else 'orange' if v > 0.3 else 'red' for v in values]
        plt.bar(metric_names, values, color=colors, alpha=0.7)
        plt.ylim(0, 1)
        plt.title('Clustering Quality Metrics\n(Normalized, Higher=Better)')
        plt.xticks(rotation=45)
        
        # Add value labels
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    def plot_2d_embeddings(self):
        """Plot 2D embedding visualization"""
        try:
            from sklearn.manifold import TSNE
            
            if len(self.embeddings) < 4:
                plt.text(0.5, 0.5, 'Too few documents', ha='center', va='center')
                plt.title('2D Embedding Visualization')
                return
            
            # Use t-SNE for visualization
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.embeddings)-1))
            embeddings_2d = tsne.fit_transform(self.embeddings)
            
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                c=self.labels, cmap='tab10', alpha=0.7)
            plt.title('2D Document Embedding Visualization')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            
            # Add colorbar if there are clusters
            if len(set(self.labels)) > 1:
                plt.colorbar(scatter)
                
        except ImportError:
            plt.text(0.5, 0.5, 'TSNE not available', ha='center', va='center')
            plt.title('2D Embedding Visualization')
    
    def plot_cluster_purity(self):
        """Plot cluster purity based on file extensions"""
        if not self.document_metadata:
            plt.text(0.5, 0.5, 'No metadata available', ha='center', va='center')
            plt.title('Cluster Purity by File Type')
            return
        
        # Group by cluster and file type
        cluster_file_types = {}
        for meta, label in zip(self.document_metadata, self.labels):
            if label != -1:
                ext = meta.get('extension', 'unknown')
                cluster_file_types.setdefault(label, Counter())[ext] += 1
        
        if not cluster_file_types:
            plt.text(0.5, 0.5, 'No clusters found', ha='center', va='center')
            plt.title('Cluster Purity by File Type')
            return
        
        # Calculate purity scores
        purity_scores = []
        cluster_ids = []
        
        for cluster_id, file_counts in cluster_file_types.items():
            total = sum(file_counts.values())
            max_count = max(file_counts.values())
            purity = max_count / total
            purity_scores.append(purity)
            cluster_ids.append(cluster_id)
        
        plt.bar(range(len(cluster_ids)), purity_scores, alpha=0.7, color='lightgreen')
        plt.xlabel('Cluster ID')
        plt.ylabel('Purity Score')
        plt.title('Cluster Purity by File Type')
        plt.xticks(range(len(cluster_ids)), cluster_ids)
        plt.ylim(0, 1)
        
        # Add value labels
        for i, v in enumerate(purity_scores):
            plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
    
    def plot_similarity_heatmap(self):
        """Plot within-cluster similarity heatmap"""
        # Sample a subset if too many documents
        max_docs = 50
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
        
        sns.heatmap(sorted_similarity, cmap='viridis', square=True, 
                   cbar_kws={'label': 'Cosine Similarity'})
        plt.title('Document Similarity Matrix\n(Sorted by Cluster)')
        plt.xlabel('Document Index')
        plt.ylabel('Document Index')
    
    def plot_cluster_coherence(self):
        """Plot cluster coherence scores"""
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
            plt.text(0.5, 0.5, 'No clusters found', ha='center', va='center')
            plt.title('Cluster Coherence')
            return
        
        colors = ['green' if score > 0.5 else 'orange' if score > 0.3 else 'red' 
                 for score in coherence_scores]
        plt.bar(range(len(cluster_ids)), coherence_scores, color=colors, alpha=0.7)
        plt.xlabel('Cluster ID')
        plt.ylabel('Average Intra-cluster Similarity')
        plt.title('Cluster Coherence Scores')
        plt.xticks(range(len(cluster_ids)), cluster_ids)
        
        # Add value labels
        for i, v in enumerate(coherence_scores):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    def plot_document_type_distribution(self):
        """Plot document type distribution across clusters"""
        if not self.document_metadata:
            plt.text(0.5, 0.5, 'No metadata available', ha='center', va='center')
            plt.title('Document Type Distribution')
            return
        
        # Get file extensions
        extensions = [meta.get('extension', 'unknown') for meta in self.document_metadata]
        unique_extensions = list(set(extensions))
        
        # Create matrix: clusters x file types
        cluster_ids = sorted([c for c in set(self.labels) if c != -1])
        if not cluster_ids:
            plt.text(0.5, 0.5, 'No clusters found', ha='center', va='center')
            plt.title('Document Type Distribution')
            return
        
        matrix = np.zeros((len(cluster_ids), len(unique_extensions)))
        
        for i, cluster_id in enumerate(cluster_ids):
            cluster_mask = self.labels == cluster_id
            cluster_extensions = [ext for ext, mask in zip(extensions, cluster_mask) if mask]
            ext_counts = Counter(cluster_extensions)
            
            for j, ext in enumerate(unique_extensions):
                matrix[i, j] = ext_counts.get(ext, 0)
        
        # Normalize by cluster size
        cluster_sizes = [np.sum(matrix[i, :]) for i in range(len(cluster_ids))]
        for i in range(len(cluster_ids)):
            if cluster_sizes[i] > 0:
                matrix[i, :] = matrix[i, :] / cluster_sizes[i]
        
        sns.heatmap(matrix, annot=True, fmt='.2f', 
                   xticklabels=unique_extensions, yticklabels=cluster_ids,
                   cmap='Blues')
        plt.title('Document Type Distribution\n(Proportion per Cluster)')
        plt.xlabel('File Extension')
        plt.ylabel('Cluster ID')
    
    def plot_nearest_neighbor_analysis(self):
        """Analyze k-nearest neighbors within vs across clusters"""
        k = min(5, len(self.embeddings) - 1)
        if k < 1:
            plt.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            plt.title('Nearest Neighbor Analysis')
            return
        
        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(self.embeddings)
        distances, indices = nbrs.kneighbors(self.embeddings)
        
        # Calculate same-cluster ratios
        same_cluster_ratios = []
        
        for i, (doc_indices, doc_label) in enumerate(zip(indices, self.labels)):
            if doc_label == -1:
                continue
            
            # Skip self (first neighbor)
            neighbor_labels = self.labels[doc_indices[1:]]
            same_cluster_count = np.sum(neighbor_labels == doc_label)
            same_cluster_ratio = same_cluster_count / k
            same_cluster_ratios.append(same_cluster_ratio)
        
        if not same_cluster_ratios:
            plt.text(0.5, 0.5, 'No valid clusters', ha='center', va='center')
            plt.title('Nearest Neighbor Analysis')
            return
        
        plt.hist(same_cluster_ratios, bins=20, alpha=0.7, color='purple', edgecolor='black')
        plt.axvline(np.mean(same_cluster_ratios), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(same_cluster_ratios):.3f}')
        plt.xlabel('Fraction of Same-Cluster Neighbors')
        plt.ylabel('Number of Documents')
        plt.title(f'K-Nearest Neighbor Analysis (k={k})')
        plt.legend()
    
    def plot_cluster_quality_scores(self):
        """Plot individual cluster quality scores"""
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
            distances_to_centroid = [cosine_similarity([emb], [centroid])[0, 0] 
                                   for emb in cluster_embeddings]
            compactness = np.mean(distances_to_centroid)
            
            cluster_scores[cluster_id] = compactness
        
        if not cluster_scores:
            plt.text(0.5, 0.5, 'No clusters found', ha='center', va='center')
            plt.title('Individual Cluster Quality')
            return
        
        clusters, scores = zip(*sorted(cluster_scores.items()))
        colors = ['green' if score > 0.7 else 'orange' if score > 0.5 else 'red' 
                 for score in scores]
        
        plt.bar(range(len(clusters)), scores, color=colors, alpha=0.7)
        plt.xlabel('Cluster ID')
        plt.ylabel('Avg Similarity to Centroid')
        plt.title('Individual Cluster Quality Scores')
        plt.xticks(range(len(clusters)), clusters)
        
        # Add value labels
        for i, v in enumerate(scores):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    def plot_summary_table(self, metrics):
        """Plot summary statistics table"""
        plt.axis('off')
        
        # Prepare data for table
        table_data = [
            ['Metric', 'Value', 'Interpretation'],
            ['Silhouette Score', f"{metrics.get('silhouette_score', 0):.4f}", 
             'Good: > 0.5, Poor: < 0.25'],
            ['Calinski-Harabasz', f"{metrics.get('calinski_harabasz_score', 0):.2f}", 
             'Higher is better'],
            ['Davies-Bouldin', f"{metrics.get('davies_bouldin_score', 0):.4f}", 
             'Lower is better'],
            ['Number of Clusters', f"{metrics.get('n_clusters', 0)}", 
             'Depends on data'],
            ['Noise Ratio', f"{metrics.get('noise_ratio', 0):.3f}", 
             'Lower is better'],
            ['Separation Ratio', f"{metrics.get('separation_ratio', 0):.3f}", 
             'Higher is better'],
            ['Avg Cluster Size', f"{metrics.get('avg_cluster_size', 0):.1f}", 
             'Balanced is better']
        ]
        
        table = plt.table(cellText=table_data[1:], colLabels=table_data[0],
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Color code the values
        for i in range(1, len(table_data)):
            if 'Silhouette' in table_data[i][0]:
                val = float(table_data[i][1])
                color = 'lightgreen' if val > 0.5 else 'lightyellow' if val > 0.25 else 'lightcoral'
                table[(i, 1)].set_facecolor(color)
            elif 'Noise Ratio' in table_data[i][0]:
                val = float(table_data[i][1])
                color = 'lightgreen' if val < 0.1 else 'lightyellow' if val < 0.3 else 'lightcoral'
                table[(i, 1)].set_facecolor(color)
        
        plt.title('Clustering Summary Statistics', y=0.9, fontsize=14, fontweight='bold')
    
    def generate_text_report(self, metrics):
        """Generate a text-based evaluation report"""
        report = []
        report.append("="*60)
        report.append("DOCUMENT CLUSTERING EVALUATION REPORT")
        report.append("="*60)
        report.append(f"Total Documents: {len(self.labels)}")
        report.append(f"Number of Clusters: {metrics.get('n_clusters', 0)}")
        report.append(f"Noise Points: {metrics.get('n_noise_points', 0)} ({metrics.get('noise_ratio', 0):.1%})")
        report.append("")
        
        report.append("QUALITY METRICS:")
        report.append("-" * 20)
        report.append(f"Silhouette Score: {metrics.get('silhouette_score', 0):.4f}")
        report.append("  → Interpretation: " + 
                     ("Excellent (>0.7)" if metrics.get('silhouette_score', 0) > 0.7 else
                      "Good (0.5-0.7)" if metrics.get('silhouette_score', 0) > 0.5 else
                      "Fair (0.25-0.5)" if metrics.get('silhouette_score', 0) > 0.25 else
                      "Poor (<0.25)"))
        
        report.append(f"Davies-Bouldin Score: {metrics.get('davies_bouldin_score', 0):.4f}")
        report.append("  → Interpretation: " + 
                     ("Excellent (<0.5)" if metrics.get('davies_bouldin_score', float('inf')) < 0.5 else
                      "Good (0.5-1.0)" if metrics.get('davies_bouldin_score', float('inf')) < 1.0 else
                      "Fair (1.0-1.5)" if metrics.get('davies_bouldin_score', float('inf')) < 1.5 else
                      "Poor (>1.5)"))
        
        report.append("")
        report.append("RECOMMENDATIONS:")
        report.append("-" * 20)
        
        # Generate recommendations based on metrics
        if metrics.get('silhouette_score', 0) < 0.25:
            report.append("• Consider different embedding models or preprocessing")
            report.append("• Try different clustering algorithms")
            report.append("• Check if documents are too similar or too diverse")
        
        if metrics.get('noise_ratio', 0) > 0.3:
            report.append("• High noise ratio - consider relaxing clustering parameters")
            report.append("• Documents might be too diverse for current approach")
        
        if metrics.get('cluster_size_ratio', 1) > 10:
            report.append("• Imbalanced clusters - consider different clustering parameters")
            report.append("• Some clusters might be too large or too small")
        
        return "\n".join(report)

def evaluate_clustering_system(clusterer):
    """
    Main function to evaluate the clustering system
    
    Args:
        clusterer: Instance of ImprovedDocumentClustering after running
    """
    if not hasattr(clusterer, 'labels') or len(clusterer.labels) == 0:
        print("No clustering results found. Run clustering first.")
        return
    
    # Create evaluation dashboard
    evaluator = ClusteringEvaluationDashboard(
        embeddings=clusterer.embeddings,
        labels=clusterer.labels,
        document_metadata=clusterer.document_metadata,
        raw_texts=clusterer.raw_texts
    )
    
    # Generate comprehensive evaluation
    print("Generating evaluation dashboard...")
    metrics = evaluator.plot_comprehensive_dashboard(
        save_path=clusterer.output_folder / "clustering_evaluation.png"
    )
    
    # Generate text report
    report = evaluator.generate_text_report(metrics)
    print("\n" + report)
    
    # Save report
    with open(clusterer.output_folder / "evaluation_report.txt", "w", encoding='utf-8') as f:
        f.write(report)
    
    return metrics, evaluator