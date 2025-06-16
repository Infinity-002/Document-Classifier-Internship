from pathlib import Path
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document as DocxDocument
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import numpy as np
import shutil
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt
from umap import UMAP
import networkx as nx
from transformers import pipeline
import re
import nltk
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ImprovedDocumentClustering:
    SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".html", ".htm"}

    def __init__(self, input_folder, output_folder, model_name="all-MiniLM-L6-v2"):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        
        # Try different models based on use case
        self.model = SentenceTransformer(model_name)
        
        self.file_paths = [p for p in self.input_folder.iterdir() if p.suffix.lower() in self.SUPPORTED_EXTENSIONS]
        self.embeddings = []  # Keep as list initially
        self.raw_texts = []
        self.processed_texts = []
        self.labels = []
        self.document_metadata = []
        
        # Initialize summarizer for cluster descriptions
        try:
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", 
                                     max_length=50, min_length=10, do_sample=False)
        except:
            self.summarizer = None
            print("Could not load summarizer. Will use TF-IDF for cluster summaries.")

    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation for sentence structure
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        # Remove very short lines that are likely noise
        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 20]
        return '\n'.join(lines)

    def extract_text(self, path: Path) -> tuple[str, dict]:
        """Extract text and metadata"""
        metadata = {
            'filename': path.name,
            'size': path.stat().st_size,
            'extension': path.suffix.lower()
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
                metadata.update({
                    'word_count': len(text.split()),
                    'char_count': len(text),
                    'sentence_count': len(nltk.sent_tokenize(text))
                })
            
            return text, metadata
        except Exception as e:
            print(f"❌ Error reading {path.name}: {e}")
            return "", metadata

    def chunk_document(self, text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
        """Split document into overlapping chunks for better representation"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.split()) > 20:  # Only keep substantial chunks
                chunks.append(chunk)
        
        return chunks if chunks else [text]

    def embed_documents(self, use_chunking=True, embedding_strategy='mean_pooling'):
        """Enhanced document embedding with multiple strategies"""
        print("🔍 Embedding documents...")
        
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
            full_text = f"{path.stem.replace('_', ' ').replace('-', ' ')}\n\n{processed_text}"
            
            if use_chunking:
                chunks = self.chunk_document(full_text)
                chunk_embeddings = self.model.encode(chunks, convert_to_tensor=False)
                
                # Different pooling strategies
                if embedding_strategy == 'mean_pooling':
                    embedding = np.mean(chunk_embeddings, axis=0)
                elif embedding_strategy == 'max_pooling':
                    embedding = np.max(chunk_embeddings, axis=0)
                elif embedding_strategy == 'weighted_mean':
                    # Weight chunks by their position (give more weight to beginning)
                    weights = np.exp(-0.1 * np.arange(len(chunks)))
                    weights = weights / np.sum(weights)
                    embedding = np.average(chunk_embeddings, axis=0, weights=weights)
                else:
                    embedding = np.mean(chunk_embeddings, axis=0)
            else:
                # Truncate if too long
                if len(full_text.split()) > 500:
                    full_text = ' '.join(full_text.split()[:500])
                embedding = self.model.encode(full_text, convert_to_tensor=False)
            
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

    def reduce_dimensionality(self, method='umap', n_components=50):
        """Reduce dimensionality before clustering"""
        if len(self.embeddings) <= n_components:
            return self.embeddings
        
        print(f"🔄 Reducing dimensionality with {method.upper()}...")
        
        if method == 'umap':
            reducer = UMAP(n_components=n_components, random_state=42, 
                          n_neighbors=min(15, len(self.embeddings)-1))
        elif method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        else:
            return self.embeddings
        
        reduced_embeddings = reducer.fit_transform(self.embeddings)
        return reduced_embeddings

    def evaluate_clustering(self, embeddings, labels):
        """Comprehensive clustering evaluation"""
        if len(set(labels)) <= 1:
            return {'silhouette': -1, 'calinski_harabasz': 0, 'davies_bouldin': float('inf')}
        
        try:
            # Filter out noise points for evaluation
            mask = labels != -1
            if np.sum(mask) <= 1:
                return {'silhouette': -1, 'calinski_harabasz': 0, 'davies_bouldin': float('inf')}
            
            filtered_embeddings = embeddings[mask]
            filtered_labels = labels[mask]
            
            silhouette = silhouette_score(filtered_embeddings, filtered_labels)
            calinski = calinski_harabasz_score(filtered_embeddings, filtered_labels)
            davies_bouldin = davies_bouldin_score(filtered_embeddings, filtered_labels)
            
            return {
                'silhouette': silhouette,
                'calinski_harabasz': calinski,
                'davies_bouldin': davies_bouldin
            }
        except:
            return {'silhouette': -1, 'calinski_harabasz': 0, 'davies_bouldin': float('inf')}

    def cluster_with_multiple_methods(self, embeddings):
        """Try multiple clustering methods and select the best"""
        methods = {}
        
        # HDBSCAN with different parameters
        for min_cluster_size in [2, 3, max(2, len(embeddings) // 10)]:
            try:
                clusterer = HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=max(1, min_cluster_size - 1),
                    metric='euclidean',
                    cluster_selection_epsilon=0.1,
                    prediction_data=True
                )
                labels = clusterer.fit_predict(embeddings)
                score = self.evaluate_clustering(embeddings, labels)
                methods[f'hdbscan_mcs{min_cluster_size}'] = {
                    'labels': labels, 
                    'score': score,
                    'clusterer': clusterer
                }
            except Exception as e:
                print(f"HDBSCAN with min_cluster_size={min_cluster_size} failed: {e}")
        
        # KMeans with different k values
        max_k = min(10, len(embeddings) - 1)
        for k in range(2, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                score = self.evaluate_clustering(embeddings, labels)
                methods[f'kmeans_k{k}'] = {
                    'labels': labels, 
                    'score': score,
                    'clusterer': kmeans
                }
            except Exception as e:
                print(f"KMeans with k={k} failed: {e}")
        
        # Agglomerative clustering
        for linkage in ['ward', 'complete', 'average']:
            for k in range(2, min(8, len(embeddings))):
                try:
                    agg = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                    labels = agg.fit_predict(embeddings)
                    score = self.evaluate_clustering(embeddings, labels)
                    methods[f'agg_{linkage}_k{k}'] = {
                        'labels': labels, 
                        'score': score,
                        'clusterer': agg
                    }
                except Exception as e:
                    print(f"Agglomerative {linkage} with k={k} failed: {e}")
        
        # Select best method based on silhouette score
        best_method = max(methods.items(), 
                         key=lambda x: x[1]['score']['silhouette'])
        
        print(f"🏆 Best method: {best_method[0]} with silhouette score: {best_method[1]['score']['silhouette']:.4f}")
        
        return best_method[1]['labels'], best_method[1]['clusterer']

    def create_similarity_graph(self, embeddings, threshold=0.7):
        """Create a similarity graph for community detection"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarity_matrix = cosine_similarity(embeddings)
        
        # Create graph
        G = nx.Graph()
        n_docs = len(embeddings)
        
        for i in range(n_docs):
            G.add_node(i, filename=self.document_metadata[i]['filename'])
        
        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                if similarity_matrix[i, j] > threshold:
                    G.add_edge(i, j, weight=similarity_matrix[i, j])
        
        # Detect communities
        try:
            communities = nx.community.louvain_communities(G, seed=42)
            labels = np.full(n_docs, -1)
            for cluster_id, community in enumerate(communities):
                for node in community:
                    labels[node] = cluster_id
            return labels
        except:
            return np.full(n_docs, -1)

    def generate_cluster_names(self, cluster_labels):
        """Generate meaningful names for clusters"""
        cluster_names = {}
        cluster_texts = {}
        
        # Group texts by cluster
        for text, label in zip(self.processed_texts, cluster_labels):
            if label == -1:
                continue
            cluster_texts.setdefault(label, []).append(text)
        
        for label, docs in cluster_texts.items():
            combined_text = ' '.join(docs)
            
            # Try summarization first
            if self.summarizer and len(combined_text.split()) > 50:
                try:
                    # Truncate if too long for summarizer
                    if len(combined_text.split()) > 1000:
                        combined_text = ' '.join(combined_text.split()[:1000])
                    
                    summary = self.summarizer(combined_text)[0]['summary_text']
                    # Extract key phrases from summary
                    words = re.findall(r'\b[A-Z][a-z]+\b', summary)
                    if words:
                        cluster_names[label] = '_'.join(words[:3]).lower()
                    else:
                        cluster_names[label] = f"cluster_{label}"
                except:
                    cluster_names[label] = f"cluster_{label}"
            
            # Fallback to TF-IDF
            if label not in cluster_names:
                try:
                    vec = TfidfVectorizer(max_features=3, stop_words="english", 
                                        ngram_range=(1, 2))
                    tfidf = vec.fit_transform(docs)
                    top_terms = vec.get_feature_names_out()
                    cluster_names[label] = '_'.join(top_terms).replace(' ', '_')
                except:
                    cluster_names[label] = f"cluster_{label}"
        
        return cluster_names

    def cluster_and_save(self):
        """Save documents to cluster folders with meaningful names"""
        cluster_names = self.generate_cluster_names(self.labels)
        cluster_stats = Counter(self.labels)
        
        print("\n📁 Creating cluster folders...")
        for i, (label, filename) in enumerate(zip(self.labels, [meta['filename'] for meta in self.document_metadata])):
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

    def visualize_clusters(self, save_plot=True):
        """Enhanced visualization with multiple methods"""
        if len(self.embeddings) < 3:
            print("Not enough samples to visualize.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # UMAP visualization
        try:
            umap_reducer = UMAP(n_components=2, random_state=42, 
                              n_neighbors=min(15, len(self.embeddings)-1))
            umap_reduced = umap_reducer.fit_transform(self.embeddings)
            
            axes[0].scatter(umap_reduced[:, 0], umap_reduced[:, 1], 
                           c=self.labels, cmap='tab10', alpha=0.7)
            axes[0].set_title("UMAP Visualization")
            axes[0].set_xlabel("UMAP 1")
            axes[0].set_ylabel("UMAP 2")
        except Exception as e:
            print(f"UMAP visualization failed: {e}")
        
        # t-SNE visualization
        try:
            tsne = TSNE(n_components=2, perplexity=min(30, len(self.embeddings) - 1), 
                       random_state=42, max_iter=500)
            tsne_reduced = tsne.fit_transform(self.embeddings)
            
            axes[1].scatter(tsne_reduced[:, 0], tsne_reduced[:, 1], 
                           c=self.labels, cmap='tab10', alpha=0.7)
            axes[1].set_title("t-SNE Visualization")
            axes[1].set_xlabel("t-SNE 1")
            axes[1].set_ylabel("t-SNE 2")
        except Exception as e:
            print(f"t-SNE visualization failed: {e}")
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(self.output_folder / "cluster_visualization.png", dpi=300, bbox_inches='tight')
        
        plt.show()

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
            cluster_docs = [meta['filename'] for meta, l in zip(self.document_metadata, self.labels) if l == label]
            sample_docs = cluster_docs[:3]
            for doc in sample_docs:
                print(f"   - {doc}")
            if len(cluster_docs) > 3:
                print(f"   ... and {len(cluster_docs) - 3} more")
            print()

    def run(self, use_dimensionality_reduction=True, dim_reduction_method='umap', 
            embedding_strategy='weighted_mean', try_graph_clustering=False):
        """Enhanced run method with more options"""
        print("🚀 Starting improved document clustering pipeline...")
        
        # Step 1: Embed documents
        self.embed_documents(embedding_strategy=embedding_strategy)
        
        if len(self.embeddings) == 0:
            print("No embeddings generated. Check file content.")
            return
        
        # Step 2: Optional dimensionality reduction
        embeddings_for_clustering = self.embeddings
        if use_dimensionality_reduction and len(self.embeddings) > 50:
            embeddings_for_clustering = self.reduce_dimensionality(
                method=dim_reduction_method, 
                n_components=min(50, len(self.embeddings) // 2)
            )
        
        # Step 3: Try graph-based clustering
        if try_graph_clustering:
            print("Trying graph-based clustering...")
            graph_labels = self.create_similarity_graph(embeddings_for_clustering)
            if len(set(graph_labels)) > 1:
                graph_score = self.evaluate_clustering(embeddings_for_clustering, graph_labels)
                print(f"Graph clustering silhouette score: {graph_score['silhouette']:.4f}")
        
        # Step 4: Multiple clustering methods
        print("🔍 Trying multiple clustering methods...")
        self.labels, self.best_clusterer = self.cluster_with_multiple_methods(embeddings_for_clustering)
        
        # Step 5: Save results
        self.cluster_and_save()
        
        # Step 6: Visualize and summarize
        self.visualize_clusters()
        self.print_cluster_summary()
        
        print("Clustering pipeline completed!")