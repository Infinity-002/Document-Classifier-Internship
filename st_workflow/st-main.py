from st_workflow.st_document_clustering import ImprovedDocumentClustering
from st_workflow.st_evaluation_dashboard import evaluate_clustering_system
# Basic usage
clusterer = ImprovedDocumentClustering("../documents", "../st-clusters")

# Advanced usage with all features
clusterer.run(
    use_dimensionality_reduction=True,
    dim_reduction_method="umap",
    embedding_strategy="weighted_mean",
    try_graph_clustering=True,
)

metrics, evaluator = evaluate_clustering_system(clusterer)