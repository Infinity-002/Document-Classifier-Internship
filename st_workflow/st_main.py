import argparse
from st_document_clustering import DocumentClustering
from st_evaluation import evaluate_clustering_system

def main():
    parser = argparse.ArgumentParser(description="CLI for Document Clustering and Evaluation")

    parser = argparse.ArgumentParser(description="CLI for Document Clustering and Evaluation")
    parser.add_argument("-i", "--input", required=True, help="Path to the input documents")
    parser.add_argument("-o", "--output", default="st_clusters", help="Folder to save clustering output")

    args = parser.parse_args()

    # Initialize clusterer
    clusterer = DocumentClustering(args.input, args.output)

    # Run clustering with chosen config
    clusterer.run()

    # Run evaluation
    metrics, evaluator = evaluate_clustering_system(clusterer)

    # Print summary metrics
    print("\nClustering Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")

if __name__ == "__main__":
    main()