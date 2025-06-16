# Hybrid NLP-Based Document Classifier 

This project provides tools for unsupervised document clustering using NLP based approaches. It includes:

- 📁 `st_workflow`: Sentence transformer based document clustering
- 🔬 `d2v_workflow`: Document2Vec-based document clustering

## Getting Started

### 1. Install Dependencies

We use [uv](https://github.com/astral-sh/uv) as a faster and more reliable alternative to pip.

#### Install `uv`:
```bash
pip install uv
uv sync
```

### 2. Run Workflows

#### Standard Clustering Workflow
```bash
uv run ./st_workflow/st_main.py -i documents/
```

- `-i`: Path to a folder containing your text documents.

- `-o`: (Optional) Output folder. Defaults to st_clusters/.

####  Doc2Vec Clustering Workflow
```bash
uv run ./d2v_workflow/d2v_document_clustering.py
```

## Notes

- `st_workflow` uses a pre-trained model & supports multiple formats: txt, pdf, docx, html, htm.
- `d2v_workflow` was trained using the ag_news dataset.
