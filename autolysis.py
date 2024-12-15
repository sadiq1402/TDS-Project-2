dependencies = ["numpy", "seaborn", "matplotlib", "scikit-learn", "requests", "openai"]

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import requests
import json


class DataAnalyzer:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file, encoding="ISO-8859-1")
        self.output_dir = os.path.splitext(os.path.basename(csv_file))[0]
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_data_summary(self):
        """Generate a comprehensive, serializable data summary."""
        summary = {
            "total_rows": len(self.df),
            "columns": list(self.df.columns),
            "column_types": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "missing_values": {
                col: int(self.df[col].isnull().sum()) for col in self.df.columns
            },
        }

        # Add numeric summary for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary["numeric_summary"] = {}
            for col in numeric_cols:
                summary["numeric_summary"][col] = {
                    "mean": float(self.df[col].mean()),
                    "median": float(self.df[col].median()),
                    "min": float(self.df[col].min()),
                    "max": float(self.df[col].max()),
                    "std": float(self.df[col].std()),
                }

        return summary

    def perform_analysis(self):
        """Perform multiple types of analysis with error handling."""
        analyses = {}

        # Correlation Analysis
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                correlation_matrix = self.df[numeric_cols].corr()
                analyses["correlation"] = {
                    "matrix": correlation_matrix.values.tolist(),
                    "columns": list(correlation_matrix.columns),
                }
        except Exception as e:
            analyses["correlation_error"] = str(e)

        # Clustering Analysis
        try:
            numeric_data = self.df.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 1:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_data)

                # Adaptive number of clusters
                n_clusters = min(3, len(numeric_data.columns))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(scaled_data)

                analyses["clustering"] = {
                    "labels": cluster_labels.tolist(),
                    "columns": list(numeric_data.columns),
                }
        except Exception as e:
            analyses["clustering_error"] = str(e)

        # PCA Analysis
        try:
            numeric_data = self.df.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 1:
                pca = PCA(n_components=min(2, len(numeric_data.columns)))
                pca_result = pca.fit_transform(
                    StandardScaler().fit_transform(numeric_data)
                )

                analyses["pca"] = {
                    "components": pca_result.tolist(),
                    "columns": list(numeric_data.columns),
                    "explained_variance": pca.explained_variance_ratio_.tolist(),
                }
        except Exception as e:
            analyses["pca_error"] = str(e)

        return analyses

    # def generate_visualizations(self, analyses):
    #     """Generate top 3 visualizations and save them in a single PNG."""
    #     numeric_cols = self.df.select_dtypes(include=[np.number]).columns

    #     # Visualization 1: Correlation Heatmap
    #     plt.figure(figsize=(6, 4))
    #     sns.heatmap(
    #         self.df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f"
    #     )
    #     plt.title("Correlation Heatmap")
    #     heatmap_path = os.path.join(self.output_dir, "heatmap.png")
    #     plt.savefig(heatmap_path, dpi=100)
    #     plt.close()

    #     # Visualization 2: PCA Scatter Plot (if available)
    #     if "pca" in analyses:
    #         pca_data = np.array(analyses["pca"]["components"])
    #         plt.figure(figsize=(6, 4))
    #         plt.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.6)
    #         plt.title("PCA Scatter Plot")
    #         plt.xlabel("Principal Component 1")
    #         plt.ylabel("Principal Component 2")
    #         pca_path = os.path.join(self.output_dir, "pca_scatter.png")
    #         plt.savefig(pca_path, dpi=100)
    #         plt.close()

    #     # Visualization 3: Distribution of First Numeric Column
    #     if len(numeric_cols) > 0:
    #         plt.figure(figsize=(6, 4))
    #         sns.histplot(self.df[numeric_cols[0]], kde=True, bins=30)
    #         plt.title(f"Distribution of {numeric_cols[0]}")
    #         dist_path = os.path.join(self.output_dir, "distribution.png")
    #         plt.savefig(dist_path, dpi=100)
    #         plt.close()

    #     return [heatmap_path, pca_path, dist_path]

    def generate_visualizations(self, analyses):
        """Generate top 3 visualizations and save them in a single PNG."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        # Visualization 1: Correlation Heatmap
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            self.df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f"
        )
        plt.title("Correlation Heatmap")
        heatmap_path = os.path.join(self.output_dir, "heatmap.png")
        plt.savefig(heatmap_path, dpi=100)
        plt.close()

        # Visualization 2: PCA Scatter Plot (if available)
        pca_path = None  # Initialize variable
        if "pca" in analyses:
            pca_data = np.array(analyses["pca"]["components"])
            plt.figure(figsize=(6, 4))
            plt.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.6)
            plt.title("PCA Scatter Plot")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            pca_path = os.path.join(self.output_dir, "pca_scatter.png")
            plt.savefig(pca_path, dpi=100)
            plt.close()

        # Visualization 3: Distribution of First Numeric Column
        dist_path = None  # Initialize variable
        if len(numeric_cols) > 0:
            plt.figure(figsize=(6, 4))
            sns.histplot(self.df[numeric_cols[0]], kde=True, bins=30)
            plt.title(f"Distribution of {numeric_cols[0]}")
            dist_path = os.path.join(self.output_dir, "distribution.png")
            plt.savefig(dist_path, dpi=100)
            plt.close()

        return [heatmap_path, pca_path, dist_path]

    def generate_story(self, summary, analyses):
        """Generate a story using LLM based on the analysis."""
        try:
            token = os.environ.get("AIPROXY_TOKEN")
            api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

            prompt = (
                f"Data Analysis Summary:\nSummary:\n{summary}\n\n"
                f"Analyses:\n{analyses}\n\nGenerate an engaging story describing the data, insights, and implications."
            )

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            }
            data = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 1000,
                "temperature": 0.7,
            }

            response = requests.post(api_url, headers=headers, data=json.dumps(data))
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error generating story: {e}")
        return "Story generation failed."

    def create_readme(self, summary, analyses, visualizations, story):
        """Create a README.md summarizing the analysis."""
        readme_path = os.path.join(self.output_dir, "README.md")
        try:
            with open(readme_path, "w") as f:
                f.write("# Data Analysis Report\n\n")
                f.write("## Summary\n")
                f.write(f"- Total Rows: {summary['total_rows']}\n")
                f.write(f"- Columns: {summary['columns']}\n")
                f.write("\n## Visualizations\n")
                for viz in visualizations:
                    if viz:
                        f.write(f"![Visualization]({viz})\n")
                f.write("\n## Insights\n")
                f.write(story)
        except Exception as e:
            print(f"Error writing README.md: {e}")

    def run(self):
        summary = self.generate_data_summary()
        analyses = self.perform_analysis()
        visualizations = self.generate_visualizations(analyses)
        story = self.generate_story(summary, analyses)
        self.create_readme(summary, analyses, visualizations, story)


# Example Usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_analyzer.py <dataset_path>")
        sys.exit(1)

    analyzer = DataAnalyzer(sys.argv[1])
    analyzer.run()
