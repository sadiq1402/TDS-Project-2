# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "numpy",
#   "scikit-learn",
#   "requests",
#   "ipykernel",
#   "scikit-learn"
# ]
# ///

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class DataAnalyzer:
    def __init__(self, csv_file, output_dir="."):
        self.csv_file = csv_file
        self.output_dir = output_dir
        self.df = None

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        try:
            self.df = pd.read_csv(self.csv_file, encoding="ISO-8859-1")
            print("Dataset loaded successfully!")
        except UnicodeDecodeError as e:
            print(f"Error reading file: {e}")
            raise

    def generate_data_summary(self):
        """Generate a comprehensive, serializable data summary"""
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
        """Perform multiple types of analysis with error handling"""
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

    def call_llm(self, prompt):
        """Call the LLM for dataset insights or narrative"""
        api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('AIPROXY_TOKEN')}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            return (
                response.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "No response")
            )
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return "Error generating response."

    def visualize_data(self):
        print("Generating visualizations...")

        # Correlation Matrix Heatmap
        heatmap_file = os.path.join(self.output_dir, "correlation_matrix.png")
        if not self.df.empty:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    self.df[numeric_cols].corr(),
                    annot=True,
                    cmap="coolwarm",
                    fmt=".2f",
                    linewidths=0.5,
                )
                plt.title("Correlation Matrix")
                plt.savefig(heatmap_file)
                plt.close()
        else:
            heatmap_file = None

        print("Visualizations generated.")
        return heatmap_file

    def create_readme(self, heatmap_file):
        print("Creating README file...")
        readme_file = os.path.join(self.output_dir, "README.md")
        try:
            with open(readme_file, "w") as f:
                f.write("# Automated Data Analysis Report\n\n")

                # Summary Statistics
                f.write("## Summary Statistics\n")
                summary = self.generate_data_summary()
                f.write(json.dumps(summary, indent=4))
                f.write("\n\n")

                # Analysis
                f.write("## Analysis\n")
                analysis_results = self.perform_analysis()
                f.write(json.dumps(analysis_results, indent=4))
                f.write("\n\n")

                # Visualizations
                f.write("## Visualizations\n")
                if heatmap_file:
                    f.write(f"![Correlation Matrix]({heatmap_file})\n")

                # LLM Insights
                f.write("\n## LLM Insights\n")
                prompt = (
                    "Provide a detailed analysis and insights based on this dataset: "
                    + json.dumps(summary)
                )
                llm_response = self.call_llm(prompt)
                f.write(llm_response)

            print("README file created.")
        except Exception as e:
            print(f"Error writing README file: {e}")

    def run(self):
        self.load_data()
        heatmap_file = self.visualize_data()
        self.create_readme(heatmap_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Automated data analysis tool.")
    parser.add_argument("csv_file", help="Path to the CSV file.")
    parser.add_argument("--output_dir", default=".", help="Directory to save outputs.")

    args = parser.parse_args()

    analyzer = DataAnalyzer(csv_file=args.csv_file, output_dir=args.output_dir)
    analyzer.run()
