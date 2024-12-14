# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "numpy",
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


class DataAnalyzer:
    def __init__(self, csv_file, output_dir="."):
        self.csv_file = csv_file
        self.output_dir = output_dir
        self.df = None
        self.summary_stats = None
        self.missing_values = None
        self.corr_matrix = None
        self.outliers = None

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        try:
            self.df = pd.read_csv(self.csv_file, encoding="ISO-8859-1")
            print("Dataset loaded successfully!")
        except UnicodeDecodeError as e:
            print(f"Error reading file: {e}")
            raise

    def analyze_data(self):
        print("Analyzing the data...")
        self.summary_stats = self.df.describe()
        self.missing_values = self.df.isnull().sum()
        numeric_df = self.df.select_dtypes(include=[np.number])
        self.corr_matrix = numeric_df.corr() if not numeric_df.empty else pd.DataFrame()
        print("Data analysis complete.")

    def detect_outliers(self):
        print("Detecting outliers...")
        df_numeric = self.df.select_dtypes(include=[np.number])
        Q1 = df_numeric.quantile(0.25)
        Q3 = df_numeric.quantile(0.75)
        IQR = Q3 - Q1
        self.outliers = (
            (df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))
        ).sum()
        print("Outliers detection complete.")

    def visualize_data(self):
        print("Generating visualizations...")

        # Correlation Matrix Heatmap
        heatmap_file = os.path.join(self.output_dir, "correlation_matrix.png")
        if not self.corr_matrix.empty:
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                self.corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
            )
            plt.title("Correlation Matrix")
            plt.savefig(heatmap_file)
            plt.close()
        else:
            heatmap_file = None

        # Outliers Plot
        outliers_file = os.path.join(self.output_dir, "outliers.png")
        if not self.outliers.empty and self.outliers.sum() > 0:
            plt.figure(figsize=(10, 6))
            self.outliers.plot(kind="bar", color="red")
            plt.title("Outliers Detection")
            plt.xlabel("Columns")
            plt.ylabel("Number of Outliers")
            plt.savefig(outliers_file)
            plt.close()
        else:
            outliers_file = None

        # Distribution Plot
        dist_plot_file = os.path.join(self.output_dir, "distribution.png")
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            first_numeric_column = numeric_columns[0]
            plt.figure(figsize=(10, 6))
            sns.histplot(self.df[first_numeric_column], kde=True, color="blue", bins=30)
            plt.title(f"Distribution of {first_numeric_column}")
            plt.savefig(dist_plot_file)
            plt.close()
        else:
            dist_plot_file = None

        print("Visualizations generated.")
        return heatmap_file, outliers_file, dist_plot_file

    def create_readme(self, heatmap_file, outliers_file, dist_plot_file):
        print("Creating README file...")
        readme_file = os.path.join(self.output_dir, "README.md")
        try:
            with open(readme_file, "w") as f:
                f.write("# Automated Data Analysis Report\n\n")
                f.write("## Summary Statistics\n")
                f.write(self.summary_stats.to_markdown())
                f.write("\n\n## Missing Values\n")
                f.write(self.missing_values.to_markdown())
                f.write("\n\n## Correlation Matrix\n")
                if heatmap_file:
                    f.write(f"![Correlation Matrix]({heatmap_file})\n")
                f.write("\n\n## Outliers\n")
                f.write(self.outliers.to_markdown())
                if outliers_file:
                    f.write(f"![Outliers]({outliers_file})\n")
                f.write("\n\n## Distribution Plot\n")
                if dist_plot_file:
                    f.write(f"![Distribution]({dist_plot_file})\n")
            print("README file created.")
        except Exception as e:
            print(f"Error writing README file: {e}")

    def question_llm(self, prompt, context):
        print("Generating story using LLM...")
        try:
            token = os.environ["AIPROXY_TOKEN"]
            api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
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
                return response.json()["choices"][0]["message"]["content"].strip()
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return "Failed to generate story."
        except Exception as e:
            print(f"Error: {e}")
            return "Failed to generate story."

    def run(self):
        self.load_data()
        self.analyze_data()
        self.detect_outliers()
        heatmap_file, outliers_file, dist_plot_file = self.visualize_data()
        self.create_readme(heatmap_file, outliers_file, dist_plot_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Automated data analysis tool.")
    parser.add_argument("csv_file", help="Path to the CSV file.")
    parser.add_argument("--output_dir", default=".", help="Directory to save outputs.")

    args = parser.parse_args()

    analyzer = DataAnalyzer(csv_file=args.csv_file, output_dir=args.output_dir)
    analyzer.run()
