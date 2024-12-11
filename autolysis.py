import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
from chardet import detect


class AutoLysis:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data = None
        self.api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        self.api_key = os.environ.get("AIPROXY_TOKEN")

    def detect_encoding(self, file_path):
        """Detect file encoding."""
        with open(file_path, "rb") as f:
            raw_data = f.read()
        result = detect(raw_data)
        return result["encoding"]

    def load_data(self):
        """Load the dataset into a DataFrame."""
        try:
            encoding = self.detect_encoding(self.csv_file)
            self.data = pd.read_csv(self.csv_file, encoding=encoding)
            print(
                f"Loaded dataset with {self.data.shape[0]} rows and {self.data.shape[1]} columns."
            )
        except Exception as e:
            print(f"Error loading data: {e}")

    def analyze_data(self):
        """Perform generic data analysis."""
        if self.data is None:
            print("Data not loaded.")
            return

        analysis = {
            "summary": self.data.describe(include="all").to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "correlation": self.data.corr(numeric_only=True).to_dict(),
        }
        return analysis

    def visualize_data(self):
        """Create visualizations and save them as a single PNG."""
        if self.data is None:
            print("Data not loaded.")
            return

        fig, axes = plt.subplots(3, 2, figsize=(18, 15))

        # Visualization 1: Correlation heatmap
        sns.heatmap(
            self.data.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm",
            ax=axes[0, 0],
        )
        axes[0, 0].set_title("Correlation Heatmap")

        # Visualization 2: Missing values
        missing_values = self.data.isnull().sum()
        if missing_values.any():
            sns.barplot(x=missing_values.index, y=missing_values.values, ax=axes[0, 1])
            axes[0, 1].set_title("Missing Values per Column")
            axes[0, 1].tick_params(axis="x", rotation=90)

        # Visualization 3: Distribution of a numeric column
        numeric_columns = self.data.select_dtypes(include=["float64", "int64"]).columns
        if not numeric_columns.empty:
            sns.histplot(self.data[numeric_columns[0]], kde=True, ax=axes[1, 0])
            axes[1, 0].set_title(f"Distribution of {numeric_columns[0]}")

        # Visualization 4: Countplot for the first categorical column
        categorical_columns = self.data.select_dtypes(
            include=["object", "category"]
        ).columns
        if not categorical_columns.empty:
            sns.countplot(x=self.data[categorical_columns[0]], ax=axes[1, 1])
            axes[1, 1].set_title(f"Countplot of {categorical_columns[0]}")
            axes[1, 1].tick_params(axis="x", rotation=90)

        # Visualization 5: Boxplot of a numeric column
        if len(numeric_columns) > 1:
            sns.boxplot(data=self.data, x=numeric_columns[0], ax=axes[2, 0])
            axes[2, 0].set_title(f"Boxplot of {numeric_columns[0]}")

        # Visualization 6: Scatterplot of first two numeric columns
        if len(numeric_columns) > 1:
            sns.scatterplot(
                x=self.data[numeric_columns[0]],
                y=self.data[numeric_columns[1]],
                ax=axes[2, 1],
            )
            axes[2, 1].set_title(
                f"Scatterplot of {numeric_columns[0]} vs {numeric_columns[1]}"
            )

        plt.tight_layout()
        output_dir = os.path.splitext(self.csv_file)[0]
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "visualizations.png"))
        print(f"Saved visualizations to {output_dir}/visualizations.png")

    def generate_story(self, analysis):
        """Use the LLM to generate a story based on the analysis."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        messages = [
            {
                "role": "system",
                "content": "You are an expert data analyst, proficient at storytelling and deriving actionable insights.",
            },
            {
                "role": "user",
                "content": (
                    "I have performed an analysis on a dataset. "
                    "The following is a detailed summary of my findings: "
                    f"{json.dumps(analysis)}. "
                    "Using this information, craft a comprehensive narrative report highlighting key insights, "
                    "significant trends, potential anomalies, and actionable recommendations. "
                    "Ensure the tone is professional and suitable for presentation to stakeholders."
                ),
            },
        ]
        payload = {
            "model": "gpt-4o-mini",
            "messages": messages,
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        if response.status_code == 200:
            story = (
                response.json()
                .get("choices", [])[0]
                .get("message", {})
                .get("content", "")
            )
            output_dir = os.path.splitext(self.csv_file)[0]
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "README.md"), "w") as file:
                file.write(story)
            print(f"Saved README.md to {output_dir}/README.md")
        else:
            print(f"Error generating story: {response.status_code}, {response.text}")

    def run(self):
        """Execute the workflow."""
        self.load_data()
        analysis = self.analyze_data()
        self.visualize_data()
        if analysis:
            self.generate_story(analysis)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    csv_file = sys.argv[1]
    auto_lysis = AutoLysis(csv_file)
    auto_lysis.run()
