import os
import sys
import subprocess
import importlib.util
import shutil


def install_dependencies():
    """Automatically install required dependencies using uv."""
    dependencies = [
        "python-dotenv==1.0.0",
        "pandas==2.2.1",
        "matplotlib==3.8.3",
        "seaborn==0.13.2",
        "requests==2.31.0",
        "chardet==5.2.0",
        "httpx==0.27.0",
    ]

    # Check if uv is available
    uv_path = shutil.which("uv")
    pip_path = shutil.which("pip")

    install_command = None
    if uv_path:
        install_command = [uv_path, "pip", "install"]
    elif pip_path:
        install_command = [sys.executable, "-m", "pip", "install"]
    else:
        print("Error: Neither uv nor pip found. Cannot install dependencies.")
        sys.exit(1)

    # Check and install missing or incorrect versions of dependencies
    missing_deps = []
    for dep in dependencies:
        try:
            # Split dependency name and version
            package, version = dep.split("==")

            # Check if package is installed
            spec = importlib.util.find_spec(package.replace("-", "_"))
            if spec is None:
                missing_deps.append(dep)

        except Exception as e:
            print(f"Error checking dependency {dep}: {e}")
            missing_deps.append(dep)

    if missing_deps:
        print(f"Installing dependencies: {', '.join(missing_deps)}")
        try:
            subprocess.check_call(install_command + missing_deps)
        except subprocess.CalledProcessError:
            print("Failed to install dependencies.")
            sys.exit(1)


# Install dependencies before importing other modules
install_dependencies()

import json
import requests
import chardet
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv


class AutoLysis:
    def __init__(self, csv_file):
        # Load environment variables
        load_dotenv()

        self.csv_file = csv_file
        self.data = None

        # Use environment variable for API if available, fallback to default
        self.api_url = os.getenv(
            "OPENAI_API_URL",
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
        )
        self.api_key = os.getenv("AIPROXY_TOKEN")

    def detect_encoding(self, file_path):
        """Detect file encoding."""
        try:
            with open(file_path, "rb") as f:
                raw_data = f.read()
            result = chardet.detect(raw_data)
            return result["encoding"] or "utf-8"
        except Exception as e:
            print(f"Error detecting file encoding: {e}")
            return "utf-8"

    def load_data(self):
        """Load the dataset into a DataFrame."""
        if not os.path.exists(self.csv_file):
            print(f"Error: File {self.csv_file} does not exist.")
            return

        try:
            encoding = self.detect_encoding(self.csv_file)
            self.data = pd.read_csv(self.csv_file, encoding=encoding)
            print(
                f"Loaded dataset with {self.data.shape[0]} rows and {self.data.shape[1]} columns."
            )
        except Exception as e:
            print(f"Error loading data: {e}")

    # def analyze_data(self):
    #     """Perform generic data analysis."""
    #     if self.data is None:
    #         print("Data not loaded.")
    #         return None

    #     try:
    #         # Convert numpy types to native Python types for JSON serialization
    #         def convert_to_native(val):
    #             if hasattr(val, "item"):
    #                 return val.item()
    #             return val

    #         analysis = {
    #             "summary": {
    #                 k: {k2: convert_to_native(v2) for k2, v2 in v.items()}
    #                 for k, v in self.data.describe(include="all").to_dict().items()
    #             },
    #             "missing_values": {
    #                 k: convert_to_native(v)
    #                 for k, v in self.data.isnull().sum().to_dict().items()
    #             },
    #             "correlation": (
    #                 {
    #                     k: {k2: convert_to_native(v2) for k2, v2 in v.items()}
    #                     for k, v in self.data.corr(numeric_only=True).to_dict().items()
    #                 }
    #                 if len(
    #                     self.data.select_dtypes(include=["float64", "int64"]).columns
    #                 )
    #                 > 1
    #                 else {}
    #             ),
    #         }
    #         return analysis
    #     except Exception as e:
    #         print(f"Error analyzing data: {e}")
    #         return None
    def analyze_data(self):
        """Perform comprehensive data analysis."""
        if self.data is None:
            print("Data not loaded.")
            return None

        try:
            # Prepare a comprehensive analysis dictionary
            analysis = {
                "dataset_overview": {
                    "total_rows": self.data.shape[0],
                    "total_columns": self.data.shape[1],
                    "columns": list(self.data.columns),
                },
                "column_types": {
                    col: str(self.data[col].dtype) for col in self.data.columns
                },
                "missing_values": {
                    col: {
                        "total_missing": self.data[col].isnull().sum(),
                        "percent_missing": round(
                            self.data[col].isnull().mean() * 100, 2
                        ),
                    }
                    for col in self.data.columns
                },
                "summary_statistics": {},
            }

            # Numeric columns analysis
            numeric_cols = self.data.select_dtypes(include=["int64", "float64"]).columns
            for col in numeric_cols:
                analysis["summary_statistics"][col] = {
                    "mean": round(self.data[col].mean(), 2),
                    "median": round(self.data[col].median(), 2),
                    "std_dev": round(self.data[col].std(), 2),
                    "min": round(self.data[col].min(), 2),
                    "max": round(self.data[col].max(), 2),
                    "skewness": round(self.data[col].skew(), 2),
                    "kurtosis": round(self.data[col].kurtosis(), 2),
                }

            # Categorical columns analysis
            categorical_cols = self.data.select_dtypes(
                include=["object", "category"]
            ).columns
            for col in categorical_cols:
                top_categories = self.data[col].value_counts().head(5)
                analysis["summary_statistics"][col] = {
                    "unique_values": self.data[col].nunique(),
                    "most_frequent": {
                        "value": top_categories.index.tolist(),
                        "count": top_categories.values.tolist(),
                    },
                }

            # Correlation for numeric columns (if more than one numeric column)
            if len(numeric_cols) > 1:
                try:
                    correlation_matrix = self.data[numeric_cols].corr()
                    analysis["correlation_matrix"] = {
                        col: {
                            inner_col: round(correlation_matrix.loc[col, inner_col], 2)
                            for inner_col in numeric_cols
                        }
                        for col in numeric_cols
                    }
                except Exception as e:
                    print(f"Correlation calculation error: {e}")
                    analysis["correlation_matrix"] = "Could not calculate correlation"

            return analysis
        except Exception as e:
            print(f"Error analyzing data: {e}")
            return None

    def visualize_data(self):
        """Create visualizations and save them as a single PNG."""
        if self.data is None:
            print("Data not loaded.")
            return

        try:
            # Create output directory
            output_dir = os.path.splitext(self.csv_file)[0]
            os.makedirs(output_dir, exist_ok=True)

            # Clear any existing plots
            plt.close("all")

            # Create a figure with subplots
            fig, axes = plt.subplots(3, 2, figsize=(18, 15))

            # Correlation heatmap
            numeric_cols = self.data.select_dtypes(include=["float64", "int64"]).columns
            if len(numeric_cols) > 1:
                corr_matrix = self.data[numeric_cols].corr()
                sns.heatmap(
                    corr_matrix, annot=True, cmap="coolwarm", ax=axes[0, 0], square=True
                )
                axes[0, 0].set_title("Correlation Heatmap")

            # Missing values
            missing_values = self.data.isnull().sum()
            if missing_values.any():
                missing_values[missing_values > 0].plot(kind="bar", ax=axes[0, 1])
                axes[0, 1].set_title("Missing Values per Column")
                axes[0, 1].tick_params(axis="x", rotation=90)

            # Distribution of numeric column
            if len(numeric_cols) > 0:
                sns.histplot(self.data[numeric_cols[0]], kde=True, ax=axes[1, 0])
                axes[1, 0].set_title(f"Distribution of {numeric_cols[0]}")

            # Categorical column countplot
            categorical_cols = self.data.select_dtypes(
                include=["object", "category"]
            ).columns
            if len(categorical_cols) > 0:
                try:
                    sns.countplot(x=self.data[categorical_cols[0]], ax=axes[1, 1])
                    axes[1, 1].set_title(f"Countplot of {categorical_cols[0]}")
                    axes[1, 1].tick_params(axis="x", rotation=90)
                except Exception as e:
                    print(f"Could not create countplot: {e}")

            # Boxplot of numeric column
            if len(numeric_cols) > 0:
                sns.boxplot(x=self.data[numeric_cols[0]], ax=axes[2, 0])
                axes[2, 0].set_title(f"Boxplot of {numeric_cols[0]}")

            # Scatterplot of first two numeric columns
            if len(numeric_cols) > 1:
                sns.scatterplot(
                    x=self.data[numeric_cols[0]],
                    y=self.data[numeric_cols[1]],
                    ax=axes[2, 1],
                )
                axes[2, 1].set_title(
                    f"Scatterplot of {numeric_cols[0]} vs {numeric_cols[1]}"
                )

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "visualizations.png"))
            print(f"Saved visualizations to {output_dir}/visualizations.png")
        except Exception as e:
            print(f"Error creating visualizations: {e}")
            import traceback

            traceback.print_exc()

    def generate_story(self, analysis):
        """Use the LLM to generate a story based on the analysis."""
        if not analysis or not self.api_key:
            print("Analysis data is not available or API key is missing.")
            return

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
                    f"{json.dumps(analysis, indent=2)}. "
                    "Using this information, craft a comprehensive narrative report highlighting key insights, "
                    "significant trends, potential anomalies, and actionable recommendations. "
                    "Ensure the tone is professional and suitable for presentation to stakeholders. "
                    "Use markdown formatting, including headers, lists, and emphasis where appropriate."
                ),
            },
        ]
        payload = {
            "model": "gpt-4o-mini",
            "messages": messages,
        }

        try:
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
                print(
                    f"Error generating story: {response.status_code}, {response.text}"
                )
        except Exception as e:
            print(f"Error in API request: {e}")

    def run(self):
        """Execute the workflow."""
        self.load_data()
        analysis = self.analyze_data()
        self.visualize_data()
        if analysis:
            self.generate_story(analysis)


if __name__ == "__main__":
    # Check if a dataset is provided
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    # Get the CSV file from command-line argument
    csv_file = sys.argv[1]

    # Create an instance of AutoLysis and run the analysis
    auto_lysis = AutoLysis(csv_file)
    auto_lysis.run()
