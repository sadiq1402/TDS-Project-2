dependencies = [
    "python-dotenv",
    "matplotlib",
    "requests",
    "chardet",
    "pandas",
    "seaborn",
]

import os
import sys
import json
import matplotlib.pyplot as plt
import requests
import chardet
import pandas as pd
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
        """Perform advanced data analysis with comprehensive insights."""
        if self.data is None:
            print("Data not loaded.")
            return None

        try:
            # Convert numpy types to native Python types for JSON serialization
            def convert_to_native(val):
                if hasattr(val, "item"):
                    return val.item()
                return val

            # Basic statistical summary
            summary = self.data.describe(include='all')

            # Advanced analysis dictionary
            analysis = {
                # Basic statistics summary
                "summary": {
                    k: {k2: convert_to_native(v2) for k2, v2 in v.items()}
                    for k, v in summary.to_dict().items()
                },

                # Missing values analysis
                "missing_values": {
                    "total_missing": convert_to_native(self.data.isnull().sum().sum()),
                    "missing_by_column": {
                        k: convert_to_native(v)
                        for k, v in self.data.isnull().sum().to_dict().items()
                    },
                    "missing_percentage": {
                        k: convert_to_native(round(v / len(self.data) * 100, 2))
                        for k, v in self.data.isnull().sum().to_dict().items()
                    }
                },

                # Correlation analysis
                "correlation": {},

                # Data type information
                "data_types": {
                    k: str(v) for k, v in self.data.dtypes.to_dict().items()
                },

                # Advanced numeric column analysis
                "numeric_analysis": {},

                # Categorical column analysis
                "categorical_analysis": {}
            }

            # Correlation for numeric columns
            numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 1:
                corr_matrix = self.data[numeric_cols].corr()
                analysis["correlation"] = {
                    k: {k2: convert_to_native(v2) for k2, v2 in v.items()}
                    for k, v in corr_matrix.to_dict().items()
                }

                # Identify highly correlated features
                analysis["correlation"]["high_correlations"] = [
                    {"pair": (col1, col2), "correlation": convert_to_native(corr_matrix.loc[col1, col2])}
                    for col1 in corr_matrix.columns
                    for col2 in corr_matrix.columns
                    if col1 != col2 and abs(corr_matrix.loc[col1, col2]) > 0.7
                ]

            # Detailed numeric column analysis
            for col in numeric_cols:
                col_data = self.data[col]
                analysis["numeric_analysis"][col] = {
                    "skewness": convert_to_native(col_data.skew()),
                    "kurtosis": convert_to_native(col_data.kurtosis()),
                    "iqr": convert_to_native(col_data.quantile(0.75) - col_data.quantile(0.25)),
                    "outliers": convert_to_native(
                        len(col_data[
                            (col_data < col_data.quantile(0.25) - 1.5 * (col_data.quantile(0.75) - col_data.quantile(0.25))) |
                            (col_data > col_data.quantile(0.75) + 1.5 * (col_data.quantile(0.75) - col_data.quantile(0.25)))
                        ])
                    )
                }

            # Categorical column analysis
            categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                value_counts = self.data[col].value_counts()
                analysis["categorical_analysis"][col] = {
                    "unique_values": convert_to_native(self.data[col].nunique()),
                    "top_5_values": {
                        str(k): convert_to_native(v)
                        for k, v in value_counts.head().to_dict().items()
                    },
                    "value_distribution_percentage": {
                        str(k): convert_to_native(round(v / len(self.data) * 100, 2))
                        for k, v in value_counts.to_dict().items()
                    }
                }

            return analysis
        except Exception as e:
            print(f"Error analyzing data: {e}")
            import traceback
            traceback.print_exc()
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
