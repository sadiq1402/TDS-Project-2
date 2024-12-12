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

    def analyze_data(self):
        """Perform generic data analysis."""
        if self.data is None:
            print("Data not loaded.")
            return None

        try:
            # Convert numpy types to native Python types for JSON serialization
            def convert_to_native(val):
                if hasattr(val, "item"):
                    return val.item()
                return val

            analysis = {
                "summary": {
                    k: {k2: convert_to_native(v2) for k2, v2 in v.items()}
                    for k, v in self.data.describe(include="all").to_dict().items()
                },
                "missing_values": {
                    k: convert_to_native(v)
                    for k, v in self.data.isnull().sum().to_dict().items()
                },
                "correlation": (
                    {
                        k: {k2: convert_to_native(v2) for k2, v2 in v.items()}
                        for k, v in self.data.corr(numeric_only=True).to_dict().items()
                    }
                    if len(
                        self.data.select_dtypes(include=["float64", "int64"]).columns
                    )
                    > 1
                    else {}
                ),
            }
            return analysis
        except Exception as e:
            print(f"Error analyzing data: {e}")
            return None

    # def visualize_data(self):
    #     """Create visualizations and save them as a single PNG."""
    #     if self.data is None:
    #         print("Data not loaded.")
    #         return

    #     try:
    #         # Create output directory
    #         output_dir = os.path.splitext(self.csv_file)[0]
    #         os.makedirs(output_dir, exist_ok=True)

    #         # Clear any existing plots
    #         plt.close("all")

    #         # Create a figure with subplots
    #         fig, axes = plt.subplots(3, 2, figsize=(18, 15))

    #         # Correlation heatmap
    #         numeric_cols = self.data.select_dtypes(include=["float64", "int64"]).columns
    #         if len(numeric_cols) > 1:
    #             corr_matrix = self.data[numeric_cols].corr()
    #             sns.heatmap(
    #                 corr_matrix, annot=True, cmap="coolwarm", ax=axes[0, 0], square=True
    #             )
    #             axes[0, 0].set_title("Correlation Heatmap")

    #         # Missing values
    #         missing_values = self.data.isnull().sum()
    #         if missing_values.any():
    #             missing_values[missing_values > 0].plot(kind="bar", ax=axes[0, 1])
    #             axes[0, 1].set_title("Missing Values per Column")
    #             axes[0, 1].tick_params(axis="x", rotation=90)

    #         # Distribution of numeric column
    #         if len(numeric_cols) > 0:
    #             sns.histplot(self.data[numeric_cols[0]], kde=True, ax=axes[1, 0])
    #             axes[1, 0].set_title(f"Distribution of {numeric_cols[0]}")

    #         # Categorical column countplot
    #         categorical_cols = self.data.select_dtypes(
    #             include=["object", "category"]
    #         ).columns
    #         if len(categorical_cols) > 0:
    #             try:
    #                 sns.countplot(x=self.data[categorical_cols[0]], ax=axes[1, 1])
    #                 axes[1, 1].set_title(f"Countplot of {categorical_cols[0]}")
    #                 axes[1, 1].tick_params(axis="x", rotation=90)
    #             except Exception as e:
    #                 print(f"Could not create countplot: {e}")

    #         # Boxplot of numeric column
    #         if len(numeric_cols) > 0:
    #             sns.boxplot(x=self.data[numeric_cols[0]], ax=axes[2, 0])
    #             axes[2, 0].set_title(f"Boxplot of {numeric_cols[0]}")

    #         # Scatterplot of first two numeric columns
    #         if len(numeric_cols) > 1:
    #             sns.scatterplot(
    #                 x=self.data[numeric_cols[0]],
    #                 y=self.data[numeric_cols[1]],
    #                 ax=axes[2, 1],
    #             )
    #             axes[2, 1].set_title(
    #                 f"Scatterplot of {numeric_cols[0]} vs {numeric_cols[1]}"
    #             )

    #         plt.tight_layout()
    #         plt.savefig(os.path.join(output_dir, "visualizations.png"))
    #         print(f"Saved visualizations to {output_dir}/visualizations.png")
    #     except Exception as e:
    #         print(f"Error creating visualizations: {e}")
    #         import traceback

    #         traceback.print_exc()

    def visualize_data(self):
        """Create enhanced visualizations and save them as a single PNG."""
        if self.data is None:
            print("Data not loaded.")
            return

        try:
            # Create output directory
            output_dir = os.path.splitext(self.csv_file)[0]
            os.makedirs(output_dir, exist_ok=True)

            # Clear any existing plots
            plt.close("all")

            # Identify numeric and categorical columns
            numeric_cols = self.data.select_dtypes(include=["float64", "int64"]).columns
            categorical_cols = self.data.select_dtypes(
                include=["object", "category"]
            ).columns

            # Define subplot grid size dynamically based on the number of visualizations
            total_plots = (
                3  # Initial default plots: Heatmap, Missing Values, First Numeric Dist
            )
            if len(categorical_cols) > 0:
                total_plots += 1
            if len(numeric_cols) > 1:
                total_plots += 2  # Scatterplot + Additional plots

            n_rows = (total_plots + 1) // 2
            fig, axes = plt.subplots(n_rows, 2, figsize=(18, n_rows * 5))
            axes = axes.flatten()

            plot_idx = 0

            # Correlation heatmap
            if len(numeric_cols) > 1:
                corr_matrix = self.data[numeric_cols].corr()
                sns.heatmap(
                    corr_matrix,
                    annot=True,
                    cmap="coolwarm",
                    ax=axes[plot_idx],
                    square=True,
                )
                axes[plot_idx].set_title("Correlation Heatmap")
                plot_idx += 1

            # Missing values
            missing_values = self.data.isnull().sum()
            if missing_values.any():
                missing_values[missing_values > 0].plot(
                    kind="bar", ax=axes[plot_idx], color="orange"
                )
                axes[plot_idx].set_title("Missing Values per Column")
                axes[plot_idx].tick_params(axis="x", rotation=90)
                plot_idx += 1

            # Distribution of the first numeric column
            if len(numeric_cols) > 0:
                sns.histplot(
                    self.data[numeric_cols[0]],
                    kde=True,
                    ax=axes[plot_idx],
                    color="blue",
                )
                axes[plot_idx].set_title(f"Distribution of {numeric_cols[0]}")
                plot_idx += 1

            # Countplot for the first categorical column
            if len(categorical_cols) > 0:
                sns.countplot(
                    x=self.data[categorical_cols[0]],
                    ax=axes[plot_idx],
                    palette="viridis",
                )
                axes[plot_idx].set_title(f"Countplot of {categorical_cols[0]}")
                axes[plot_idx].tick_params(axis="x", rotation=90)
                plot_idx += 1

            # Boxplot of the first numeric column
            if len(numeric_cols) > 0:
                sns.boxplot(
                    x=self.data[numeric_cols[0]], ax=axes[plot_idx], color="green"
                )
                axes[plot_idx].set_title(f"Boxplot of {numeric_cols[0]}")
                plot_idx += 1

            # Scatterplot of the first two numeric columns
            if len(numeric_cols) > 1:
                sns.scatterplot(
                    x=self.data[numeric_cols[0]],
                    y=self.data[numeric_cols[1]],
                    ax=axes[plot_idx],
                    hue=(
                        self.data[categorical_cols[0]]
                        if len(categorical_cols) > 0
                        else None
                    ),
                    palette="coolwarm",
                )
                axes[plot_idx].set_title(
                    f"Scatterplot of {numeric_cols[0]} vs {numeric_cols[1]}"
                )
                plot_idx += 1

            # Pair plot for numeric data (optional, as it's computationally expensive)
            if len(numeric_cols) > 2:
                sns.pairplot(
                    self.data[numeric_cols].sample(
                        500 if len(self.data) > 500 else len(self.data)
                    )
                )
                plt.savefig(os.path.join(output_dir, "pairplot.png"))
                print(f"Saved pairplot to {output_dir}/pairplot.png")

            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "visualizations.png"))
            print(f"Saved visualizations to {output_dir}/visualizations.png")

        except Exception as e:
            print(f"Error creating visualizations: {e}")
            import traceback

            traceback.print_exc()

    # def generate_story(self, analysis):
    #     """Use the LLM to generate a story based on the analysis."""
    #     if not analysis or not self.api_key:
    #         print("Analysis data is not available or API key is missing.")
    #         return

    #     headers = {
    #         "Content-Type": "application/json",
    #         "Authorization": f"Bearer {self.api_key}",
    #     }
    #     messages = [
    #         {
    #             "role": "system",
    #             "content": "You are an expert data analyst, proficient at storytelling and deriving actionable insights.",
    #         },
    #         {
    #             "role": "user",
    #             "content": (
    #                 "I have performed an analysis on a dataset. "
    #                 "The following is a detailed summary of my findings: "
    #                 f"{json.dumps(analysis, indent=2)}. "
    #                 "Using this information, craft a comprehensive narrative report highlighting key insights, "
    #                 "significant trends, potential anomalies, and actionable recommendations. "
    #                 "Ensure the tone is professional and suitable for presentation to stakeholders. "
    #                 "Use markdown formatting, including headers, lists, and emphasis where appropriate."
    #             ),
    #         },
    #     ]
    #     payload = {
    #         "model": "gpt-4o-mini",
    #         "messages": messages,
    #     }

    #     try:
    #         response = requests.post(self.api_url, headers=headers, json=payload)
    #         if response.status_code == 200:
    #             story = (
    #                 response.json()
    #                 .get("choices", [])[0]
    #                 .get("message", {})
    #                 .get("content", "")
    #             )
    #             output_dir = os.path.splitext(self.csv_file)[0]
    #             os.makedirs(output_dir, exist_ok=True)
    #             with open(os.path.join(output_dir, "README.md"), "w") as file:
    #                 file.write(story)
    #             print(f"Saved README.md to {output_dir}/README.md")
    #         else:
    #             print(
    #                 f"Error generating story: {response.status_code}, {response.text}"
    #             )
    #     except Exception as e:
    #         print(f"Error in API request: {e}")

    def generate_story(self, analysis):
        """Use the LLM to generate a story based on the analysis with efficient token usage."""
        if not analysis or not self.api_key:
            print("Analysis data is not available or API key is missing.")
            return

        # Generate a concise summary for the prompt
        summary = self._create_summary(analysis)

        # Dynamic and reusable prompt template
        prompt_template = (
            "You are a professional data analyst skilled at deriving actionable insights.\n\n"
            "Here is the dataset analysis summary:\n\n"
            "{summary}\n\n"
            "Please craft a professional report that:\n"
            "- Highlights key trends and insights.\n"
            "- Identifies anomalies or issues (e.g., missing values).\n"
            "- Recommends actionable steps based on the analysis.\n\n"
            "Use markdown formatting with sections, lists, and emphasis."
        )
        prompt = prompt_template.format(summary=summary)

        # Prepare API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are an expert data analyst."},
                {"role": "user", "content": prompt},
            ],
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

    def _create_summary(self, analysis):
        """Create a concise summary of the analysis."""
        try:
            summary_parts = []

            # Include key points from the descriptive statistics
            if "summary" in analysis:
                summary_parts.append(
                    f"Descriptive Summary:\n"
                    f"- Total rows: {self.data.shape[0]}, Total columns: {self.data.shape[1]}\n"
                    f"- Key statistics:\n"
                    f"  {json.dumps({k: analysis['summary'][k] for k in list(analysis['summary'])[:3]}, indent=2)}"
                )

            # Include information about missing values
            if "missing_values" in analysis:
                missing = {k: v for k, v in analysis["missing_values"].items() if v > 0}
                if missing:
                    summary_parts.append(
                        f"Columns with missing values:\n"
                        f"{json.dumps(missing, indent=2)}"
                    )

            # Include correlations
            if "correlation" in analysis and analysis["correlation"]:
                summary_parts.append(
                    "Key correlations detected among numeric features."
                )

            return "\n\n".join(summary_parts)
        except Exception as e:
            print(f"Error creating summary: {e}")
            return "Analysis summary unavailable."

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
