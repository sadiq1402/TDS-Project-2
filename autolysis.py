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
import sys
import json
import traceback
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class DataAnalyzer:
    def __init__(self, csv_path):
        # Validate input
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"The file {csv_path} does not exist.")

        # Try multiple encodings
        encodings_to_try = ["utf-8", "latin-1", "iso-8859-1", "cp1252", "utf-8-sig"]

        for encoding in encodings_to_try:
            try:
                # Load data with the current encoding
                self.df = pd.read_csv(csv_path, encoding=encoding)
                break  # If successful, exit the loop
            except UnicodeDecodeError:
                continue
            except Exception as e:
                # If any other unexpected error occurs
                raise ValueError(f"Could not read the CSV file with any encoding: {e}")
        else:
            # If no encoding worked
            raise ValueError(
                "Could not read the CSV file with any of the attempted encodings"
            )

        # Determine output folder name based on CSV filename
        self.output_folder = self._get_output_folder(csv_path)

        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

        # Store original columns and path
        self.original_columns = self.df.columns.tolist()
        self.csv_path = csv_path

    def _get_output_folder(self, csv_path):
        """
        Generate output folder name based on CSV filename
        Handles various naming scenarios
        """
        # Extract filename without extension
        base_filename = os.path.splitext(os.path.basename(csv_path))[0]

        # Convert to lowercase, remove special characters
        folder_name = "".join(c if c.isalnum() else "_" for c in base_filename.lower())

        # Ensure folder name is valid
        if not folder_name:
            folder_name = "dataset_analysis"

        return folder_name

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

    # def generate_visualizations(self, analyses):
    #     """Create visualizations with error handling"""
    #     plt.figure(figsize=(15, 5))
    #     plt.subplots_adjust(wspace=0.3)

    #     # Correlation Heatmap
    #     try:
    #         if "correlation" in analyses:
    #             plt.subplot(131)
    #             corr_matrix = np.array(analyses["correlation"]["matrix"])
    #             sns.heatmap(
    #                 corr_matrix,
    #                 annot=True,
    #                 cmap="coolwarm",
    #                 linewidths=0.5,
    #                 xticklabels=analyses["correlation"]["columns"],
    #                 yticklabels=analyses["correlation"]["columns"],
    #             )
    #             plt.title("Correlation Heatmap")
    #     except Exception as e:
    #         print(f"Correlation visualization error: {e}")

    #     # Clustering Visualization
    #     try:
    #         if "clustering" in analyses:
    #             plt.subplot(132)
    #             numeric_data = self.df.select_dtypes(include=[np.number])
    #             scaler = StandardScaler()
    #             scaled_data = scaler.fit_transform(numeric_data)
    #             plt.scatter(
    #                 scaled_data[:, 0],
    #                 scaled_data[:, 1],
    #                 c=analyses["clustering"]["labels"],
    #                 cmap="viridis",
    #             )
    #             plt.title("Clustering Visualization")
    #     except Exception as e:
    #         print(f"Clustering visualization error: {e}")

    #     # PCA Visualization
    #     try:
    #         if "pca" in analyses:
    #             plt.subplot(133)
    #             pca_result = np.array(analyses["pca"]["components"])
    #             plt.scatter(pca_result[:, 0], pca_result[:, 1])
    #             plt.title("PCA Visualization")
    #             if "explained_variance" in analyses["pca"]:
    #                 var = analyses["pca"]["explained_variance"]
    #                 plt.xlabel(f"PC1 ({var[0]:.2%})")
    #                 plt.ylabel(f"PC2 ({var[1]:.2%})")
    #     except Exception as e:
    #         print(f"PCA visualization error: {e}")

    #     # Save the visualization in the output folder
    #     visualization_path = os.path.join(
    #         self.output_folder, "analysis_visualizations.png"
    #     )
    #     plt.tight_layout()
    #     plt.savefig(visualization_path, dpi=300)
    #     plt.close()
    def generate_visualizations(self, analyses):
        """Create comprehensive visualizations with enhanced insights"""
        plt.figure(figsize=(20, 10))
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        # Correlation Heatmap
        try:
            if "correlation" in analyses:
                plt.subplot(221)
                corr_matrix = np.array(analyses["correlation"]["matrix"])
                sns.heatmap(
                    corr_matrix,
                    annot=True,
                    cmap="coolwarm",
                    linewidths=0.5,
                    xticklabels=analyses["correlation"]["columns"],
                    yticklabels=analyses["correlation"]["columns"],
                    fmt=".2f",
                )
                plt.title("Correlation Heatmap")
        except Exception as e:
            print(f"Error generating correlation heatmap: {e}")

        # Clustering Visualization with Centroids
        try:
            if "clustering" in analyses:
                plt.subplot(222)
                numeric_data = self.df.select_dtypes(include=[np.number])
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_data)

                cluster_labels = np.array(analyses["clustering"]["labels"])
                unique_labels = set(cluster_labels)

                for label in unique_labels:
                    cluster_points = scaled_data[cluster_labels == label]
                    plt.scatter(
                        cluster_points[:, 0],
                        cluster_points[:, 1],
                        label=f"Cluster {label}",
                        alpha=0.6,
                    )

                # Plot cluster centroids if available
                kmeans = KMeans(n_clusters=len(unique_labels), random_state=42)
                kmeans.fit(scaled_data)
                centroids = kmeans.cluster_centers_
                plt.scatter(
                    centroids[:, 0],
                    centroids[:, 1],
                    s=200,
                    c="black",
                    marker="X",
                    label="Centroids",
                )

                plt.title("Cluster Analysis with Centroids")
                plt.legend()
        except Exception as e:
            print(f"Error generating clustering visualization: {e}")

        # PCA Scatter Plot
        try:
            if "pca" in analyses:
                plt.subplot(223)
                pca_result = np.array(analyses["pca"]["components"])
                plt.scatter(
                    pca_result[:, 0],
                    pca_result[:, 1],
                    alpha=0.6,
                    c="blue",
                    label="Data",
                )

                if "explained_variance" in analyses["pca"]:
                    explained_var = analyses["pca"]["explained_variance"]
                    plt.xlabel(f"PC1 ({explained_var[0]:.2%})")
                    plt.ylabel(f"PC2 ({explained_var[1]:.2%})")

                plt.title("PCA Scatter Plot")
                plt.legend()
        except Exception as e:
            print(f"Error generating PCA visualization: {e}")

        # Distribution Plots
        try:
            numeric_data = self.df.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                plt.subplot(224)
                for col in numeric_data.columns[:4]:  # Limit to the first 4 columns
                    sns.kdeplot(
                        numeric_data[col],
                        label=col,
                        fill=True,
                        alpha=0.5,
                    )
                plt.title("Distribution of Key Numeric Features")
                plt.legend()
        except Exception as e:
            print(f"Error generating distribution plots: {e}")

        # Save the visualization as a single PNG file
        visualization_path = os.path.join(
            self.output_folder, "analysis_visualizations.png"
        )
        plt.tight_layout()
        plt.savefig(visualization_path, dpi=300)
        plt.close()
        print(f"Visualizations saved to {visualization_path}")

    def generate_narrative(self, data_summary, analyses):
        """Generate a comprehensive, deeply insightful narrative based on data analysis"""
        try:
            # Comprehensive Data Profiling
            column_count = len(data_summary.get("columns", []))
            total_rows = data_summary.get("total_rows", "N/A")

            # Sophisticated Column Classification
            column_types = data_summary.get("column_types", {})
            numeric_cols = [
                col
                for col, dtype in column_types.items()
                if "int" in str(dtype).lower() or "float" in str(dtype).lower()
            ]
            categorical_cols = [
                col
                for col, dtype in column_types.items()
                if "object" in str(dtype).lower()
            ]
            datetime_cols = [
                col
                for col, dtype in column_types.items()
                if "datetime" in str(dtype).lower()
            ]

            # Advanced Missing Data Analysis
            missing_values = data_summary.get("missing_values", {})
            missing_columns = [
                col for col, count in missing_values.items() if count > 0
            ]
            missing_percentage = {
                col: round(missing_values[col] / total_rows * 100, 2)
                for col in missing_columns
            }

            # Correlation Deep Dive
            correlation_insights = ""
            correlation_details = []
            if "correlation" in analyses and "matrix" in analyses["correlation"]:
                correlation_matrix = analyses["correlation"]["matrix"]
                corr_columns = analyses["correlation"]["columns"]

                # Comprehensive correlation analysis
                for i in range(len(corr_columns)):
                    for j in range(i + 1, len(corr_columns)):
                        correlation = correlation_matrix[i][j]
                        if abs(correlation) > 0.5:  # Significant correlation threshold
                            correlation_type = (
                                "Strong Positive"
                                if correlation > 0.8
                                else (
                                    "Moderate Positive"
                                    if correlation > 0.5
                                    else (
                                        "Strong Negative"
                                        if correlation < -0.8
                                        else "Moderate Negative"
                                    )
                                )
                            )
                            correlation_details.append(
                                {
                                    "col1": corr_columns[i],
                                    "col2": corr_columns[j],
                                    "correlation": correlation,
                                    "type": correlation_type,
                                }
                            )

                # Sort correlations by absolute strength
                correlation_details.sort(
                    key=lambda x: abs(x["correlation"]), reverse=True
                )

                # Generate correlation narrative
                correlation_insights = "### Correlation Analysis\n"
                for detail in correlation_details[:5]:  # Top 5 correlations
                    correlation_insights += (
                        f"- **{detail['col1']}** and **{detail['col2']}**: "
                        f"{detail['type']} correlation (r = {detail['correlation']:.2f})\n"
                    )

            # Clustering Insights
            clustering_insights = ""
            if "clustering" in analyses:
                cluster_labels = analyses["clustering"]["labels"]
                n_clusters = len(set(cluster_labels))
                cluster_distribution = {}
                for label in set(cluster_labels):
                    cluster_distribution[label] = cluster_labels.count(label)

                clustering_insights = "### Advanced Clustering Analysis\n"
                clustering_insights += (
                    f"- Identified {n_clusters} distinct data clusters\n"
                )
                clustering_insights += "- Cluster Distribution:\n"
                for cluster, count in cluster_distribution.items():
                    percentage = round(count / len(cluster_labels) * 100, 2)
                    clustering_insights += (
                        f"  * Cluster {cluster}: {count} data points ({percentage}%)\n"
                    )

            # PCA Advanced Analysis
            pca_insights = ""
            if "pca" in analyses:
                explained_variance = analyses["pca"].get("explained_variance", [])
                if explained_variance:
                    pca_insights = "### Dimensionality Reduction (PCA) Insights\n"
                    cumulative_variance = 0
                    for i, variance in enumerate(explained_variance, 1):
                        cumulative_variance += variance
                        pca_insights += (
                            f"- Principal Component {i}: "
                            f"Explains {variance*100:.2f}% of variance\n"
                        )
                    pca_insights += f"- Cumulative Variance Explained: {cumulative_variance*100:.2f}%\n"

            # Numeric Summary Deep Dive
            numeric_summary_insights = ""
            numeric_summary = data_summary.get("numeric_summary", {})
            if numeric_summary:
                numeric_summary_insights = "### Comprehensive Numeric Analysis\n"
                for col, stats in numeric_summary.items():
                    # Identify distributions and outlier potential
                    iqr = stats["max"] - stats["min"]
                    std_dev = stats["std"]
                    outlier_potential = (
                        "Low"
                        if std_dev < iqr * 0.5
                        else "Moderate" if std_dev < iqr else "High"
                    )

                    numeric_summary_insights += (
                        f"#### {col} Statistical Profile\n"
                        f"- **Central Tendency**: Mean = {stats['mean']:.2f}, Median = {stats['median']:.2f}\n"
                        f"- **Variability**: Std Dev = {std_dev:.2f}, Range = {stats['min']} to {stats['max']}\n"
                        f"- **Outlier Potential**: {outlier_potential}\n"
                    )

            # Final Narrative Composition
            narrative = f"""# Comprehensive Data Analysis Report for {os.path.basename(self.csv_path)}

    ## Dataset Architectural Overview
    - **Total Records**: {total_rows}
    - **Columns**: {column_count}
    * Numeric Columns: {len(numeric_cols)}
    * Categorical Columns: {len(categorical_cols)}
    * Datetime Columns: {len(datetime_cols)}

    ## Data Integrity Assessment
    ### Missing Data Analysis
    {chr(10).join(f"- **{col}**: {missing_values[col]} missing entries ({missing_percentage[col]}%)" for col in missing_columns) if missing_columns else "- No missing values detected"}

    {numeric_summary_insights}

    {correlation_insights}

    {clustering_insights}

    {pca_insights}

    ## Strategic Insights and Recommendations
    - Automated analysis reveals complex data characteristics
    - Significant correlations and clustering patterns identified
    - Visualizations available in analysis_visualizations.png
    - Recommended next steps:
    1. Validate key correlations with domain expertise
    2. Investigate clusters for potential subgroup analysis
    3. Consider feature engineering based on PCA insights
    4. Address missing data if impactful to analysis

    **Interpretative Note**: Algorithmic insights provide a starting point. Domain-specific validation is crucial for comprehensive understanding.
    """

            # Save and output
            readme_path = os.path.join(self.output_folder, "README.md")
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(narrative)

            print(f"Ultra-detailed narrative generated in {readme_path}")

        except Exception as e:
            error_message = f"""# Analysis Report Generation Failed

    ## Critical Error Details
    - Error: {str(e)}
    - Traceback: {traceback.format_exc()}

    ### Available Data Snapshot
    - Data Summary: {json.dumps(data_summary, indent=2)}
    - Analyses: {json.dumps(analyses, indent=2)}
    """
            # Save error narrative
            readme_path = os.path.join(self.output_folder, "README.md")
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(error_message)

            print(f"Narrative generation encountered a critical error: {e}")


def main(csv_path):
    try:
        analyzer = DataAnalyzer(csv_path)
        data_summary = analyzer.generate_data_summary()
        analyses = analyzer.perform_analysis()
        analyzer.generate_visualizations(analyses)
        analyzer.generate_narrative(data_summary, analyses)
        print(f"Analysis complete. Check {analyzer.output_folder} folder.")
    except Exception as e:
        print(f"Comprehensive error during analysis:")
        print(traceback.format_exc())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <path_to_csv>")
        sys.exit(1)

    main(sys.argv[1])
