# ///script

# Dependencies = [
#     "pandas",
#     "numpy",
#     "scikit-learn",
#     "seaborn",
#     "matplotlib"
#     ]
# ///

import os
import sys
import json
import traceback
import warnings
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer


class DataAnalyzer:
    """
    A comprehensive data analysis class that provides advanced
    data processing, statistical analysis, and reporting capabilities.

    Supports adaptive analysis based on dataset characteristics.
    """

    def __init__(self, csv_path: str, encoding_priority: List[str] = None):
        """
        Initialize the DataAnalyzer with robust file loading.

        Args:
            csv_path (str): Path to the input CSV file
            encoding_priority (List[str], optional): Custom encoding priorities
        """
        # Configure warning suppression for cleaner output
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Default encoding priority with extended options
        self.encoding_priority = encoding_priority or [
            "utf-8",
            "utf-8-sig",
            "latin-1",
            "iso-8859-1",
            "cp1252",
            "windows-1252",
        ]

        # Load the CSV with robust encoding detection
        self.df = self._load_csv(csv_path)
        self.csv_path = csv_path
        self.output_folder = self._create_output_folder(csv_path)

        # Pre-process data immediately after loading
        self._preprocess_data()

    def _load_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Robustly load CSV with multiple encoding attempts.

        Args:
            csv_path (str): Path to CSV file

        Returns:
            pd.DataFrame: Loaded dataframe

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If no encoding works
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        for encoding in self.encoding_priority:
            try:
                return pd.read_csv(csv_path, encoding=encoding)
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue

        raise ValueError("Unable to read CSV with any attempted encoding")

    def _preprocess_data(self):
        """
        Advanced data preprocessing with adaptive strategies.
        """
        # Handle missing values adaptively
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=["object"]).columns

        # Robust imputation strategies
        if len(numeric_cols) > 0:
            imputer = SimpleImputer(strategy="median")
            self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])

        if len(categorical_cols) > 0:
            imputer = SimpleImputer(strategy="most_frequent")
            self.df[categorical_cols] = imputer.fit_transform(self.df[categorical_cols])

    def _create_output_folder(self, csv_path: str) -> str:
        """
        Create a sanitized output folder name based on CSV filename.

        Args:
            csv_path (str): Path to input CSV

        Returns:
            str: Sanitized folder name
        """
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        sanitized_name = "".join(c if c.isalnum() else "_" for c in base_name.lower())
        folder_name = sanitized_name or "dataset_analysis"

        output_path = os.path.join(os.getcwd(), folder_name)
        os.makedirs(output_path, exist_ok=True)
        return output_path

    def generate_advanced_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive, multi-dimensional dataset summary.

        Returns:
            Dict containing detailed dataset insights
        """
        summary = {
            "metadata": {
                "total_rows": len(self.df),
                "total_columns": len(self.df.columns),
                "file_path": self.csv_path,
            },
            "column_analysis": {},
            "missing_data": {},
            "data_distribution": {},
        }

        # Detailed column-level analysis
        for column in self.df.columns:
            col_summary = {
                "dtype": str(self.df[column].dtype),
                "unique_values": self.df[column].nunique(),
                "missing_count": self.df[column].isnull().sum(),
                "missing_percentage": round(self.df[column].isnull().mean() * 100, 2),
            }

            # Additional numeric column insights
            if pd.api.types.is_numeric_dtype(self.df[column]):
                col_summary.update(
                    {
                        "min": self.df[column].min(),
                        "max": self.df[column].max(),
                        "mean": round(self.df[column].mean(), 2),
                        "median": round(self.df[column].median(), 2),
                        "std_dev": round(self.df[column].std(), 2),
                        "skewness": round(self.df[column].skew(), 2),
                        "kurtosis": round(self.df[column].kurtosis(), 2),
                    }
                )

            # Categorical column insights
            if pd.api.types.is_object_dtype(self.df[column]):
                top_categories = self.df[column].value_counts().head(5).to_dict()
                col_summary["top_categories"] = top_categories

            summary["column_analysis"][column] = col_summary

        return summary

    def perform_advanced_analysis(self) -> Dict[str, Any]:
        """
        Conduct sophisticated multi-dimensional data analysis.

        Returns:
            Dict containing various advanced analytical insights
        """
        analyses = {
            "correlation": self._compute_correlations(),
            "feature_importance": self._compute_feature_importance(),
            "clustering": self._perform_clustering(),
            "dimensionality_reduction": self._compute_dimensionality_reduction(),
        }
        return analyses

    def _compute_correlations(self) -> Dict[str, Any]:
        """
        Compute advanced correlation metrics.

        Returns:
            Correlation analysis results
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) <= 1:
            return {"error": "Insufficient numeric columns for correlation analysis"}

        # Compute multiple correlation matrices
        pearson_corr = self.df[numeric_cols].corr(method="pearson")
        spearman_corr = self.df[numeric_cols].corr(method="spearman")

        return {
            "pearson_correlation": pearson_corr.values.tolist(),
            "spearman_correlation": spearman_corr.values.tolist(),
            "columns": list(numeric_cols),
        }

    def _compute_feature_importance(self) -> Dict[str, float]:
        """
        Compute feature importance using mutual information.

        Returns:
            Dict of feature importance scores
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) <= 1:
            return {"error": "Insufficient numeric columns for feature importance"}

        # Use a reference column (last column) as target
        target = numeric_cols[-1]
        features = numeric_cols[:-1]

        X = self.df[features]
        y = self.df[target]

        # Compute mutual information
        importances = mutual_info_regression(X, y)
        return dict(zip(features, importances))

    def _perform_clustering(self) -> Dict[str, Any]:
        """
        Advanced clustering with adaptive techniques.

        Returns:
            Clustering analysis results
        """
        numeric_data = self.df.select_dtypes(include=[np.number])
        if len(numeric_data.columns) <= 1:
            return {"error": "Insufficient numeric columns for clustering"}

        # Scale data robustly
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(numeric_data)

        # Adaptive clustering: DBSCAN for adaptive cluster detection
        dbscan = DBSCAN(eps=0.5, min_samples=max(2, len(scaled_data) // 10))
        cluster_labels = dbscan.fit_predict(scaled_data)

        return {
            "labels": cluster_labels.tolist(),
            "n_clusters": len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            "columns": list(numeric_data.columns),
        }

    def _compute_dimensionality_reduction(self) -> Dict[str, Any]:
        """
        Advanced dimensionality reduction with comprehensive insights.

        Returns:
            PCA and t-SNE reduction results
        """
        numeric_data = self.df.select_dtypes(include=[np.number])
        if len(numeric_data.columns) <= 1:
            return {
                "error": "Insufficient numeric columns for dimensionality reduction"
            }

        # Scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)

        # PCA
        pca = PCA(n_components=min(2, len(numeric_data.columns)))
        pca_result = pca.fit_transform(scaled_data)

        return {
            "pca_components": pca_result.tolist(),
            "explained_variance": pca.explained_variance_ratio_.tolist(),
            "columns": list(numeric_data.columns),
        }

    def generate_advanced_visualizations(
        self, summary: Dict[str, Any], analyses: Dict[str, Any]
    ):
        """
        Create comprehensive, informative visualizations.

        Args:
            summary (Dict): Dataset summary
            analyses (Dict): Analysis results
        """
        plt.figure(figsize=(20, 15))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)

        # Color palette for consistent visualization
        palette = sns.color_palette("coolwarm", as_cmap=True)

        # 1. Correlation Heatmap (Improved)
        plt.subplot(2, 3, 1)
        if (
            "correlation" in analyses
            and "pearson_correlation" in analyses["correlation"]
        ):
            corr_matrix = np.array(analyses["correlation"]["pearson_correlation"])
            columns = analyses["correlation"]["columns"]

            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap=palette,
                linewidths=0.5,
                fmt=".2f",
                square=True,
                cbar_kws={"shrink": 0.8},
                xticklabels=columns,
                yticklabels=columns,
            )
            plt.title("Pearson Correlation Heatmap", fontsize=10)
            plt.tight_layout()

        # 2. Feature Importance
        plt.subplot(2, 3, 2)
        if "feature_importance" in analyses:
            features = list(analyses["feature_importance"].keys())
            importances = list(analyses["feature_importance"].values())

            plt.bar(
                features, importances, color=palette(np.linspace(0, 1, len(features)))
            )
            plt.title("Feature Importance", fontsize=10)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

        # 3. Clustering Visualization
        plt.subplot(2, 3, 3)
        if "clustering" in analyses and "labels" in analyses["clustering"]:
            cluster_data = analyses["dimensionality_reduction"]["pca_components"]
            cluster_labels = analyses["clustering"]["labels"]

            scatter = plt.scatter(
                [point[0] for point in cluster_data],
                [point[1] for point in cluster_data],
                c=cluster_labels,
                cmap="viridis",
                alpha=0.7,
            )
            plt.colorbar(scatter, label="Cluster")
            plt.title("Clustering Visualization", fontsize=10)
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            plt.tight_layout()

        # 4. Distribution of Numeric Columns
        plt.subplot(2, 3, 4)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            sns.boxplot(data=self.df[numeric_cols], palette=palette)
            plt.title("Numeric Columns Distribution", fontsize=10)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

        # 5. Missing Data Visualization
        plt.subplot(2, 3, 5)
        missing_data = self.df.isnull().sum()
        missing_data = missing_data[missing_data > 0]

        if len(missing_data) > 0:
            missing_data.plot(kind="bar", color=palette(0.5))
            plt.title("Missing Data per Column", fontsize=10)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

        # 6. PCA Variance Explained
        plt.subplot(2, 3, 6)
        if "dimensionality_reduction" in analyses:
            variance_ratio = analyses["dimensionality_reduction"]["explained_variance"]
            plt.bar(
                range(1, len(variance_ratio) + 1),
                variance_ratio,
                color=palette(np.linspace(0, 1, len(variance_ratio))),
            )
            plt.title("PCA: Variance Explained", fontsize=10)
            plt.xlabel("Principal Components")
            plt.ylabel("Variance Ratio")
            plt.tight_layout()

        # Save the visualization
        visualization_path = os.path.join(
            self.output_folder, "advanced_analysis_visualization.png"
        )
        plt.savefig(visualization_path, dpi=300, bbox_inches="tight")
        plt.close()

    def generate_comprehensive_narrative(
        self, summary: Dict[str, Any], analyses: Dict[str, Any]
    ) -> str:
        """
        Generate a sophisticated, insights-driven narrative report.

        Args:
            summary (Dict): Dataset summary
            analyses (Dict): Analysis results

        Returns:
            str: Comprehensive markdown narrative
        """

        def format_numeric(value, precision=2):
            """Helper to format numeric values consistently."""
            return round(value, precision) if isinstance(value, (int, float)) else value

        # Dataset Overview
        metadata = summary.get("metadata", {})
        column_analysis = summary.get("column_analysis", {})

        # Correlation Insights
        correlation_insights = ""
        if "correlation" in analyses:
            corr_columns = analyses["correlation"]["columns"]
            pearson_corr = np.array(analyses["correlation"]["pearson_correlation"])

            # Find top significant correlations
            significant_correlations = []
            for i in range(len(corr_columns)):
                for j in range(i + 1, len(corr_columns)):
                    if abs(pearson_corr[i][j]) > 0.5:
                        significant_correlations.append(
                            {
                                "col1": corr_columns[i],
                                "col2": corr_columns[j],
                                "correlation": pearson_corr[i][j],
                            }
                        )

            # Sort correlations by absolute value
            significant_correlations.sort(
                key=lambda x: abs(x["correlation"]), reverse=True
            )

            correlation_insights = "### Correlation Analysis\n"
            for corr in significant_correlations[:5]:
                correlation_insights += (
                    f"- **{corr['col1']}** and **{corr['col2']}**: "
                    f"Correlation of {format_numeric(corr['correlation'])}\n"
                )

        # Clustering Insights
        clustering_insights = ""
        if "clustering" in analyses:
            n_clusters = analyses["clustering"].get("n_clusters", 0)
            clustering_insights = f"### Clustering Analysis\n- Identified {n_clusters} distinct data clusters\n"

        # PCA Insights
        pca_insights = ""
        if "dimensionality_reduction" in analyses:
            variance_explained = analyses["dimensionality_reduction"][
                "explained_variance"
            ]
            cumulative_variance = sum(variance_explained)
            pca_insights = (
                "### Dimensionality Reduction\n"
                f"- First {len(variance_explained)} principal components explain "
                f"{format_numeric(cumulative_variance * 100, 2)}% of total variance\n"
            )

        # Comprehensive Narrative
        narrative = f"""# Comprehensive Data Analysis Report

## Dataset Overview
- **Total Records**: {metadata.get('total_rows', 'N/A')}
- **Total Columns**: {metadata.get('total_columns', 'N/A')}
- **Source**: {metadata.get('file_path', 'Unknown')}

## Data Composition
{correlation_insights}

{clustering_insights}

{pca_insights}

## Strategic Recommendations
1. Validate key correlations with domain expertise
2. Investigate clusters for potential subgroup analysis
3. Consider feature engineering based on correlation and PCA insights

**Note**: This analysis provides algorithmic insights. Domain-specific validation is crucial for comprehensive understanding.
"""

        # Save narrative
        narrative_path = os.path.join(self.output_folder, "comprehensive_analysis.md")
        with open(narrative_path, "w", encoding="utf-8") as f:
            f.write(narrative)

        return narrative


def main(csv_path):
    """
    Main execution function for the Autolysis data analysis tool.

    Args:
        csv_path (str): Path to input CSV file
    """
    try:
        # Initialize analyzer
        analyzer = DataAnalyzer(csv_path)

        # Generate comprehensive summary
        summary = analyzer.generate_advanced_summary()

        # Perform advanced analysis
        analyses = analyzer.perform_advanced_analysis()

        # Generate visualizations
        analyzer.generate_advanced_visualizations(summary, analyses)

        # Generate narrative report
        analyzer.generate_comprehensive_narrative(summary, analyses)

        print(f"Analysis complete. Check {analyzer.output_folder} for results.")

    except Exception as e:
        print(f"Analysis failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <path_to_csv>")
        sys.exit(1)

    main(sys.argv[1])
