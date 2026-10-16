import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Dict
from langchain_core.tools import tool

def tools_for_dataframe(df: pd.DataFrame):
    @tool
    def statistics_and_correlation_tool() -> Dict:
        """Generates the statistical report and the correlation report ofthe numrical columns present in the dataset."""

        numeric_df = df.select_dtypes(include=['int32', 'float32', 'int64', 'float64'])
        numeric_count = len(numeric_df.columns)

        if numeric_count:
            stats_result = numeric_df.describe().to_dict()
            corr_result = numeric_df.corr().to_dict()

            return {
                "Statistics" : stats_result,
                "Correlation" : corr_result
            }
        
        else:
            return {'Status' : 'No statistics and correlation can be made.'}

    @tool
    def anomaly_detection_tool(column: str) -> dict:
        """Detecting outliers in a single column"""

        if column not in df.columns or df[column].dtype not in ['int32', 'int64', 'float32', 'float64']:
            return {"error": f"Column '{column}' is invalid or non-numeric."}
        
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr
        
        outliers = df[(df[column] < lower_limit) | (df[column] > upper_limit)]
        return {
            "outlier_count": len(outliers),
            "lower_bound": lower_limit,
            "upper_bound": upper_limit,
            "sample_outliers": outliers.head(5).to_dict(orient="records")
        }

    @tool
    def grouping_tool(group_col: str, target_col: str, agg: str = "mean") -> Dict:
        """Groups data by a categorical column and sums it and compares a target numeric column."""

        if group_col not in df.columns or target_col not in df.columns:
            return {"error": "Provided columns not found in dataset."}
        
        grouped_df = df.groupby(group_col)[target_col].agg(agg).reset_index()

        return grouped_df.to_dict(orient="records")

    @tool
    def data_visualization_tool(type: str, x_axis: str, y_axis: str = None) -> dict:
        """Generates a visualization plot (scatter, bar, box, hist) and saves it as png."""

        plt.figure(figsize=(8, 5))
        if type == "scatter" and y_axis:
            sns.scatterplot(data=df, x=x_axis, y=y_axis)

        elif type == "bar" and y_axis:
            sns.barplot(data=df, x=x_axis, y=y_axis)

        elif type == "histogram":
            sns.histplot(data=df, x=x_axis, kde=True)

        elif type == "box" and y_axis:
            sns.boxplot(data=df, x=x_axis, y=y_axis)

        else:
            sns.countplot(data=df, x=x_axis)
        
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        # Save plot to buffer(this is by AI)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        
        return {"chart_type": type, "image_base64": img_base64}

    return [
        statistics_and_correlation_tool,
        anomaly_detection_tool,
        grouping_tool,
        data_visualization_tool
    ]