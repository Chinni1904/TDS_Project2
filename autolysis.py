

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import ChatCompletion
from dotenv import load_dotenv
import markdown2

# Load environment variables
load_dotenv()
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise EnvironmentError("AIPROXY_TOKEN not set. Please set it in your environment.")

# Constants
LLM_MODEL = "gpt-4o-mini"
OUTPUT_README = "README.md"

# LLM API setup
class LLMClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = ChatCompletion(api_key=api_key)

    def chat(self, messages, functions=None):
        return self.client.create(messages=messages, functions=functions, model=LLM_MODEL)

# Helper functions
def analyze_dataset(file_path):
    """Analyze the dataset and generate summary statistics."""
    data = pd.read_csv(file_path)
    summary = {
        "shape": data.shape,
        "columns": [{"name": col, "type": data[col].dtype.name, "sample": data[col].dropna().unique()[:5].tolist()} for col in data.columns],
        "missing_values": data.isnull().sum().to_dict(),
        "description": data.describe(include="all").to_dict(),
    }
    return data, summary

def visualize_data(data, output_prefix):
    """Generate visualizations for the dataset."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    heatmap_path = f"{output_prefix}_heatmap.png"
    plt.savefig(heatmap_path, dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    data.isnull().sum().plot(kind='bar', color='skyblue')
    plt.title("Missing Values Count")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    missing_path = f"{output_prefix}_missing.png"
    plt.savefig(missing_path, dpi=300)
    plt.close()

    return [heatmap_path, missing_path]

def generate_narrative(llm_client, filename, summary, chart_paths):
    """Generate a narrative about the dataset and analysis."""
    prompt = f"""
    You are an AI analyst. I analyzed a dataset called {filename}. Here are some details:
    - Dataset shape: {summary['shape']}
    - Columns: {[f"{col['name']} ({col['type']})" for col in summary['columns']]}
    - Missing values: {summary['missing_values']}
    - Summary statistics: {summary['description']}

    Additionally, I generated these visualizations:
    - Correlation heatmap
    - Missing values count

    Write a Markdown story summarizing the dataset, analysis, and insights, referencing these visualizations where appropriate.
    """
    response = llm_client.chat([{"role": "system", "content": "You are a data analyst."}, {"role": "user", "content": prompt}])
    return response.choices[0].message.content

# Main script
def main():
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    if not os.path.exists(dataset_path):
        print(f"Error: File '{dataset_path}' not found.")
        sys.exit(1)

    try:
        # Analyze the dataset
        data, summary = analyze_dataset(dataset_path)

        # Visualize the data
        output_prefix = os.path.splitext(os.path.basename(dataset_path))[0]
        chart_paths = visualize_data(data, output_prefix)

        # Initialize LLM client
        llm_client = LLMClient(api_key=AIPROXY_TOKEN)

        # Generate the narrative
        narrative = generate_narrative(llm_client, dataset_path, summary, chart_paths)

        # Write results to README.md
        with open(OUTPUT_README, "w") as readme:
            readme.write(markdown2.markdown(narrative))
            for chart in chart_paths:
                readme.write(f"\n![{os.path.basename(chart)}]({chart})\n")

        print(f"Analysis completed. Results saved in {OUTPUT_README}. Charts: {', '.join(chart_paths)}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()






