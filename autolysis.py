import pandas as pd
import json
import requests
import os
import requests    
from dotenv import load_dotenv


def load_dataset(file_path):
    """Loads the dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {e}")

def visualize_data(df, output_dir):
    """Performs basic visualization and saves plots."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    # Generate correlation heatmap
    try:
        correlation = df.corr(numeric_only=True)
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm')
        heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()
        print(f"Correlation heatmap saved at: {heatmap_path}")
    except Exception as e:
        print(f"Error generating correlation heatmap: {e}")

    # Return visualization paths
    return {"correlation_heatmap": heatmap_path}

def analyze_data(df):
    """Performs basic data analysis and returns a dictionary of findings."""
    try:
        analysis = {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "column_types": df.dtypes.astype(str).to_dict(),
            "summary_statistics": df.describe(include='all').to_dict()
        }
        print("Data analysis completed.")
        return analysis
    except Exception as e:
        raise RuntimeError(f"Error during data analysis: {e}")



# Function: Get LLM Summary
# Function: Get LLM Summary
def llm_summary(analysis):
    

    # Load API key from .env file
    load_dotenv()
    api_key = os.getenv("AIPROXY_TOKEN")

    if not api_key:
        raise Exception("API key not found. Please set OPENAI_API_KEY in your .env file.")

    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    prompt = f"""
    Analyze the following dataset metadata and provide insights:
    {json.dumps(analysis, indent=2)}
    """

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a data analyst."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Request failed: {response.status_code}, {response.text}")



# Function: Write Results to README.md
def write_readme(output_dir, summary, images):
    with open(f"{output_dir}/README.md", "w") as f:
        f.write("# Automated Analysis Report\n\n")
        f.write(summary)
        f.write("\n\n## Visualizations\n")
        for img in images:
            f.write(f"![Visualization]({img})\n")


def main():
    file_path = "goodreads.csv"
    output_dir = "output"

    # Step 1: Load the dataset
    df = load_dataset(file_path)

    # Step 2: Visualize the data
    images = visualize_data(df, output_dir)

    # Step 3: Analyze the data
    analysis = analyze_data(df)

    # Step 4: Generate a summary using LLM
    summary = llm_summary(analysis)

    # Step 5: Save the summary
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Summary saved at: {summary_path}")

if __name__ == "__main__":
    main()
