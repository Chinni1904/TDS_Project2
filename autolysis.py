import pandas as pd
import json
import requests
import os   
from dotenv import load_dotenv
import markdown2
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

#Load the Dataset
def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {e}")


#Data Visualization
def visualize_data(df):
    try:
        correlation = df.corr(numeric_only=True)
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm')
        
        # Save the plot to a buffer
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        base64_image = base64.b64encode(buf.read()).decode('utf-8')
        print("Heatmap generated.")
        return base64_image
    except Exception as e:
        raise RuntimeError(f"Error generating heatmap: {e}")


#Data Analysis
def analyze_data(df):
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



def save_to_readme(heatmap_base64, analysis, summary):
    """Saves all results to a README.md file."""
    readme_content = f"""
# Data Analysis Report

## Correlation Heatmap
![Heatmap](data:image/png;base64,{heatmap_base64})

## Analysis Results
**Number of Rows:** {analysis["num_rows"]}  
**Number of Columns:** {analysis["num_columns"]}  

### Missing Values:
```json
{json.dumps(analysis["missing_values"], indent=2)}
```

### Column Types:
```json
{json.dumps(analysis["column_types"], indent=2)}
```

### Summary Statistics:
```json
{json.dumps(analysis["summary_statistics"], indent=2)}
```

## Summary
{summary}
    """

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("Results saved in README.md")

def main():
    file_path = "goodreads.csv"

    # Step 1: Load the dataset
    df = load_dataset(file_path)

    # Step 2: Visualize the data
    heatmap_base64 = visualize_data(df)

    # Step 3: Analyze the data
    analysis = analyze_data(df)

    # Step 4: Generate a summary using LLM
    summary = llm_summary(analysis)

    # Step 5: Save everything to README.md
    save_to_readme(heatmap_base64, analysis, summary)

if __name__ == "__main__":
    main()

