import pandas as pd
import numpy as numpy
import mlflow
from commom_functions import *
from datetime import datetime

CONTENT_FILE_PATH = 'data/content.csv'
CORRELATIONS_FILE_PATH = 'data/correlations.csv'
OUTPUT_FILE_PATH = 'preprocessed_data_classifier.csv'
MLFLOW_EXPERIMENT_NAME = "Content Topic Classifier"

def load_and_prepare_data(content_path, correlations_path):
    """
    Loads content and topic correlation data, and prepares the topic data by exploding content IDs.
    """
    print(f"Loading content data from: {content_path}")
    df_content = pd.read_csv(content_path)
    
    print(f"Loading and preparing topic data from: {correlations_path}")
    df_topic_content = pd.read_csv(correlations_path)
    
    # Prepare the topic content dataframe by splitting and exploding content IDs
    df_topic_content = (df_topic_content[['topic_id', 'content_ids']]
                        .assign(content_id=lambda df: df['content_ids'].str.split())
                        .explode('content_id'))
    
    print("Data loading and initial preparation complete.")
    return df_content, df_topic_content

def filter_and_clean_content(df_content):
    """
    Filters content for specific kinds and languages, then cleans and combines text fields.
    """
    print("Filtering content for kind ('document', 'html5', 'video') and language ('en')...")
    filtered_df = df_content[(df_content['language'] == 'en') & 
                             (df_content['kind'].isin(['document', 'html5', 'video']))].copy()
    
    print("Cleaning text fields ('description', 'text')...")
    filtered_df = clean_textlines(filtered_df, ['description', 'text'])
    filtered_df['full_text'] = (filtered_df['description'].fillna('') + ' ' + filtered_df['text'].fillna('')).str.strip()
    cleaned_df = filtered_df[filtered_df['full_text'].astype(bool)].copy()
    cleaned_df['full_text'] = cleaned_df['full_text'].str.lower()
    
    print("Content filtering and cleaning complete.")
    return cleaned_df

def merge_and_filter_by_topic(df_content, df_topics):
    """
    Merges content with topics, performs Pareto analysis to find top topics,
    and filters the dataset to include only content from those topics.
    """
    print("Merging content and topic data...")
    df_merged = pd.merge(df_content[['id', 'full_text']], 
                         df_topics[['topic_id', 'content_id']], 
                         left_on='id', 
                         right_on='content_id', 
                         how='left')
    
    print("Performing Pareto analysis to identify top topics...")
    pareto_topics_df = pareto_analysis(df_merged.dropna(subset=['topic_id']), 'topic_id')
    top_topics = set(pareto_topics_df[pareto_topics_df['cumulative_percentage'] <= 80].index)
    
    print(f"Found {len(top_topics)} topics covering 80% of content.")
    mlflow.log_param("num_top_topics", len(top_topics))
    
    print("Filtering merged data to keep only top topics...")
    df_classifier = df_merged[df_merged['topic_id'].isin(top_topics)].copy()
    
    return df_classifier[['id', 'full_text', 'topic_id']]

def main():
    """
    Main function to run the data ingestion and processing pipeline.
    """
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"Data_Processing_{current_time}"):
        
        print("--- Starting Data Ingestion and Processing Pipeline ---")
        
        # 1. Load and Prepare Data
        df_content, df_topic = load_and_prepare_data(CONTENT_FILE_PATH, CORRELATIONS_FILE_PATH)
        mlflow.log_param("raw_content_shape", df_content.shape)
        mlflow.log_param("prepared_topics_shape", df_topic.shape)

        # 2. Filter and Clean Content
        df_content_clean = filter_and_clean_content(df_content)
        mlflow.log_param("cleaned_content_shape", df_content_clean.shape)
        print(f"Shape after cleaning content: {df_content_clean.shape}")

        # 3. Merge data and filter by top topics
        df_classifier = merge_and_filter_by_topic(df_content_clean, df_topic)
        mlflow.log_param("final_classifier_data_shape", df_classifier.shape)
        print(f"Shape of final data for classifier: {df_classifier.shape}")
     
        # 4. Save preprocessed data
        print(f"Saving preprocessed data to: {OUTPUT_FILE_PATH}")
        df_classifier.to_csv(OUTPUT_FILE_PATH, index=False)
        mlflow.log_artifact(OUTPUT_FILE_PATH)
        print("Preprocessed data saved successfully.")
        
        print("\n--- Pipeline Finished ---")

if __name__ == "__main__":
    main()