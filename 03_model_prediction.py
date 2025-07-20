import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
# Specify the name of your MLflow experiment
MLFLOW_EXPERIMENT_NAME = "Content Topic Classifier" 
OOT_FILE_PATH = "evaluation_dataset.csv" 
CONFUSION_MATRIX_OUTPUT_PATH = "oot_confusion_matrix.png"

def get_latest_successful_run_id(experiment_name):
    """
    Finds the run ID of the latest successfully completed run in the experiment.
    """
    print(f"Searching for the latest successful run in experiment: '{experiment_name}'...")
    try:
        # Search for runs in the specified experiment, filtering for success and ordering by time
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string="status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if not runs.empty:
            latest_run_id = runs.iloc[0]['run_id']
            print(f"Found latest successful run with ID: {latest_run_id}")
            return latest_run_id
        else:
            print(f"Error: No successful runs found in experiment '{experiment_name}'.")
            return None
    except Exception as e:
        print(f"An error occurred while searching for runs: {e}")
        print("Please ensure the experiment name is correct and you are connected to MLflow.")
        return None


def load_model_and_encoder(run_id):
    """
    Loads the MLflow pyfunc model and extracts the associated label encoder from it.
    """
    if not run_id:
        return None, None
        
    print(f"Loading model and artifacts from MLflow run: {run_id}...")
    try:
        model_uri = f"runs:/{run_id}/classifier_model"
        
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        print("Pyfunc model loaded successfully.")

        # loaded_model -> _model_impl (wrapper) -> python_model (your custom class instance) -> label_encoder
        label_encoder = loaded_model._model_impl.python_model.label_encoder
        print("Label encoder extracted successfully from the loaded model.")
        
        return loaded_model, label_encoder
    except Exception as e:
        print(f"Error loading model or extracting encoder: {e}")
        print(f"Please ensure the run ID '{run_id}' is correct, the model was logged correctly as a pyfunc model, and MLflow is accessible.")
        return None, None

def calculate_mrr(results_df):
    """
    Calculates and prints the Mean Reciprocal Rank (MRR).
    """
    print("\nPerforming MRR analysis...")
    
    def calculate_mrr_score(row):
        """Calculates the reciprocal rank for a single row."""
        true_label = row['topic_id_grouped']
        # Check top 3 predictions for the true label
        if true_label == row['top_1_label']:
            return 1.0
        elif true_label == row['top_2_label']:
            return 1 / 2
        elif true_label == row['top_3_label']:
            return 1 / 3
        else:
            return 0.0

    results_df['MRR'] = results_df.apply(calculate_mrr_score, axis=1)
    mean_mrr = results_df['MRR'].mean()
    
    print(f"Mean Reciprocal Rank (MRR) on OOT data: {mean_mrr:.4f}")
    return mean_mrr

def evaluate_predictions(results_df, label_encoder):
    """
    Prints a classification report and generates a confusion matrix.
    """
    print("\nGenerating evaluation metrics...")
    
    true_labels = results_df['topic_id_grouped']
    predicted_labels = results_df['top_1_label']
    
    # Ensure all labels used in the report are known to the encoder to prevent errors
    all_labels_in_data = np.unique(np.concatenate([true_labels, predicted_labels]))
    known_labels = [label for label in all_labels_in_data if label in label_encoder.classes_]

    # --- Classification Report ---
    print("\nClassification Report:")
    report = classification_report(
        true_labels, 
        predicted_labels, 
        labels=known_labels,
        target_names=known_labels,
        zero_division=0
    )
    print(report)

    # --- Confusion Matrix ---
    print(f"\nGenerating confusion matrix and saving to '{CONFUSION_MATRIX_OUTPUT_PATH}'...")
    cm = confusion_matrix(true_labels, predicted_labels, labels=known_labels)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, 
                xticklabels=known_labels, 
                yticklabels=known_labels,
                annot=True, 
                fmt='d', 
                cmap='Blues')
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix - Out-of-Time Data", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    try:
        plt.savefig(CONFUSION_MATRIX_OUTPUT_PATH)
        print(f"Confusion matrix saved successfully to '{CONFUSION_MATRIX_OUTPUT_PATH}'.")
    except Exception as e:
        print(f"Error saving confusion matrix: {e}")


def main():
    """
    Main function to run the prediction and evaluation pipeline.
    """
    # 1. Get the latest successful run ID from your experiment
    latest_run_id = get_latest_successful_run_id(MLFLOW_EXPERIMENT_NAME)
    if not latest_run_id:
        print("\n--- Script finished: Could not proceed without a valid run ID ---")
        return

    # 2. Load the trained model and label encoder from that run
    model, label_encoder = load_model_and_encoder(latest_run_id)
    if model is None or label_encoder is None:
        print("\n--- Script finished: Failed to load model or artifacts ---")
        return

    # 3. Load the out-of-time (OOT) data
    print(f"\nLoading OOT data from: {OOT_FILE_PATH}")
    try:
        oot_df = pd.read_csv(OOT_FILE_PATH)
        # Ensure the required columns exist
        if 'full_text' not in oot_df.columns or 'topic_id_grouped' not in oot_df.columns:
            print("Error: OOT file must contain 'full_text' and 'topic_id_grouped' columns.")
            return
    except FileNotFoundError:
        print(f"Error: OOT file not found at '{OOT_FILE_PATH}'")
        return
    except Exception as e:
        print(f"Error reading OOT file: {e}")
        return

    # 4. Generate predictions
    print(f"Generating predictions for {len(oot_df)} records...")
    # The model expects a DataFrame with a 'full_text' column
    predictions_df = model.predict(oot_df[['full_text']])
    print("Predictions generated.")

    # 5. Combine predictions with original data
    # Reset index to ensure a clean join
    oot_df.reset_index(drop=True, inplace=True)
    predictions_df.reset_index(drop=True, inplace=True)
    results_df = pd.concat([oot_df, predictions_df], axis=1)

    # 6. Calculate MRR
    calculate_mrr(results_df)

    # 7. Print classification report and confusion matrix
    evaluate_predictions(results_df, label_encoder)

    print("\n--- Script finished ---")

if __name__ == "__main__":
    main()