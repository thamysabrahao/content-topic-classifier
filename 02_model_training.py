import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import torch
import mlflow
import mlflow.pyfunc
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding


class ClassifierModel(mlflow.pyfunc.PythonModel):
    '''
    MLflow Wrapper for a text classification model built with DistilBERT.
    '''
    def load_context(self, context):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(context.artifacts["tokenizer_path"])
        self.model = DistilBertForSequenceClassification.from_pretrained(context.artifacts["model_path"])
        with open(context.artifacts["label_encoder_path"], "rb") as f:
            self.label_encoder = pickle.load(f)

    def predict(self, context, model_input):
        # Ensure model_input is a DataFrame with a 'full_text' column
        if not isinstance(model_input, pd.DataFrame) or 'full_text' not in model_input.columns:
            raise ValueError("model_input must be a DataFrame with a 'full_text' column.")

        texts = model_input['full_text'].tolist()
        
        # Tokenize inputs
        inputs = self.tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).numpy()

        # Get top 3 predictions
        top3_indices = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
        top3_labels = np.vectorize(lambda idx: self.label_encoder.classes_[idx])(top3_indices)
        top3_probs = np.take_along_axis(probs, top3_indices, axis=1)

        # Prepare output DataFrame
        predictions_df = pd.DataFrame({
            'top_1_label': top3_labels[:, 0],
            'top_1_prob': top3_probs[:, 0],
            'top_2_label': top3_labels[:, 1],
            'top_2_prob': top3_probs[:, 1],
            'top_3_label': top3_labels[:, 2],
            'top_3_prob': top3_probs[:, 2],
        })
        return predictions_df


def load_data(file_path):
    '''
    Loads data from a CSV file
    '''
    df = pd.read_csv(file_path)
    return df


def filter_topics(df, min_docs=100):
    '''
    Filters topics to include only those with at least `min_docs` contents
    '''
    print(f"Filtering topics (min_docs={min_docs})...")
    topic_counts = df['topic_id'].value_counts()
    frequent_topics = topic_counts[topic_counts >= min_docs].index
    df_filtered = df.copy()
    df_filtered['topic_id_grouped'] = df_filtered['topic_id'].where(df_filtered['topic_id'].isin(frequent_topics), other='other')

    freq_df = df_filtered[df_filtered['topic_id_grouped'] != 'other']
    other_df = df_filtered[df_filtered['topic_id_grouped'] == 'other']
    
    # Sample 10% from "other" category
    other_sampled = other_df.sample(frac=0.1, random_state=42)
    
    result_df = pd.concat([freq_df, other_sampled])
    df_filtered = result_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Data after topic filtering and sampling: {df_filtered.shape}")
    return df_filtered


def create_eval_dataset(df_filtered, frac=0.05):
    '''
    Creates a separate evaluation dataset to prevent data leakage.
    '''
    print("Creating evaluation dataset...")
    eval_df = df_filtered.groupby('topic_id_grouped', group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=42)).copy()
    df_train_test = df_filtered[~df_filtered['id'].isin(eval_df['id'])].copy()
    print(f"Train/Test dataset shape: {df_train_test.shape}")
    print(f"Evaluation dataset shape: {eval_df.shape}")
    return df_train_test, eval_df


def encode_labels(df_train_test, eval_df):
    '''
    Encodes topic_id to numerical labels.
    '''
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    df_train_test['label'] = label_encoder.fit_transform(df_train_test['topic_id_grouped'])
    eval_df['label'] = label_encoder.transform(eval_df['topic_id_grouped'])
    
    num_labels = len(label_encoder.classes_)
    id_to_label = {i: label for i, label in enumerate(label_encoder.classes_)}
    label_to_id = {label: i for i, label in enumerate(label_encoder.classes_)}

    print(f"Number of unique topics: {num_labels}")
    print(f"Label mapping: {id_to_label}")
    return df_train_test, eval_df, label_encoder, num_labels, id_to_label, label_to_id


def train_model(train_df, test_df, num_labels, id_to_label, label_to_id):
    '''
    Trains the classifier model using DistilBERT.
    '''
    print("Initializing tokenizer and model...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=num_labels,
        id2label=id_to_label,
        label2id=label_to_id
    )

    def tokenize_function(batch):
        return tokenizer(batch['full_text'], truncation=True)

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

    print("Tokenizing datasets...")
    train_ds = train_ds.map(tokenize_function, batched=True)
    test_ds = test_ds.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir='./results',
        do_eval=True,
        per_device_train_batch_size=20,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        learning_rate=5e-5,
        save_steps=500,
        logging_dir='./logs',
        logging_steps=10,
        seed=42,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting model training...")
    trainer.train()
    print("Model training completed.")
    return model, tokenizer, trainer


def evaluate_model(trainer, eval_ds, label_encoder):
    '''
    Evaluates the trained model and logs metrics and confusion matrix.
    '''
    print("Starting model evaluation...")
    # Tokenize eval_ds for prediction
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    def tokenize_function(batch):
        return tokenizer(batch['full_text'], truncation=True)
    
    eval_dataset = Dataset.from_pandas(eval_ds.reset_index(drop=True))
    eval_ds_tokenized = eval_dataset.map(tokenize_function, batched=True)

    predictions = trainer.predict(eval_ds_tokenized)
    preds = np.argmax(predictions.predictions, axis=1)
    true = predictions.label_ids

    # Classification Report
    report = classification_report(true, preds, target_names=label_encoder.classes_, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(true, preds, target_names=label_encoder.classes_))

    # Log metrics to MLflow
    mlflow.log_metrics({
        "accuracy": report["accuracy"],
        "f1_macro": report["macro avg"]["f1-score"],
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
    })

    # Confusion Matrix
    cm = confusion_matrix(true, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,
                annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    
    # Log confusion matrix as an artifact
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close() # Close plot to free memory
    print("Model evaluation completed and metrics logged.")
    return predictions, eval_ds_tokenized

def analyze_mrr(predictions, eval_df, label_encoder):
    '''
    Performs MRR (Mean Reciprocal Rank) analysis.
    '''
    print("Performing MRR analysis...")
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
    top3_indices = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
    top3_labels = np.vectorize(lambda idx: label_encoder.classes_[idx])(top3_indices)
    top3_probs = np.take_along_axis(probs, top3_indices, axis=1)

    results_df = pd.DataFrame({
        'text': eval_df['full_text'].values,
        'true_label': label_encoder.inverse_transform(predictions.label_ids),
        'top_1_label': top3_labels[:, 0],
        'top_1_prob': top3_probs[:, 0],
        'top_2_label': top3_labels[:, 1],
        'top_2_prob': top3_probs[:, 1],
        'top_3_label': top3_labels[:, 2],
        'top_3_prob': top3_probs[:, 2],
    })

    def calculate_mrr_score(row):
        if row['true_label'] == row['top_1_label']:
            return 1.0
        elif row['true_label'] == row['top_2_label']:
            return 1 / 2
        elif row['true_label'] == row['top_3_label']:
            return 1 / 3
        else:
            return 0.0

    results_df['MRR'] = results_df.apply(calculate_mrr_score, axis=1)
    mean_mrr = results_df['MRR'].mean()
    print(f"\nMean Reciprocal Rank (MRR): {mean_mrr}")
    mlflow.log_metric("mean_reciprocal_rank", mean_mrr)

    plt.figure(figsize=(8, 5))
    sns.histplot(results_df['MRR'], bins=[0, 0.33, 0.5, 1.0, 1.01], kde=False)
    plt.title("MRR Distribution")
    plt.xlabel("MRR Value")
    plt.ylabel("Number of Documents")
    plt.xticks([0, 1/3, 0.5, 1.0], ['0.0', '1/3', '0.5', '1.0'])
    plt.grid(True)
    
    # Log MRR distribution plot
    plt.savefig("mrr_distribution.png")
    mlflow.log_artifact("mrr_distribution.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=results_df, x='true_label', y='MRR')
    plt.xticks(rotation=90)
    plt.title("MRR Distribution by True Label")
    plt.tight_layout()
    
    # Log MRR by true label plot
    plt.savefig("mrr_by_true_label.png")
    mlflow.log_artifact("mrr_by_true_label.png")
    plt.close()
    print("MRR analysis completed and results logged.")

def main():
    
    mlflow.set_experiment("Content Topic Classifier")
    with mlflow.start_run():
        mlflow.log_param("min_docs_for_topic_filtering", 100)
        mlflow.log_param("eval_dataset_fraction", 0.05)
        mlflow.log_param("num_train_epochs", 1)
        mlflow.log_param("learning_rate", 5e-5)
        mlflow.log_param("per_device_train_batch_size", 20)
        mlflow.log_param("per_device_eval_batch_size", 8)

        # 1. Data Load
        df_model = load_data('preprocessed_data_classifier.csv')

        # 2. Topic Filtering
        df_filtered = filter_topics(df_model, min_docs=100)

        # 3. Creating Evaluation Dataset
        df_train_test, eval_df = create_eval_dataset(df_filtered, frac=0.05)
        
        # Split df_train_test into training and testing sets
        train_df, test_df = train_test_split(
            df_train_test[['full_text', 'topic_id_grouped']],
            test_size=0.2,
            random_state=42,
            stratify=df_train_test['topic_id_grouped']
        )
        print(f"Training dataset shape: {train_df.shape}")
        print(f"Test dataset shape: {test_df.shape}")

        # 4. Label Encoding
        train_df, test_df, label_encoder, num_labels, id_to_label, label_to_id = encode_labels(train_df, test_df)
        # Ensure eval_df also has the 'label' column
        eval_df['label'] = label_encoder.transform(eval_df['topic_id_grouped'])

        # 5. Model Training
        model, tokenizer, trainer = train_model(train_df, test_df, num_labels, id_to_label, label_to_id)
        
        # Save model and tokenizer as MLflow artifacts
        # Create temporary directories for saving
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "distilbert_model")
            tokenizer_path = os.path.join(tmp_dir, "distilbert_tokenizer")
            label_encoder_path = os.path.join(tmp_dir, "label_encoder.pkl")

            model.save_pretrained(model_path)
            tokenizer.save_pretrained(tokenizer_path)
            with open(label_encoder_path, "wb") as f:
                pickle.dump(label_encoder, f)

            # Log the model with MLflow
            mlflow.pyfunc.log_model(
                "classifier_model",
                python_model=ClassifierModel(),
                artifacts={
                    "model_path": model_path,
                    "tokenizer_path": tokenizer_path,
                    "label_encoder_path": label_encoder_path,
                },
                input_example=pd.DataFrame({"full_text": ["example text for prediction"]}),
                signature=mlflow.models.signature.infer_signature(
                    pd.DataFrame({"full_text": ["example text"]}),
                    pd.DataFrame({"top_1_label": ["label"], "top_1_prob": [0.0],
                                  "top_2_label": ["label"], "top_2_prob": [0.0],
                                  "top_3_label": ["label"], "top_3_prob": [0.0]})
                )
            )
        print("Model, tokenizer, and label encoder saved to MLflow.")

        # 6. Model Evaluation
        predictions, eval_ds_tokenized = evaluate_model(trainer, eval_df, label_encoder)

        # 7. MRR Analysis
        analyze_mrr(predictions, eval_df, label_encoder)
        
        print("MLflow run completed.")

if __name__ == "__main__":
    main()