# content-topic-classifier

This repository contains an initial model that, given a text input, predicts a list of relevant topics the content may belong to. The project is inspired by the [Learning Equality - Curriculum Recommendations](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations) Kaggle competition dataset.

The model predicts **3 topics per content item** using a combination of sentence embeddings and multi-label classification.

The full dataset can be retrieved from the above link.

---

## Repository Structure

```
content-topic-classifier/
├─ README.md        <- Top-level README for developers and users
├─ requirements.txt <- Python dependencies needed to run the notebooks and scripts
├─ eda.ipynb        <- Exploratory analysis to understand the data and define business rules for the POC model
├─ ML_pipeline.py   <- Script to run the full pipeline: ingestion, training, and prediction
├─ common_functions.py <- Utility functions reused across notebooks and scripts
├─ 03_model_prediction.py <- TopicPredictor file for loading the model and generating topic predictions
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/content-topic-classifier.git
cd content-topic-classifier
```

2. Create a virtual environment and install dependencies:

```
python3 -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
```

3. eda.ipynb was created to explore the data and define the business rules used to build the proof-of-concept (POC) classifier model.

4. To run the full pipeline:
- execute ML_pipeline.py: this file will execute the Machine Learning pipeline:
    - 01_data_ingestion_and_preprocessing.py,
    - 02_model_training.py,
    - 03_model_prediction.py

## POC Approach & Results

The classifier was built based on business rules designed to accelerate development. The proof of concept (POC) followed these rules during model creation:

- Only English-language content, which accounts for over 40% of the total, is currently included. As the data product evolves, additional languages may be incorporated. Filtering by language at this stage helps reduce variability in text structure and model complexity, making it easier to prototype and validate assumptions before scaling to a multilingual setting.

- Only content of types ‘document’, ‘html5’, and ‘video’ are included, as they typically contain more text. This provides the classifier with richer information to learn from, which should improve its performance.

- Only topics with more than 100 content items were used to create the POC classifier to speed up model training. There are 19 distinct topics meeting this criterion. Additionally, a new topic named “other” was created using a random sample of contents from the remaining topics.

- Classification metrics were used to evaluate the results:

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|:---|---:|---:|---:|---:|
| other | 0.94 | 0.98 | 0.96 | 287 |
| t_1736b64bf087 | 0.29 | 0.40 | 0.33 | 5 |
| t_17bf2af6d82b | 1.00 | 1.00 | 1.00 | 11 |
| t_1d7ae58a79e6 | 1.00 | 1.00 | 1.00 | 8 |
| t_329497f8cf41 | 1.00 | 0.12 | 0.22 | 8 |
| t_3bea3588613c | 1.00 | 1.00 | 1.00 | 6 |
| t_58ca0afc0bae | 1.00 | 1.00 | 1.00 | 14 |
| t_59bf60f88801 | 1.00 | 1.00 | 1.00 | 7 |
| t_66676e6f7fd5 | 0.83 | 0.83 | 0.83 | 6 |
| t_689e6562417a | 1.00 | 1.00 | 1.00 | 8 |
| t_82dad323a28d | 0.67 | 0.80 | 0.73 | 5 |
| t_92cf4e58f786 | 0.67 | 0.62 | 0.64 | 13 |
| t_a6554500a6df | 0.25 | 0.40 | 0.31 | 5 |
| t_a765d0035891 | 0.00 | 0.00 | 0.00 | 6 |
| t_aab1bc41d83c | 0.00 | 0.00 | 0.00 | 5 |
| t_afb9306e5f18 | 0.60 | 1.00 | 0.75 | 9 |
| t_b3fa23bfbeaa | 1.00 | 1.00 | 1.00 | 13 |
| t_ba5de272b4cb | 1.00 | 1.00 | 1.00 | 10 |
| t_c341f777c725 | 1.00 | 1.00 | 1.00 | 6 |
| t_f3754353b800 | 0.00 | 0.00 | 0.00 | 5 |
| | | | | |
| **Accuracy** | | | **0.90** | **437** |
| **Macro Avg** | **0.71** | **0.71** | **0.69** | **437** |
| **Weighted Avg** | **0.88** | **0.90** | **0.88** | **437** |

- The Mean Reciprocal Rank (MRR) metric was used to evaluate the results when predicting up to 3 topics per content. On the out-of-time (OOT) data, the MRR was 0.9443, indicating that most of the time, one of the top 3 predicted topics was correct.

## Next steps: 

- Try out other models and and fine-tuning to improve performance.
  
- Add more training examples to help the model fine-tune and generalize better.

- Build a feature engineering process to improve classification and recommendations, focusing on things like:

    - Text level
    - School or institution information
    - Reading time estimates
    - Content type — checking if some students prefer certain kinds of content, which could help personalize recommendations

- Use prompt engineering and a retrieval-augmented generation (RAG) system with topics.csv as a knowledge base to improve predictions in cases where the classifier struggles.

