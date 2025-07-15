# content-topic-classifier

This repository contains an initial model that, given a text input, predicts a list of relevant topics the content may belong to. The project is inspired by the [Learning Equality - Curriculum Recommendations](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations) Kaggle competition dataset.

The model predicts **3 topics per content item** using a combination of sentence embeddings and multi-label classification.

---

## Repository Structure

```
content-topic-classifier/
├─ README.md <- Top-level README for developers and users
├─ requirements.txt <- Python dependencies needed to run the notebooks and scripts
├─ classifier_model.ipynb <- Proof of concept (POC) notebook for training a topic classifier
├─ prediction_example.ipynb <- Example notebook showing how to load the model and make predictions on new data
├─ common_functions.py <- Utility functions reused across notebooks and scripts
├─ model_api.py <- TopicPredictor class for loading the model and generating topic predictions
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

3. To run the full pipeline:
- Open and run the classifier_model.ipynb to train the model;
- Use prediction_example.ipynb to see how predictions are made on new content.

