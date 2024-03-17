import pandas as pd
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# Load models and tokenizers for analysis
finbert_TEST = BertForSequenceClassification.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis",num_labels=3)
tokenizer_TEST = BertTokenizer.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis")
finbert_prosus = BertForSequenceClassification.from_pretrained('ProsusAI/finbert', num_labels=3)
tokenizer_prosus = BertTokenizer.from_pretrained('ProsusAI/finbert')

# Setup HuggingFace pipelines
pipeline_TEST = pipeline("sentiment-analysis", model=finbert_TEST, tokenizer=tokenizer_TEST)
pipeline_prosus = pipeline("sentiment-analysis", model=finbert_prosus, tokenizer=tokenizer_prosus)


def evaluate_model(pipeline, df):
    predictions = []

    for text in df['text']:
        # Get ML model result.
        pipeline_result = pipeline(text)
        # Need to parse result that looks like this: [{'label': 'neutral', 'score': 0.9888190627098083}]
        predictions.append(pipeline_result[0]['label'].lower())

    # Return the accuracy score with the test dataset compared to what our model predicted
    return accuracy_score(df['label'], predictions)


data_size = 300

# Load Test Data
df = pd.read_csv('./data/all-data.csv', encoding='ISO-8859-1')

df = df[:data_size]

# Get the accuracy scores of the benchmark PROSUS financial BERT model vs test model.
accuracy_test = evaluate_model(pipeline_TEST, df)
accuracy_prosus = evaluate_model(pipeline_prosus, df)

print(f"Accuracy FinBERT Test: {accuracy_test}\n Accuracy ProsusAI FinBERT: {accuracy_prosus}")
