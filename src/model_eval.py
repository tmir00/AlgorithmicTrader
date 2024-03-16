import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# Load models and tokenizers
finbert_TEST = BertForSequenceClassification.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis",num_labels=3)
tokenizer_TEST = BertTokenizer.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis")

finbert_prosus = BertForSequenceClassification.from_pretrained('ProsusAI/finbert', num_labels=3)
tokenizer_prosus = BertTokenizer.from_pretrained('ProsusAI/finbert')

# Setup pipelines
pipeline_TEST = pipeline("sentiment-analysis", model=finbert_TEST, tokenizer=tokenizer_TEST)
pipeline_prosus = pipeline("sentiment-analysis", model=finbert_prosus, tokenizer=tokenizer_prosus)

def evaluate_model(pipeline, df):
    predictions = []
    for text in df['text']:
        pipeline_result = pipeline(text)
        # Need to parse result that looks like this: [{'label': 'neutral', 'score': 0.9888190627098083}]
        predictions.append(pipeline_result[0]['label'])
    mapped_predictions = [label.lower() for label in predictions]
    return accuracy_score(df['label'], mapped_predictions)

# data_size = 300
df = pd.read_csv('../data/all-data.csv', encoding='ISO-8859-1')
# df = df[:data_size + 1]

accuracy_tone = evaluate_model(pipeline_TEST, df)
accuracy_prosus = evaluate_model(pipeline_prosus, df)

print(f"Accuracy FinBERT Test: {accuracy_tone}\nAccuracy ProsusAI FinBERT: {accuracy_prosus}")
