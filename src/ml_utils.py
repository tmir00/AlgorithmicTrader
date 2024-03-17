from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from collections import Counter
import numpy as np


def determine_sentiment(texts):
    # Load the model and tokenizer we will be using.
    finbert = BertForSequenceClassification.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis", num_labels=3)
    tokenizer = BertTokenizer.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis")

    # Use HuggingFace pipeline API to execute the model and tokenizer on our data and obtain results.
    sentiment_pipeline = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)
    results = sentiment_pipeline(texts)

    # Filter out results with the "neutral" label leaving only 'positive' or 'negative'
    filtered_results = [result for result in results if result['label'] != 'neutral']

    if not filtered_results:
        return "", 0

    # Generate list of labels.
    labels = [result['label'] for result in filtered_results]

    # Need to parse counter result which looks like this: [('negative', 1)] to get most common label: 'negative' here.
    most_common_label = Counter(labels).most_common(1)[0][0]

    # Only consider the scores of the most common label (not 'neutral') in mean calculation.
    filtered_results_common = [result['score'] for result in filtered_results if result['label'] == most_common_label]
    average_score = np.mean(filtered_results_common)

    return most_common_label, average_score
