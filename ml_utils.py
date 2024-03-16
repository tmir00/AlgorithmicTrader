from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from collections import Counter
import numpy as np

def determine_sentiment(texts):
    finbert = BertForSequenceClassification.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis", num_labels=3)
    tokenizer = BertTokenizer.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis")

    sentiment_pipeline = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)
    results = sentiment_pipeline(texts)

    # Filter out results with the "neutral" label
    filtered_results = [result for result in results if result['label'] != 'neutral']  # Assuming 'LABEL_1' corresponds to neutral

    if not filtered_results:
        return "", 0

    labels = [result['label'] for result in filtered_results]
    scores = [result['score'] for result in filtered_results]

    most_common_label = Counter(labels).most_common(1)[0][0]
    average_score = np.mean(scores)

    return most_common_label, average_score

print(determine_sentiment(['markets responded negatively to the news!','traders were displeased!']))