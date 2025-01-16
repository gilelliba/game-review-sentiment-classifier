import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingRegressor

class SentimentClassifier:
    def __init__(self):
        # Initialize TF-IDF vectorizer for text processing
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 3),
            min_df=2,
            stop_words='english'
        )
        
        # Gaming-specific negative words
        self.negative_words = {
            'terrible', 'worst', 'awful', 'horrible', 'bad', 'poor', 'garbage',
            'waste', 'trash', 'boring', 'hate', 'disappointing', 'disappointed',
            'useless', 'stupid', 'worse', 'mess', 'mediocre', 'avoid',
            'broken', 'buggy', 'lag', 'crash', 'unplayable', 'poorly optimized',
            'unbalanced', 'pay to win', 'p2w', 'dead game', 'toxic', 'grindy',
            'repetitive', 'clunky', 'unresponsive', 'spyware', 'clone', 'ripoff',
            'cash grab', 'microtransactions', 'predatory', 'unfinished'
        }
        
        # Gaming-specific positive words
        self.positive_words = {
            'good', 'great', 'awesome', 'excellent', 'amazing', 'love',
            'fantastic', 'perfect', 'fun', 'enjoyable', 'best', 'wonderful',
            'brilliant', 'outstanding', 'solid', 'nice', 'incredible',
            'addictive', 'polished', 'smooth', 'balanced', 'engaging',
            'immersive', 'innovative', 'masterpiece', 'recommended', 'fresh',
            'competitive', 'rewarding', 'satisfying'
        }
        
        # Initialize ML model for sentiment prediction
        self.model = GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=7,
            random_state=42
        )
        
    def count_sentiment_words(self, text):
        # Count positive and negative words in review
        text = text.lower()
        words = set(text.split())
        
        neg_count = sum(1 for word in words if word in self.negative_words)
        pos_count = sum(1 for word in words if word in self.positive_words)
        
        return neg_count, pos_count
    
    def get_sentiment_label(self, rating):
        # Convert rating to sentiment category
        if rating >= 3.5:
            return "Positive"
        elif rating < 2.5:
            return "Negative"
        else:
            return "Neutral"
    
    def train(self, csv_path):
        # Train model on processed review data
        print("Loading and training on review data...")
        df = pd.read_csv(csv_path)
        
        X = self.vectorizer.fit_transform(df['review_clean'])
        y = df['stars']
        
        self.model.fit(X, y)
        print("Training complete!")
        
    def predict(self, review_text):
        # Predict sentiment using hybrid approach (ML + rule-based)
        X = self.vectorizer.transform([review_text.lower()])
        rating = self.model.predict(X)[0]
        
        neg_count, pos_count = self.count_sentiment_words(review_text)
        
        # Adjust rating based on sentiment words
        if neg_count > pos_count:
            adjustment = -1.0 * neg_count
            rating = max(rating + adjustment, 0)
        elif pos_count > neg_count:
            adjustment = 0.5 * pos_count
            rating = min(rating + adjustment, 5)
        
        # Force lower rating for negative-only reviews
        if neg_count > 0 and pos_count == 0:
            rating = min(rating, 2.0)
        
        # Keep neutral reviews in middle range
        if neg_count == 0 and pos_count == 0:
            rating = min(max(rating, 2.5), 3.5)
        
        rating = round(rating * 2) / 2
        sentiment = self.get_sentiment_label(rating)
        
        return rating, sentiment, neg_count, pos_count

def main():
    # Initialize and run sentiment analyzer
    classifier = SentimentClassifier()
    classifier.train('processed_game_reviews.csv')
    
    while True:
        print("\n" + "="*50)
        review = input("Enter a game review (or 'quit' to exit): ")
        
        if review.lower() == 'quit':
            break
        
        rating, sentiment, neg_count, pos_count = classifier.predict(review)
        
        print("\nAnalysis Results:")
        print(f"Rating: {rating:.1f} stars")
        print(f"Sentiment: {sentiment}")
        print(f"Positive words found: {pos_count}")
        print(f"Negative words found: {neg_count}")

if __name__ == "__main__":
    main()
