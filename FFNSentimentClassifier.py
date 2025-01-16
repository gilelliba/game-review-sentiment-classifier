import os
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer

class FFNModel(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super(FFNModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

class FFNSentimentClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 3),
            min_df=2,
            stop_words='english'
        )
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = 'ffn_model.pth'  # Path to save the model

        # Gaming-specific words
        self.negative_words = {
            'terrible', 'worst', 'awful', 'horrible', 'bad', 'poor', 'garbage',
            'waste', 'trash', 'boring', 'hate', 'disappointing', 'disappointed',
            'useless', 'stupid', 'worse', 'mess', 'mediocre', 'avoid',
            'broken', 'buggy', 'lag', 'crash', 'unplayable', 'poorly optimized',
            'unbalanced', 'pay to win', 'p2w', 'dead game', 'toxic', 'grindy',
            'repetitive', 'clunky', 'unresponsive', 'spyware', 'clone', 'ripoff',
            'cash grab', 'microtransactions', 'predatory', 'unfinished'
        }
        
        self.positive_words = {
            'good', 'great', 'awesome', 'excellent', 'amazing', 'love',
            'fantastic', 'perfect', 'fun', 'enjoyable', 'best', 'wonderful',
            'brilliant', 'outstanding', 'solid', 'nice', 'incredible',
            'addictive', 'polished', 'smooth', 'balanced', 'engaging',
            'immersive', 'innovative', 'masterpiece', 'recommended', 'fresh',
            'competitive', 'rewarding', 'satisfying'
        }

    def train(self, csv_path):
        print("Loading and training on review data...")
        df = pd.read_csv(csv_path)
        
        X = self.vectorizer.fit_transform(df['review_clean'])
        y = df['stars'].values
        
        input_size = X.shape[1]
        self.model = FFNModel(input_size).to(self.device)
        
        X_tensor = torch.FloatTensor(X.toarray()).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(100):  # Adjust epochs as needed
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs.squeeze(), y_tensor)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')
        
        print("Training complete!")

        # Save the model after training
        self.save_model()
        
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print("Model saved!")

    def load_model(self):
        if os.path.exists(self.model_path):
            input_size = self.vectorizer.max_features  # Ensure this matches the fitted vectorizer
            self.model = FFNModel(input_size).to(self.device)  # Initialize the model
            self.model.load_state_dict(torch.load(self.model_path))  # Load the model state
            self.model.eval()
            print("Model loaded!")
        else:
            print("No saved model found. Please train the model first.")

    def count_sentiment_words(self, text):
        text = text.lower()
        words = set(text.split())
        
        neg_count = sum(1 for word in words if word in self.negative_words)
        pos_count = sum(1 for word in words if word in self.positive_words)
        
        return neg_count, pos_count

    def get_sentiment_label(self, rating):
        if rating >= 3.5:
            return "Positive"
        elif rating < 2.5:
            return "Negative"
        else:
            return "Neutral"

    def predict(self, review_text):
        self.model.eval()
        with torch.no_grad():
            X = self.vectorizer.transform([review_text.lower()])
            X_tensor = torch.FloatTensor(X.toarray()).to(self.device)
            rating = self.model(X_tensor).item()
        
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
    classifier = FFNSentimentClassifier()
    if os.path.exists(classifier.model_path):
        classifier.load_model()
    else:
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