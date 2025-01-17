import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

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
        self.device = torch.device('cpu')

        # Initialize gaming-specific negative and positive words
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

    def evaluate(self, X_test_vectorized, y_test):
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_vectorized.toarray()).to(self.device)
            y_test_tensor = torch.FloatTensor(y_test.values).to(self.device)
            outputs = self.model(X_test_tensor)
            loss = nn.MSELoss()(outputs.squeeze(), y_test_tensor)
            print(f'Test Loss: {loss.item():.4f}')

    def train(self, csv_path):
        print("Loading and training on review data...")
        df = pd.read_csv(csv_path)
        
        # Split the dataset into training and testing sets (80-20)
        X_train, X_test, y_train, y_test = train_test_split(df['review_clean'], df['stars'], test_size=0.2, random_state=42)
        
        # Fit the vectorizer on the training data
        self.vectorizer.fit(X_train)
        
        # Transform both training and testing data
        X_train_vectorized = self.vectorizer.transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train_vectorized.toarray()).to(self.device)
        y_tensor = torch.FloatTensor(y_train.values).to(self.device)

        # Initialize the model
        input_size = X_tensor.shape[1]
        self.model = FFNModel(input_size).to(self.device)

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

        # After training, evaluate the model
        self.evaluate(X_test_vectorized, y_test)

    def load_model(self):
        print("Model loading is disabled.")

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
            rating = max(rating, 2.5)  # Ensure a minimum rating of 2.5 for neutral reviews
        
        rating = round(rating * 2) / 2
        sentiment = self.get_sentiment_label(rating)
        
        return rating, sentiment, neg_count, pos_count

def main():
    classifier = FFNSentimentClassifier()
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