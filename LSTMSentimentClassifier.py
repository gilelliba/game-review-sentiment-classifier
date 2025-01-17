import os
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, Dataset
import joblib
import json

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=256):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)

class LSTMDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

class LSTMSentimentClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
        self.model = None
        self.device = torch.device('cpu')
        self.model_path = 'lstm_model.pth'  # Path to save the model
        self.vocab_size = None  # Initialize vocab_size
        self.model = None  # Ensure model is initialized to None

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
        
        X = self.vectorizer.fit_transform(df['review_clean']).toarray()
        y = df['stars'].values
        
        self.vocab_size = len(self.vectorizer.vocabulary_) + 1  # Store vocab_size
        with open('vocab_size.json', 'w') as f:  # Save vocab_size to a file
            json.dump({'vocab_size': self.vocab_size}, f)
        
        self.model = LSTMModel(self.vocab_size).to(self.device)  # Initialize model during training
        
        # Split the data into training and validation sets (80-20 split)
        split_index = int(0.8 * len(X))
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]

        dataset = LSTMDataset(torch.LongTensor(X_train).to(self.device), torch.FloatTensor(y_train).to(self.device))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(100):  # Adjust epochs as needed
            self.model.train()
            for texts, labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model(texts.to(self.device))
                loss = criterion(outputs.squeeze(), labels.to(self.device))
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')
        
        print("Training complete!")

        # Save the model after training
        torch.save(self.model.state_dict(), self.model_path)
        print("Model saved!")

        # Save the fitted vectorizer
        joblib.dump(self.vectorizer, 'vectorizer.pkl')  # Save the vectorizer
        print("Vectorizer saved!")

    def load_model(self):
        if os.path.exists(self.model_path):
            # Load vocab_size from the file
            if os.path.exists('vocab_size.json'):
                with open('vocab_size.json', 'r') as f:
                    self.vocab_size = json.load(f)['vocab_size']
            else:
                raise ValueError("vocab_size file not found. Please train the model first.")
                
            self.model = LSTMModel(self.vocab_size).to(self.device)  # Use stored vocab_size
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()
            print("Model loaded!")

            # Load the fitted vectorizer
            if os.path.exists('vectorizer.pkl'):
                self.vectorizer = joblib.load('vectorizer.pkl')  # Load the vectorizer
            else:
                raise ValueError("Vectorizer file not found. Please train the model first.")
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
            X = self.vectorizer.transform([review_text.lower()]).toarray()
            X_tensor = torch.LongTensor(X).to(self.device)
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
    classifier = LSTMSentimentClassifier()
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
