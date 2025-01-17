import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

class BERTSentimentClassifier:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
        self.device = torch.device('cpu')
        self.model.to(self.device)
        self.model_path = 'bert_model.pth'
        
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

    def evaluate(self, test_encodings, test_labels):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            data_loader = DataLoader(list(zip(test_encodings['input_ids'], test_labels)), batch_size=4)
            for batch in data_loader:
                input_ids, labels = batch
                outputs = self.model(input_ids=input_ids.to(self.device), labels=labels.to(self.device))
                loss = outputs.loss
                total_loss += loss.item()
        
        average_loss = total_loss / len(data_loader)
        print(f'Test Loss: {average_loss:.4f}')

    def train(self, csv_path):
        print("Loading and training on review data...")
        df = pd.read_csv(csv_path)
        
        # Split the dataset into training and testing sets (80-20)
        X_train, X_test, y_train, y_test = train_test_split(df['review_clean'], df['stars'], test_size=0.2, random_state=42)
        
        # Tokenization with reduced max_length
        train_encodings = self.tokenizer(list(X_train), truncation=True, padding=True, return_tensors='pt', max_length=64)
        test_encodings = self.tokenizer(list(X_test), truncation=True, padding=True, return_tensors='pt', max_length=64)

        train_labels = torch.FloatTensor(y_train.values).unsqueeze(1)
        test_labels = torch.FloatTensor(y_test.values).unsqueeze(1)

        # Training loop
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-5)
        self.model.train()
        
        batch_size = 4  # Smaller batch size to fit in memory
        data_loader = DataLoader(list(zip(train_encodings['input_ids'], train_labels)), batch_size=batch_size)

        for epoch in range(3):  # Adjust epochs as needed
            total_loss = 0  # Initialize total loss for the epoch
            for batch in data_loader:
                input_ids, labels = batch
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids.to(self.device), labels=labels.to(self.device))
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()  # Accumulate loss

            # Print the average loss for the epoch
            print(f'Epoch [{epoch + 1}/3], Average Loss: {total_loss / len(data_loader):.4f}')
        
        print("Training complete!")

        # Save the model
        torch.save(self.model.state_dict(), self.model_path)  # Save the model state
        print(f"Model saved to {self.model_path}")

        # After training, evaluate the model
        self.evaluate(test_encodings, test_labels)

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()
            print("Model loaded!")
        else:
            print("No saved model found. Please train the model first.")

    def predict(self, review_text):
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(review_text, return_tensors='pt', truncation=True, padding=True).to(self.device)
            outputs = self.model(**inputs)
            rating = outputs.logits.item()

        # Use the rule-based sentiment analysis
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
    classifier = BERTSentimentClassifier()
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