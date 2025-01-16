import pandas as pd
import numpy as np
from langdetect import detect
from deep_translator import GoogleTranslator
import nltk
nltk.download('punkt')

class ReviewProcessor:
    def __init__(self):
        # Initialize translator for non-English reviews
        self.translator = GoogleTranslator(source='auto', target='en')
        
    def detect_and_translate(self, text):
        # Translate non-English reviews to English
        try:
            lang = detect(text)
            if lang != 'en':
                translated = self.translator.translate(text)
                return translated
            return text
        except:
            return text
    
    def convert_rating(self, width_str):
        # Convert width percentage to star rating (0-5)
        try:
            if pd.isna(width_str):
                return 0
            
            width = float(width_str.split(':')[1].rstrip('%'))
            return width / 20  # Convert percentage to 5-star scale
        except (ValueError, IndexError, AttributeError) as e:
            print(f"Rating conversion error: {e} for value: {width_str}")
            return 0
    
    def preprocess_data(self, df):
        # Main preprocessing function
        processed_df = df.copy()
        
        # Remove empty and zero-rated reviews
        processed_df = processed_df[
            (processed_df['review'].notna()) & 
            (processed_df['review'] != '') & 
            (processed_df['rating'] != 'width:0%')
        ]
        
        # Convert ratings and translate reviews
        processed_df['stars'] = processed_df['rating'].apply(self.convert_rating)
        print("Translating reviews to English...")
        processed_df['review_english'] = processed_df['review'].apply(self.detect_and_translate)
        processed_df['review_clean'] = processed_df['review_english'].str.lower()
        
        # Drop the original review, rating, and review_english columns
        processed_df.drop(columns=['review', 'rating', 'review_english'], inplace=True)
        
        return processed_df

# Process the scraped reviews
df = pd.read_csv('marvel_rivals_reviews_all.csv')
processor = ReviewProcessor()
processed_df = processor.preprocess_data(df)
processed_df.to_csv('processed_game_reviews.csv', index=False)