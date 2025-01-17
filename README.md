# Game Review Sentiment Classifier

This project is a sentiment analysis system for game reviews, combining machine learning and rule-based approaches to analyze player feedback from backloggd.com.

## About The Project

This project provides an end-to-end solution for analyzing game reviews through:
1. Web scraping reviews from backloggd.com
2. Cleaning and processing the review data
3. Performing sentiment analysis using a hybrid approach

### Key Features
* Automated web scraping of game reviews
* Multi-language support with automatic translation
* Hybrid sentiment analysis combining ML and gaming-specific rules
* Rating prediction (0-5 stars)
* Sentiment classification (Positive/Neutral/Negative)

### Built With
* Python 3.x
* Selenium (Web Scraping)
* BeautifulSoup4 (HTML Parsing)
* Pandas (Data Processing)
* Scikit-learn (Machine Learning)
* NLTK (Text Processing)
* Google Translator (Language Translation)

## Getting Started

### Prerequisites & Installation
1. Ensure you have Python 3.x installed
2. Clone the repository
```sh
git clone https://github.com/gilelliba/game-review-sentiment-classifier.git
```

3. Install required Python packages
```sh
pip install -r requirements.txt
```

## Usage

### 1. Scraping Reviews
```sh
python WebScraping.py
```
This will:
- Scrape reviews from backloggd.com
- Save raw data to `marvel_rivals_reviews_all.csv`

### 2. Processing Reviews
```sh
python DataProcessing.py
```
This will:
- Clean and standardize review text
- Translate non-English reviews
- Convert ratings to 5-star scale
- Save processed data to `processed_game_reviews.csv`

### 3. Sentiment Analysis
```sh
python SentimentClassifier.py
```
This will:
- Train the sentiment model
- Start an interactive prompt for review analysis
- Example usage:
```
Enter a game review (or 'quit' to exit): This game is amazing but has some lag issues

Analysis Results:
Rating: 3.5 stars
Sentiment: Positive
Positive words found: 1
Negative words found: 1
```

## Project Structure
```
project/
│
├── WebScraping.py      # Web scraper for game reviews
├── DataProcessing.py   # Data cleaning and preprocessing
├── SentimentClassifier.py  # Sentiment analysis model
└── README.md           # Project documentation
```

## Future Improvements
- Consider gathering additional review data from other gaming websites to improve model accuracy:
  - Steam reviews (via Steam API)
  - Metacritic user reviews
  - IGN community reviews
  - GameSpot user reviews
  - OpenCritic aggregated reviews
  
This would provide:
- A larger training dataset
- More diverse perspectives
- Better handling of gaming-specific terminology
- Reduced bias from any single platform's user base
- More robust sentiment predictions

## Notes
- Requires Chrome WebDriver (usually pre-installed in most development environments)
- The scraper is configured for backloggd.com and may need adjustments if the site structure changes
- Translation services require an internet connection
- Large datasets may require significant processing time
- The sentiment analysis uses a hybrid approach combining ML predictions with gaming-specific vocabulary
- Additional data sources may require modifications to the scraping and processing scripts to handle different site structures and review formats

---

## Part 2: Sentiment Analysis Model Performance

In this section, we look at how different sentiment analysis models (FFN, LSTM, and BERT) performed when combined with rule-based sentiment analysis.

1. **FFN Model**:
```sh
python FFNSentimentClassifier.py
```
   - The Feedforward Neural Network (FFN) does a decent job on its own, but it gets much better when paired with rule-based methods. This combination helps capture the nuances in sentiment that the FFN might miss.
   - Unlike the LSTM and BERT models, the FFN model does not save the trained model after training because it does not take much time to train. Therefore, you can easily retrain it whenever needed without significant delays.

2. **LSTM Model**:
```sh
python LSTMSentimentClassifier.py
```
   - The Long Short-Term Memory (LSTM) model struggles when used alone, showing poor predictions. However, when combined with rule-based analysis, it performs significantly better, indicating that the rules help fill in the gaps in the LSTM's understanding. 
   - When running the LSTM model, it saves the trained model after training, so you don't have to retrain it every time you want to use it, which can be time-consuming.

3. **BERT Model**:
```sh
python BERTSentimentClassifier.py
```
   - The BERT model, known for its strong performance in NLP tasks, also has issues when used by itself. It may need fine-tuning on specific datasets to improve.
   - When running the BERT model, it saves the trained model after training, so you don't have to retrain it every time you want to use it, which can be time-consuming.
   - I tried using the BERT model with and without rule-based sentiment analysis, and it performed better with the hybrid approach, demonstrating the value of combining machine learning with rule-based methods.

### Overall Insights
- **Combining Approaches**: The results show that using machine learning models alongside rule-based methods can enhance performance. Each model has its strengths, but the rules provide valuable context.
  
- **Need for Fine-Tuning**: The LSTM and BERT models may require additional training on specific data to improve their accuracy.

- **Hybrid Models**: The success of combining rule-based and machine learning approaches suggests a promising direction for future sentiment analysis work, potentially leading to more accurate results.

### Conclusion
In summary, while each model has its limitations, integrating them with rule-based sentiment analysis can significantly boost their performance. This highlights the importance of using diverse methods in sentiment analysis for better outcomes. Further fine-tuning and optimization could lead to even stronger solutions.

---