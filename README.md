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




