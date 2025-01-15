from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time

class MarvelRivalsReviewScraper:
    def __init__(self):
        # Initialize headless Chrome browser
        self.options = webdriver.ChromeOptions()
        self.options.add_argument('--headless')
        self.driver = webdriver.Chrome(options=self.options)
        self.base_url = "https://backloggd.com/reviews/everyone/week/recent/marvel-rivals/"
        self.reviews_data = []

    def extract_reviews(self, html):
        # Parse HTML and extract review elements
        soup = BeautifulSoup(html, 'html.parser')
        review_elements = soup.select('.pt-2 > div > div')
        
        for review in review_elements:
            try:
                # Extract username, review text, and rating
                username_elem = review.find('p', class_='mb-0')
                username = username_elem.text.strip() if username_elem else "Unknown"
                
                review_text_elem = review.find('div', class_='mb-0 card-text')
                review_text = review_text_elem.text.strip() if review_text_elem else ""
                
                rating_elem = review.find('div', class_='stars-top')
                rating = rating_elem['style'] if rating_elem else "width:0%"
                
                self.reviews_data.append({
                    'username': username,
                    'review': review_text,
                    'rating': rating
                })
            except AttributeError as e:
                print(f"Skipping malformed review: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error extracting review: {e}")
                continue

    def click_load_more(self):
        # Click 'Load more reviews' button to get additional reviews
        try:
            load_more = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "a.btn.btn-sm.btn-general[rel='next'][aria-label='next']"))
            )
            load_more.click()
            time.sleep(2)  # Wait for new content to load
            return True
        except Exception as e:
            print(f"No more reviews to load or error: {e}")
            return False

    def scrape_all_reviews(self, max_clicks=None):
        # Main scraping function to collect all reviews
        try:
            self.driver.get(self.base_url)
            time.sleep(3)  # Initial page load wait
            
            clicks = 0
            while True:
                self.extract_reviews(self.driver.page_source)
                
                if max_clicks and clicks >= max_clicks:
                    print(f"Reached maximum number of clicks ({max_clicks})")
                    break
                
                if not self.click_load_more():
                    break
                
                clicks += 1
                print(f"Clicked Load More {clicks} times. Total reviews so far: {len(self.reviews_data)}")
                
        except Exception as e:
            print(f"Error during scraping: {e}")
        finally:
            self.driver.quit()
        
        return pd.DataFrame(self.reviews_data)

# Initialize scraper and collect reviews
scraper = MarvelRivalsReviewScraper()
df = scraper.scrape_all_reviews()
df = df.drop_duplicates()
df.to_csv('marvel_rivals_reviews_all.csv', index=False)
print(f"Total unique reviews collected: {len(df)}")