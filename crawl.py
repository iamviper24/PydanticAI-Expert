import os
import requests
from xml.etree import ElementTree
from bs4 import BeautifulSoup
import time
import re

SITEMAP_URL = "https://ai.pydantic.dev/sitemap.xml"
OUTPUT_DIR = "crawled_content"

def url_to_filename(url: str) -> str:
    filename = re.sub(r'^https?://', '', url)
    filename = re.sub(r'[/\\?%*:|"<>]', '_', filename)
    return f"{filename}.txt" 

def get_sitemap_urls() -> list[str]:
    print(f"Fetching sitemap from: {SITEMAP_URL}")
    try:
        response = requests.get(SITEMAP_URL, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        print(f"Found {len(urls)} URLs.")
        return urls
    except Exception as e:
        print(f"FATAL: Error fetching sitemap: {e}")
        return []

def scrape_and_save(urls: list[str]):
    print("\n Stage 1: Synchronous Crawling and Saving to Files ")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_urls = len(urls)
    success_count = 0
    
    for i, url in enumerate(urls):
        try:
            print(f"  [{i+1}/{total_urls}] Scraping: {url}")
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'lxml') 
            main_content = soup.find('main') # The Documentation for the topic is embedded in the <main> tag
            
            if main_content:
                text_content = main_content.get_text(separator='\n', strip=True)

                filename = url_to_filename(url) #Save it after processing it
                filepath = os.path.join(OUTPUT_DIR, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                success_count += 1
            else:
                print(f"  - WARNING: Could not find <main> tag for {url}. Skipping.")

            time.sleep(0.1) # To prevent from making rapid requests to the server

        except requests.RequestException as e:
            print(f"  - ERROR: Could not fetch {url}: {e}")
    
    print("\nCrawling Summary:")
    print(f"  - Successfully saved: {success_count} files")
    print(f"  - Total URLs attempted: {total_urls}")

def main():
    urls = get_sitemap_urls()
    if urls:
        scrape_and_save(urls)
        print("\n--- Crawling Complete! ---")
        print("You can now run the embedding script: python 2_embed.py")
    else:
        print("Halting: No URLs were found.")

if __name__ == "__main__":
    main()