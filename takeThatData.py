import requests
import json
import csv
import time
from bs4 import BeautifulSoup

def scrape_reddit() -> list[dict]:
    subreddits = [
        "https://www.reddit.com/r/food"
    ]
# "https://www.reddit.com/r/food/top"

    all_data = []
    for url in subreddits:
        headers = {
            'User-Agent':
        }
        subreddit_name = url.split("/")[-1]

        try:
            response = request.get(url, headers=headers, timeout=10)

            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            subreddit_data ={
                "subreddit_name" : subreddit_name,
                "url" : url
                "title" : soup.title.string

            }
