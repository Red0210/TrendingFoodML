import os
import time
import requests
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from playwright.sync_api import sync_playwright
from datetime import datetime, timedelta, timezone

# --- CONFIGURAZIONE PARAMETRI ---
BASE_PROJECT_PATH = Path.home() / "Desktop" / "scrapingForFood"
FILE_NAME = BASE_PROJECT_PATH / "reddit_food_full_dataset.json"

SUBREDDIT = "food"
KEYWORDS = [] 

MAX_POSTS_PER_KEYWORD = 50 
MAX_TOTAL_POSTS = 300
MAX_RUNTIME_MINUTES = 30 

if not BASE_PROJECT_PATH.exists():
    BASE_PROJECT_PATH.mkdir(parents=True, exist_ok=True)

def analyze_image_from_url(url):
    """Analisi Computer Vision in RAM tramite OpenCV."""
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            image_array = np.asarray(bytearray(response.content), dtype="uint8")
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if img is None: return None

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurriness = cv2.Laplacian(gray, cv2.CV_64F).var()

            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            _, _, b_channel = cv2.split(lab)
            warmth = b_channel.mean()

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            _, s_channel, _ = cv2.split(hsv)
            saturation = s_channel.mean()

            M = cv2.moments(gray)
            dist_centro = 0
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                H, W = img.shape[:2]
                dist_centro = np.sqrt((cX - W/2)**2 + (cY - H/2)**2)

            return {
                "blurriness": round(blurriness, 2),
                "warmth": round(warmth, 2),
                "saturation": round(saturation, 2),
                "centering": round(dist_centro, 2)
            }
    except Exception as e:
        print(f"  [ERROR] Analisi fallita per {url}: {e}")
    return None

def save_and_append_data(new_posts):
    """Gestisce la persistenza dei dati con logica di merge e deduplicazione."""
    if not new_posts: return
    if FILE_NAME.exists():
        try:
            existing_df = pd.read_json(FILE_NAME)
        except:
            existing_df = pd.DataFrame()
    else:
        existing_df = pd.DataFrame()

    new_df = pd.DataFrame(new_posts)
    final_df = pd.concat([existing_df, new_df], ignore_index=True)
    if not final_df.empty:
        final_df.drop_duplicates(subset=['url'], keep='first', inplace=True)
    
    final_df.to_json(FILE_NAME, orient="records", indent=4)
    print(f"\n[LOG] Persistenza JSON completata. Record totali: {len(final_df)}")

def scrape_reddit_multi_keyword(subreddit_name):
    """Orchestrazione dello scraping con filtri temporali e Computer Vision."""
    # Imposta il limite di tempo rispetto a oggi
    date_limit = datetime.now(timezone.utc) - timedelta(days=7)
    execution_start = time.time()
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(locale="en-US")
        page = context.new_page()

        for query in KEYWORDS:
            elapsed = (time.time() - execution_start) / 60
            if elapsed >= MAX_RUNTIME_MINUTES: break

            print(f"\n--- ELABORAZIONE QUERY: {query.upper()} ---")
            page.goto(f"https://www.reddit.com/r/{subreddit_name}/search/?q={query}&restrict_sr=1&t=all")
            time.sleep(3)
            
            for _ in range(3):
                page.mouse.wheel(0, 2000)
                time.sleep(1)

            links = [n.get_attribute('href') for n in page.locator('a[data-testid="post-title"]').all()]
            unique_links = list(set([f"https://www.reddit.com{l}" if l.startswith('/') else l for l in links]))

            current_batch = []
            for post_url in unique_links:
                if len(current_batch) >= MAX_POSTS_PER_KEYWORD: break

                try:
                    page.goto(post_url)
                    page.wait_for_selector('shreddit-post', timeout=8000)
                    post_object = page.locator('shreddit-post').first
                    
                    # Filtro tipo di post: solo immagini
                    if post_object.get_attribute('post-type') != 'image': continue
                    image_url = post_object.get_attribute('content-href')
                    if not image_url or "i.redd.it" not in image_url: continue

                    # Estrazione e validazione data del post
                    raw_ts = post_object.get_attribute('created-timestamp')
                    dt_object = datetime.fromisoformat(raw_ts.replace('Z', '+00:00'))

                    # --- FILTRO TEMPORALE AGGIUNTO ---
                    # Salva il post solo se è più vecchio di 30 giorni (dt_object < date_limit)
                    if dt_object < date_limit:
                        img_features = analyze_image_from_url(image_url)
                        
                        if img_features:
                            # --- LOGICA DI ESTRAZIONE TESTO OTTIMIZZATA ---
                            post_text = ""
                            try:
                                # Prova a cercare il corpo del testo specifico di Shreddit
                                text_element = post_object.locator('div[slot="text-body"], #post-rtjson-content').first
                                if text_element.count() > 0:
                                    post_text = text_element.inner_text().strip()
                                else:
                                    # Tentativo secondario se il primo fallisce (selettore più generico)
                                    text_element = page.locator('div[data-click-id="text"]').first
                                    if text_element.count() > 0:
                                        post_text = text_element.inner_text().strip()
                            except:
                                post_text = ""

                            record = {
                                "post_id": post_object.get_attribute('id'),
                                "title": post_object.get_attribute('post-title'),
                                "author": post_object.get_attribute('author'),
                                "post_text": post_text,
                                "score": int(post_object.get_attribute('score') or 0),
                                "comments": int(post_object.get_attribute('comment-count') or 0),
                                "date_full": dt_object.strftime('%Y-%m-%d %H:%M:%S'),
                                "hour": dt_object.hour,
                                "day_of_week": dt_object.strftime('%A'),
                                "is_weekend": dt_object.weekday() >= 5,
                                "url": post_url,
                                "image_url": image_url,
                                "query_used": query,
                                "img_blurriness": img_features['blurriness'],
                                "img_warmth": img_features['warmth'],
                                "img_saturation": img_features['saturation'],
                                "img_centering": img_features['centering']
                            }
                            current_batch.append(record)
                            print(f"  [{len(current_batch)}] Analizzato: {record['post_id']} (Data: {record['date_full']})")
                    else:
                        print(f"  [SKIP] Post troppo recente: {dt_object.strftime('%Y-%m-%d')}")

                except Exception: continue
            
            if current_batch:
                save_and_append_data(current_batch)

        browser.close()

if __name__ == "__main__":
    scrape_reddit_multi_keyword(SUBREDDIT)