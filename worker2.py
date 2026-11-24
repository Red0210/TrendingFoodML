import os
import time
import requests
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from playwright.sync_api import sync_playwright
from datetime import datetime, timedelta, timezone

# --- CONFIGURAZIONE PARAMETRI OTTIMIZZATA ---
BASE_PROJECT_PATH = Path.home() / "Desktop" / "scrapingForFood"
FILE_NAME = BASE_PROJECT_PATH / "reddit_food_full_dataset.json"

SUBREDDIT = "food"
# Lista di cibi che statisticamente generano più engagement (Upvote Magnets)
KEYWORDS = [] 

MAX_POSTS_PER_QUERY = 100 
MAX_TOTAL_POSTS = 1000  # Aumentato per avere una base statistica solida
MAX_RUNTIME_MINUTES = 60 

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
    if not new_posts: return
    existing_df = pd.read_json(FILE_NAME) if FILE_NAME.exists() else pd.DataFrame()
    new_df = pd.DataFrame(new_posts)
    final_df = pd.concat([existing_df, new_df], ignore_index=True)
    if not final_df.empty:
        final_df.drop_duplicates(subset=['url'], keep='first', inplace=True)
    final_df.to_json(FILE_NAME, orient="records", indent=4)
    print(f"\n[LOG] Record totali nel dataset: {len(final_df)}")

def scrape_reddit_top_content(subreddit_name):
    """Scraping mirato ai post TOP e alle KEYWORD con gestione errori."""
    date_limit = datetime.now(timezone.utc) - timedelta(days=7)
    execution_start = time.time()
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            locale="en-US", 
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        # --- PARTE 1: SEZIONI TOP (Filtrate per tempo) ---
        top_sections = [] # "top/?t=week", "top/?t=month", "top/?t=year", "top/?t=all"
        
        for section in top_sections:
            if (time.time() - execution_start) / 60 >= MAX_RUNTIME_MINUTES: break
            
            print(f"\n--- ACCESSO SEZIONE: {section.upper()} ---")
            try:
                page.goto(f"https://www.reddit.com/r/{subreddit_name}/{section}", wait_until="domcontentloaded", timeout=60000)
                time.sleep(5)
                
                for _ in range(5):
                    page.keyboard.press("PageDown")
                    time.sleep(2)

                # Inizializziamo sempre la lista per evitare l'UnboundLocalError
                links = page.eval_on_selector_all(
                    'a[href*="/r/food/comments/"]', 
                    "nodes => nodes.map(n => n.href)"
                )
                unique_links = list(set(links))
                
                print(f"[DEBUG] Trovati {len(unique_links)} link nella sezione {section}")
                if unique_links:
                    process_links(page, unique_links, f"TOP_{section}", date_limit, execution_start)
            except Exception as e:
                print(f"  [ERROR] Fallimento navigazione sezione {section}: {e}")

        # --- PARTE 2: RICERCA PER KEYWORDS (Aggiunta per non far chiudere lo script) ---
        print("\n--- INIZIO RICERCA PER KEYWORDS ---")
        for query in KEYWORDS:
            if (time.time() - execution_start) / 60 >= MAX_RUNTIME_MINUTES: break
            
            print(f"\n--- RICERCA KEYWORD: {query.upper()} ---")
            try:
                page.goto(f"https://www.reddit.com/r/{subreddit_name}/search/?q={query}&restrict_sr=1&t=all", wait_until="domcontentloaded")
                time.sleep(5)
                
                for _ in range(5):
                    page.keyboard.press("PageDown")
                    time.sleep(1)

                links = page.eval_on_selector_all('a[href*="/r/food/comments/"]', "nodes => nodes.map(n => n.href)")
                unique_links = list(set(links)) # Inizializzata localmente qui
                
                if unique_links:
                    process_links(page, unique_links, f"KEY_{query}", date_limit, execution_start)
            except Exception as e:
                print(f"  [ERROR] Fallimento keyword {query}: {e}")

        browser.close()

def process_links(page, unique_links, source_tag, date_limit, start_time):
    current_batch = []
    for post_url in unique_links:
        if (time.time() - start_time) / 60 >= MAX_RUNTIME_MINUTES: break
        if len(current_batch) >= MAX_POSTS_PER_QUERY: break

        try:
            print(f"  [->] Navigazione: {post_url}")
            page.goto(post_url, wait_until="domcontentloaded")
            
            # Cerchiamo l'oggetto post principale
            page.wait_for_selector('shreddit-post', timeout=5000)
            post_obj = page.locator('shreddit-post').first
            
            # Estrazione dati con gestione errori per singolo campo
            try:
                score = int(post_obj.get_attribute('score') or 0)
                # --- FILTRO SCORE MINIMO ---
                if score < 700:
                    print(f"    [SKIP] Score troppo basso: {score}")
                    continue  # Salta il post e passa al prossimo
                # Se lo score è 0, proviamo a prenderlo dal tag specifico se shreddit fallisce
                if score == 0:
                    score_text = page.locator('[id^="score-column-"]').first.inner_text()
                    score = int(''.join(filter(str.isdigit, score_text)) or 0)
            except: score = 0

            img_url = post_obj.get_attribute('content-href')
            if not img_url or "i.redd.it" not in img_url: continue

            raw_ts = post_obj.get_attribute('created-timestamp')
            dt_obj = datetime.fromisoformat(raw_ts.replace('Z', '+00:00'))

            # Analisi Immagine
            features = analyze_image_from_url(img_url)
            if features:
                record = {
                    "post_id": post_obj.get_attribute('id'),
                    "title": post_obj.get_attribute('post-title'),
                    "score": score,
                    "comments": int(post_obj.get_attribute('comment-count') or 0),
                    "date_full": dt_obj.strftime('%Y-%m-%d %H:%M:%S'),
                    "hour": dt_obj.hour,
                    "day_of_week": dt_obj.strftime('%A'),
                    "is_weekend": dt_obj.weekday() >= 5,
                    "url": post_url,
                    "img_blurriness": features['blurriness'],
                    "img_warmth": features['warmth'],
                    "img_saturation": features['saturation'],
                    "img_centering": features['centering'],
                    "query_used": source_tag
                }
                current_batch.append(record)
                print(f"    [OK] Acquisito: {record['title'][:30]}... | Score: {score}")
        except Exception as e:
            print(f"    [SKIP] Errore nel post: {e}")
            continue
    
    if current_batch:
        save_and_append_data(current_batch)

if __name__ == "__main__":
    scrape_reddit_top_content(SUBREDDIT)