import os
import json
import requests
import datetime as dt

# -----------------------------
# Configurazione
# -----------------------------
SUBREDDIT = "food"
DATA_FILE = "data_all_posts.json"
IMAGES_DIR = "images"
LIMIT = 300  # max post per run
BOUND_DAYS = 180  # limite dei giorni da considerare

os.makedirs(IMAGES_DIR, exist_ok=True)

# -----------------------------
# Calcolo periodo in timestamp UTC
# -----------------------------
END = END = int((dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=BOUND_DAYS)).timestamp())
START = None


# -----------------------------
# Carica dataset esistente
# -----------------------------
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)
else:
    dataset = []

existing_ids = set(post["id"] for post in dataset)

# -----------------------------
# Funzione: scarica immagine
# -----------------------------
def download_image(url):
    if "i.redd.it" not in url:
        return None
    filename = url.split("/")[-1]
    path = os.path.join(IMAGES_DIR, filename)
    if os.path.exists(path):
        return path
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
            return path
    except:
        return None
    return None

# -----------------------------
# Funzione: ottieni post (Pushshift o fallback Reddit JSON)
# -----------------------------
def fetch_posts(subreddit, limit=LIMIT, start=START, end=END):
    posts = []
    # Proviamo Pushshift
    try:
        from psaw import PushshiftAPI
        api = PushshiftAPI()
        gen = api.search_submissions(
            subreddit=subreddit,
            after=start,
            before=end,
            limit=limit,
            filter=[
                "id","title","selftext","score",
                "num_comments","created_utc",
                "url","post_hint"
            ]
        )
        for p in gen:
            posts.append(p.d_)
        if posts:
            print(f"Pushshift: trovati {len(posts)} post")
            return posts
    except:
        print("Pushshift non disponibile, uso JSON pubblico Reddit")

    # Fallback JSON pubblico Reddit
    url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={limit}"
    headers = {"User-Agent": f"scraper:{subreddit}:v1.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()
        for item in data["data"]["children"]:
            posts.append(item["data"])
        print(f"Reddit JSON: trovati {len(posts)} post")
    except:
        print("Errore: impossibile ottenere post")
    return posts

# -----------------------------
# Scarica e aggiorna dataset
# -----------------------------
new_posts = fetch_posts(SUBREDDIT)

UPVOTE_BUCKETS = {
    "0_50": (0, 50),
    "51_1000": (51, 1000),
    "1001_8000": (1001, 8000),
    "8001_25000": (8001, 25000),
    "25001_plus": (25001, float("inf"))
}

TARGET_PER_BUCKET = 60
buckets = {k: [] for k in UPVOTE_BUCKETS}

def get_bucket(score):
    for name, (low, high) in UPVOTE_BUCKETS.items():
        if low <= score <= high:
            return name
    return None

for d in new_posts:
    post_id = d.get("id")
    if not post_id or post_id in existing_ids:
        continue

    score = d.get("score", 0)
    bucket = get_bucket(score)
    if bucket is None:
        continue

    if len(buckets[bucket]) >= TARGET_PER_BUCKET:
        continue

    image_path = None
    if d.get("post_hint") == "image" and "i.redd.it" in d.get("url", ""):
        image_path = download_image(d["url"])

    post_obj = {
        "id": post_id,
        "title": d.get("title", ""),
        "text": d.get("selftext", ""),
        "score": score,
        "num_comments": d.get("num_comments", 0),
        "created_utc": d.get("created_utc"),
        "image_path": image_path,
        "upvote_bucket": bucket
    }

    buckets[bucket].append(post_obj)
    existing_ids.add(post_id)

    if all(len(b) >= TARGET_PER_BUCKET for b in buckets.values()):
        break


added_count = 0
for bucket_posts in buckets.values():
    for post in bucket_posts:
        dataset.append(post)
        added_count += 1


# -----------------------------
# Salva dataset aggiornato
# -----------------------------
with open(DATA_FILE, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"Fatto. Aggiunti {added_count} nuovi post. Totale dataset: {len(dataset)}")
