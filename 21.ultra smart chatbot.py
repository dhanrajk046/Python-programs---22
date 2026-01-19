#!/usr/bin/env python3
"""
ultra_smart_chatbot.py

One-file, self-contained, enhanced chatbot:
- Auto-installs packages if missing (best-effort)
- Auto-downloads NLTK corpora if available
- Embedded intents (large set) for day-to-day and business scenarios
- TF-IDF + LogisticRegression model with cosine-similarity fallback
- Teach mode with persistent saving, retrain-on-teach
- Context memory, conversation logging
- Utilities: calculator, currency converter (offline rates), reminders, basic scheduling
- Optional simple Tkinter GUI (--gui)
- Works in reduced offline mode if internet or packages not available

Usage:
    python ultra_smart_chatbot.py
    python ultra_smart_chatbot.py --gui
"""

# Standard libs
import os
import sys
import json
import time
import math
import random
import argparse
import logging
import re
import threading
import subprocess
from datetime import datetime, timedelta
from collections import defaultdict, deque, Counter
from typing import List, Tuple, Optional

# -------------------------
# Auto-install helper (best-effort)
# -------------------------
def try_install(packages: List[str]):
    """Try to pip install given packages. Best-effort — works only if environment allows."""
    import importlib
    to_install = []
    for pkg in packages:
        name = pkg.split("==")[0]
        try:
            importlib.import_module(name)
        except Exception:
            to_install.append(pkg)
    if not to_install:
        return True
    print("Attempting to install missing packages:", to_install)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *to_install])
        return True
    except Exception as e:
        print("Auto-install failed:", e)
        print("Please install required packages manually:", to_install)
        return False

# List packages we will try to use
_required_pkgs = ["nltk", "numpy", "scikit-learn", "requests"]

# Attempt install (will do nothing if all exist)
try_install(_required_pkgs)

# -------------------------
# Now import (with fallback logic)
# -------------------------
USE_FULL_FEATURES = True
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet, stopwords
    from nltk import word_tokenize
except Exception:
    nltk = None
    WordNetLemmatizer = None
    wordnet = None
    stopwords = None
    word_tokenize = None
    USE_FULL_FEATURES = False

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    # degrade gracefully if sklearn/numpy missing
    np = None
    TfidfVectorizer = None
    LogisticRegression = None
    cosine_similarity = None
    USE_FULL_FEATURES = False

try:
    import requests
except Exception:
    requests = None
    # currency converter and weather fetch will be offline-only

# -------------------------
# NLTK data downloader (best-effort)
# -------------------------
def ensure_nltk_data():
    """Attempt to download required NLTK corpora. Return True if available after this call."""
    global nltk, wordnet, stopwords, word_tokenize, WordNetLemmatizer
    if not nltk:
        return False
    needed = [("tokenizers/punkt","punkt"), ("tokenizers/punkt_tab","punkt_tab"),
              ("corpora/wordnet","wordnet"), ("corpora/omw-1.4","omw-1.4"),
              ("corpora/stopwords","stopwords")]
    success = True
    for path, pkg in needed:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                print(f"Downloading NLTK data: {pkg} ...")
                nltk.download(pkg, quiet=True)
            except Exception as e:
                print(f"Failed to download {pkg}: {e}")
                success = False
    # rebind
    try:
        from nltk.stem import WordNetLemmatizer as _WNL
        from nltk.corpus import wordnet as _wn, stopwords as _sw
        from nltk import word_tokenize as _wt
        WordNetLemmatizer = _WNL
        wordnet = _wn
        stopwords = _sw
        word_tokenize = _wt
    except Exception:
        success = False
    return success

NLTK_OK = ensure_nltk_data()
if NLTK_OK:
    USE_FULL_FEATURES = USE_FULL_FEATURES and True
else:
    # If NLTK not fully available, we still proceed with fallback tokenization
    USE_FULL_FEATURES = False

# -------------------------
# Logging
# -------------------------
LOG_FILE = "ultra_chat_log.txt"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(message)s")

# -------------------------
# File paths for persistence
# -------------------------
BASE_DIR = os.path.dirname(__file__) if "__file__" in globals() else "."
LEARNED_FILE = os.path.join(BASE_DIR, "ultra_learned.json")
MEMORY_FILE = os.path.join(BASE_DIR, "ultra_memory.json")
REMINDERS_FILE = os.path.join(BASE_DIR, "ultra_reminders.json")
MODEL_META_FILE = os.path.join(BASE_DIR, "ultra_model_meta.json")

# -------------------------
# Lightweight fallback NLP utilities (used if NLTK not available)
# -------------------------
def simple_tokenize(text: str) -> List[str]:
    text = re.sub(r"[^a-zA-Z0-9'\s]", " ", text)
    return [t for t in text.split() if t]

def simple_lemmatize(word: str) -> str:
    # primitive stemming/lemmatization
    w = word.lower()
    if w.endswith("'s"):
        w = w[:-2]
    for suf in ("ing", "ed", "ly", "es", "s"):
        if len(w) > len(suf)+2 and w.endswith(suf):
            return w[:-len(suf)]
    return w

def get_synonyms(word: str, max_syn=4):
    if NLTK_OK and wordnet:
        syns = set()
        for s in wordnet.synsets(word):
            for l in s.lemmas():
                nm = l.name().replace("_"," ")
                if nm != word:
                    syns.add(nm)
                    if len(syns) >= max_syn:
                        break
            if len(syns) >= max_syn:
                break
        return list(syns)
    return []

# -------------------------
# Preprocessing helpers
# -------------------------
if NLTK_OK:
    LEMMATIZER = WordNetLemmatizer()
    STOPWORDS = set(stopwords.words("english"))
else:
    LEMMATIZER = None
    STOPWORDS = set(["a","an","the","is","are","was","were","be","been","am","in","on","at","of","for","to","and","or","but"])

def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9'\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_and_lemmatize(text: str) -> List[str]:
    t = clean_text(text)
    if NLTK_OK and word_tokenize and LEMMATIZER:
        toks = word_tokenize(t)
        out = []
        for w in toks:
            lw = w.lower()
            if lw in STOPWORDS:
                continue
            try:
                outw = LEMMATIZER.lemmatize(lw)
            except Exception:
                outw = lw
            if outw:
                out.append(outw)
        return out
    else:
        toks = simple_tokenize(t)
        return [simple_lemmatize(w) for w in toks if w.lower() not in STOPWORDS]

def preprocess_for_vector(text: str) -> str:
    return " ".join(tokenize_and_lemmatize(text))

# -------------------------
# Embedded intents — large curated set (extended for day-to-day and business)
# You can expand this list further; keys: tag, patterns, responses.
# -------------------------
EMBEDDED_INTENTS = {
    "intents": [
        # greetings
        {"tag":"greeting",
         "patterns":["hi","hello","hey","good morning","good afternoon","good evening","hiya"],
         "responses":["Hello! How can I help you today?","Hi — what can I do for you?","Hey! How can I assist?"]},
        {"tag":"goodbye",
         "patterns":["bye","goodbye","see you","talk later","catch you later"],
         "responses":["Goodbye! Have a nice day.","See you later!"]},
        {"tag":"thanks",
         "patterns":["thanks","thank you","thx","much appreciated","thanks a lot"],
         "responses":["You're welcome!","Happy to help.","No problem!"]},
        # small talk
        {"tag":"how_are_you",
         "patterns":["how are you","how's it going","how are you doing"],
         "responses":["I'm a bot, but I'm functioning well — thanks! How can I help you?"]},
        {"tag":"weather_smalltalk",
         "patterns":["how's the weather","is it raining","what's the weather like"],
         "responses":["I can't fetch live weather in offline mode, but I can show you how to check or use an API."]},
        # business / support
        {"tag":"business_hours",
         "patterns":["what are your business hours","when are you open","opening hours","work hours"],
         "responses":["Our business hours are Monday to Friday, 9 AM to 6 PM (local time)."]},
        {"tag":"contact_support",
         "patterns":["how do i contact support","support email","customer service number","contact info"],
         "responses":["You can reach support at support@example.com or call +1-800-555-1234."]},
        {"tag":"pricing",
         "patterns":["price","pricing","how much","cost","how much does it cost","what is the price"],
         "responses":["Pricing varies by product/service — tell me which product and I can give an estimate."]},
        {"tag":"refund_policy",
         "patterns":["refund","returns","how to return","return policy"],
         "responses":["Our refund policy: returns accepted within 30 days, subject to terms. We typically process refunds in 7-10 business days."]},
        {"tag":"order_status",
         "patterns":["where is my order","order status","track order","track my order by id"],
         "responses":["Please provide your order ID and I will try to locate the order (or use the Orders page)."]},
        {"tag":"cancel_order",
         "patterns":["cancel my order","i want to cancel order","how to cancel order"],
         "responses":["To cancel an order, provide the order ID. If the order is already shipped we may not be able to cancel."]},
        # scheduling
        {"tag":"book_meeting",
         "patterns":["schedule a meeting","book a meeting","set up meeting","arrange meeting"],
         "responses":["Sure — what date and time would you like? I can note it as a reminder."]},
        {"tag":"set_reminder",
         "patterns":["remind me","set a reminder","remember to","create reminder"],
         "responses":["Okay — tell me what to remind you about and when (e.g., 'remind me to call john tomorrow 10am')."]},
        # finance / invoice
        {"tag":"send_invoice",
         "patterns":["send invoice","invoice","i need an invoice","generate invoice"],
         "responses":["I can help draft an invoice message. Provide invoice number and amount."]},
        {"tag":"payment_methods",
         "patterns":["payment methods","how can i pay","payment options","pay"],
         "responses":["We accept credit card, debit card, and bank transfer. Contact billing for other methods."]},
        # product info
        {"tag":"product_features",
         "patterns":["tell me about product","features","what does it do","product description"],
         "responses":["Which product are you interested in? I can provide specs and pricing for individual items."]},
        # admin / developer
        {"tag":"teach_mode",
         "patterns":["teach","teach mode","learn this","i want to teach you"],
         "responses":["You can teach me by typing: teach: intent_tag | example phrase"]},
        {"tag":"show_intents",
         "patterns":["show intents","list intents","what can you do","capabilities"],
         "responses":["I can handle greetings, booking, reminders, simple business Q&A and more. Use 'show_intents' to see tags."]},
        # utilities
        {"tag":"calculator",
         "patterns":["calculate","what is","compute","evaluate","math"],
         "responses":["I can do calculations — type 'calc: <expression>' to evaluate (e.g., calc: 2+2*3)."]},
        {"tag":"currency_convert",
         "patterns":["convert","currency","how much is","exchange rate"],
         "responses":["Type 'convert: 100 USD to INR' or 'convert: 50 eur to gbp'. I will try to convert using built-in rates or online data." ]},
        {"tag":"news",
         "patterns":["news","latest news","what's happening","headlines"],
         "responses":["I can fetch headlines if you enable internet mode; otherwise I can summarize topics if you give me text."]},
        {"tag":"time",
         "patterns":["what time is it","current time","time now"],
         "responses":["I can tell the current local time."]},
        # fallback/default
        {"tag":"default",
         "patterns":[],
         "responses":["Sorry, I didn't understand. Could you rephrase?","I didn't get that — can you say it another way?"]}
    ]
}

# -------------------------
# Merge embedded with learned intents (persisted)
# -------------------------
def load_json_file(path, default=None):
    if default is None:
        default = {}
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Failed loading {path}: {e}")
    return default

def save_json_file(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Failed saving {path}: {e}")

learned = load_json_file(LEARNED_FILE, {"intents": []})
memory_store = load_json_file(MEMORY_FILE, {})
reminders_store = load_json_file(REMINDERS_FILE, {})

def merge_intents(embedded, learned):
    tag_map = {}
    for it in embedded.get("intents", []):
        tag_map[it["tag"]] = {"tag":it["tag"], "patterns":list(it.get("patterns",[])), "responses":list(it.get("responses",[]))}
    for it in learned.get("intents", []):
        tag = it.get("tag")
        if not tag:
            continue
        if tag in tag_map:
            tag_map[tag]["patterns"].extend(it.get("patterns", []))
            tag_map[tag]["responses"].extend(it.get("responses", []))
        else:
            tag_map[tag] = {"tag": tag, "patterns": list(it.get("patterns", [])), "responses": list(it.get("responses", []))}
    merged = {"intents": list(tag_map.values())}
    return merged

intents_data = merge_intents(EMBEDDED_INTENTS, learned)

# -------------------------
# Build dataset (with synonym expansion for robustness)
# -------------------------
def build_dataset(intents):
    X_texts = []
    y_tags = []
    for it in intents.get("intents", []):
        tag = it["tag"]
        for p in it.get("patterns", []):
            processed = preprocess_for_vector(p)
            if processed.strip():
                X_texts.append(processed)
                y_tags.append(tag)
            # lightweight synonym augmentation for single words in pattern
            words = [w for w in p.split() if len(w)>1]
            for i,w in enumerate(words):
                syns = get_synonyms(w, max_syn=2)
                for s in syns:
                    new = words[:]
                    new[i] = s
                    new_text = " ".join(new)
                    proc2 = preprocess_for_vector(new_text)
                    X_texts.append(proc2)
                    y_tags.append(tag)
    return X_texts, y_tags

X_texts, y_tags = build_dataset(intents_data)

# If dataset too small, seed basic phrases
if len(X_texts) < 10:
    for it in intents_data.get("intents", []):
        for p in it.get("patterns", []):
            X_texts.append(preprocess_for_vector(p + " please"))
            y_tags.append(it["tag"])

# -------------------------
# Vectorizer and classifier training (with robust fallbacks)
# -------------------------
if np is None or TfidfVectorizer is None or LogisticRegression is None:
    # Minimal fallback: map patterns to tags by exact token overlap scoring
    VECTOR_MODE = "fallback"
    vectorizer = None
    clf = None
    X_mat = None
else:
    VECTOR_MODE = "tfidf"
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    try:
        X_mat = vectorizer.fit_transform(X_texts)
    except Exception as e:
        # create a safe small vocabulary
        safe_texts = ["hello","bye","price","order","remind me"]
        vectorizer = TfidfVectorizer(ngram_range=(1,1))
        X_mat = vectorizer.fit_transform(safe_texts)
        y_tags = ["greeting","goodbye","pricing","order","set_reminder"]
    # train classifier
    try:
        clf = LogisticRegression(max_iter=600)
        clf.fit(X_mat, y_tags)
    except Exception as e:
        # fallback to simple frequency-based dummy classifier
        print("Classifier train failed — using fallback:", e)
        most_common = Counter(y_tags).most_common(1)[0][0] if y_tags else "default"
        class Dummy:
            def __init__(self, cls):
                self.cls = cls
                self.classes_ = np.array([cls])
            def predict(self, X):
                return [self.cls]*X.shape[0]
            def predict_proba(self, X):
                return np.ones((X.shape[0],1))
        clf = Dummy(most_common)

# Save model meta lightly
try:
    save_json_file(MODEL_META_FILE, {"vector_mode": VECTOR_MODE, "vocab_size": len(vectorizer.vocabulary_) if vectorizer else 0})
except Exception:
    pass

# -------------------------
# Utility tools: calculator, currency conversion, reminders
# -------------------------
# Calculator: evaluate safely with restricted eval
SAFE_MATH_NAMES = {k: getattr(math,k) for k in dir(math) if not k.startswith("_")}
SAFE_MATH_NAMES.update({"abs": abs, "round": round, "min": min, "max": max})

def safe_eval(expr: str):
    # keep only numbers, operators, parentheses, decimals, and math identifiers
    expr = expr.strip()
    # replace '^' with '**'
    expr = expr.replace("^", "**")
    # very basic validation
    if re.search(r"[a-zA-Z];", expr):
        raise ValueError("Invalid expression")
    # allow names from SAFE_MATH_NAMES
    code = compile(expr, "<string>", "eval")
    for name in code.co_names:
        if name not in SAFE_MATH_NAMES:
            raise ValueError(f"Use of '{name}' not allowed")
    return eval(code, {"__builtins__": {}}, SAFE_MATH_NAMES)

# Currency converter: first try online (if 'requests' available), else use static rates
STATIC_RATES = {
    # base: USD
    "USD": 1.0, "EUR": 0.92, "GBP": 0.79, "INR": 82.0, "JPY": 155.0, "AUD": 1.57, "CAD": 1.34
}
def convert_currency(amount: float, from_curr: str, to_curr: str) -> Tuple[float,str]:
    from_curr = from_curr.upper()
    to_curr = to_curr.upper()
    # try online
    if requests:
        try:
            # Use exchangerate.host (free) — no key required
            url = f"https://api.exchangerate.host/convert?from={from_curr}&to={to_curr}&amount={amount}"
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if data.get("success") and "result" in data:
                    return float(data["result"]), f"(live rates)"
        except Exception:
            pass
    # fallback to static
    if from_curr in STATIC_RATES and to_curr in STATIC_RATES:
        usd_amount = amount / STATIC_RATES[from_curr]
        converted = usd_amount * STATIC_RATES[to_curr]
        return round(converted, 4), "(static rates)"
    else:
        raise ValueError("Currency not supported in offline mode.")

# Reminders: store with timestamp; background thread to check
REMINDER_LOCK = threading.Lock()
def save_reminders():
    try:
        with REMINDER_LOCK:
            save_json_file(REMINDERS_FILE, reminders_store)
    except Exception as e:
        print("Failed saving reminders:", e)

def add_reminder(reminder_id: str, text: str, at_timestamp: float):
    reminders_store[reminder_id] = {"text": text, "at": at_timestamp, "notified": False}
    save_reminders()

def check_reminders_loop(stop_event: threading.Event):
    while not stop_event.is_set():
        now = time.time()
        to_notify = []
        with REMINDER_LOCK:
            for rid, item in list(reminders_store.items()):
                if not item.get("notified") and item.get("at", 0) <= now:
                    to_notify.append((rid, item))
        for rid, item in to_notify:
            # call notification handler (we will print to console and log)
            msg = f"Reminder: {item['text']}"
            print("\n*** " + msg + " ***\n")
            logging.info("REMINDER: " + item['text'])
            with REMINDER_LOCK:
                reminders_store[rid]["notified"] = True
                save_reminders()
        time.sleep(2)

# Start reminder background thread
reminder_stop = threading.Event()
reminder_thread = threading.Thread(target=check_reminders_loop, args=(reminder_stop,), daemon=True)
reminder_thread.start()

# -------------------------
# Core prediction + response logic
# -------------------------
CONFIDENCE_THRESHOLD = 0.55
SIMILARITY_THRESHOLD = 0.35

def find_best_similarity(user_vec) -> Tuple[float,str]:
    try:
        sims = cosine_similarity(user_vec, X_mat)
        idx = int(np.argmax(sims))
        return float(sims[0, idx]), y_tags[idx]
    except Exception:
        return 0.0, "default"

def predict_intent(user_input: str) -> Tuple[str,float]:
    proc = preprocess_for_vector(user_input)
    if VECTOR_MODE == "fallback" or vectorizer is None or clf is None:
        # fallback heuristic: tag with best token overlap
        tokens = set(proc.split())
        scores = []
        for i, txt in enumerate(X_texts):
            overlap = len(tokens.intersection(set(txt.split())))
            scores.append((overlap, y_tags[i]))
        scores.sort(reverse=True)
        if scores and scores[0][0] > 0:
            return scores[0][1], float(scores[0][0])
        return "default", 0.0
    try:
        vec = vectorizer.transform([proc])
        probs = clf.predict_proba(vec)
        idx = int(np.argmax(probs))
        tag = clf.classes_[idx]
        confidence = float(np.max(probs))
        # also compute cosine fallback score for tie-break
        sim_score, sim_tag = find_best_similarity(vec)
        # prefer classifier if above threshold, else fallback to sim if sim strong
        return tag, confidence
    except Exception:
        # fallback similarity
        vec = vectorizer.transform([proc])
        sim_score, sim_tag = find_best_similarity(vec)
        return sim_tag, sim_score

def get_response_for_tag(tag: str) -> str:
    for it in intents_data.get("intents", []):
        if it["tag"] == tag:
            resp = random.choice(it.get("responses", ["I don't have a response for that right now."]))
            return resp
    # default
    for it in intents_data["intents"]:
        if it["tag"] == "default":
            return random.choice(it.get("responses", ["I didn't understand."]))
    return "I didn't understand."

# -------------------------
# Teach mode and retraining
# -------------------------
def teach_intent(tag: str, example: str, response: Optional[str]=None):
    # Append to learned dataset and retrain immediately
    ld = load_json_file(LEARNED_FILE, {"intents":[]})
    for it in ld.get("intents", []):
        if it.get("tag") == tag:
            it.setdefault("patterns", []).append(example)
            if response:
                it.setdefault("responses", []).append(response)
            break
    else:
        newit = {"tag":tag, "patterns":[example], "responses":[response] if response else []}
        ld.setdefault("intents", []).append(newit)
    save_json_file(LEARNED_FILE, ld)
    # Merge and retrain
    global intents_data, X_texts, y_tags, X_mat, vectorizer, clf, VECTOR_MODE
    learned_local = load_json_file(LEARNED_FILE, {"intents":[]})
    intents_data = merge_intents(EMBEDDED_INTENTS, learned_local)
    X_texts, y_tags = build_dataset(intents_data)
    if len(X_texts) < 3 and VECTOR_MODE == "tfidf":
        # keep vectorizer but add minimal data
        X_texts = X_texts + ["hello", "bye", "price"]
        y_tags = y_tags + ["greeting","goodbye","pricing"]
    try:
        if VECTOR_MODE == "tfidf":
            vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
            X_mat = vectorizer.fit_transform(X_texts)
            clf = LogisticRegression(max_iter=600)
            clf.fit(X_mat, y_tags)
        else:
            # fallback
            pass
        print(f"Teach: learned '{example}' for tag '{tag}'. Model retrained.")
        return True
    except Exception as e:
        print("Teach saved but retrain failed:", e)
        return False

# -------------------------
# Knowledge base lookup
# -------------------------
KB = {
    "hours": "Business hours: Mon-Fri, 9am-6pm.",
    "contact": "support@example.com, +1-800-555-1234",
    "pricing": "We have basic, pro, and enterprise tiers. Ask which product for details.",
    "shipping": "Shipping typically 3-7 business days."
}

def kb_lookup(user_input: str) -> Optional[str]:
    u = clean_text(user_input)
    for k,v in KB.items():
        if k in u or any(word in u for word in k.split()):
            return v
    return None

# -------------------------
# Conversation context + utilities
# -------------------------
class ChatContext:
    def __init__(self):
        self.history = deque(maxlen=200)
        self.slots = {}
    def add_user(self, text):
        self.history.append(("user", text))
    def add_bot(self, text):
        self.history.append(("bot", text))
    def set_slot(self, k, v):
        self.slots[k] = v
    def get_slot(self, k, default=None):
        return self.slots.get(k, default)
    def clear(self):
        self.history.clear()
        self.slots.clear()

# Persistent memory functions
def save_memory():
    try:
        save_json_file(MEMORY_FILE, memory_store)
    except Exception:
        pass

# -------------------------
# High-level processing of user input (handles commands / utilities)
# -------------------------
def handle_command(user_input: str, ctx: ChatContext) -> Optional[str]:
    # check for special prefixed commands: calc:, convert:, teach:, remind:, show_intents, memory
    ui = user_input.strip()
    lower = ui.lower()
    if lower.startswith("calc:") or lower.startswith("calculate:") or lower.startswith("compute:"):
        expr = ui.split(":",1)[1].strip()
        try:
            val = safe_eval(expr)
            return f"Result: {val}"
        except Exception as e:
            return f"Calculation error: {e}"
    if lower.startswith("convert:"):
        body = ui.split(":",1)[1].strip()
        # expect "100 usd to inr" or "100 usd in inr"
        m = re.search(r"([0-9,.]+)\s*([a-zA-Z]{3})\s*(to|in)?\s*([a-zA-Z]{3})", body)
        if m:
            amt = float(m.group(1).replace(",",""))
            src = m.group(2)
            dst = m.group(4)
            try:
                conv, srcinfo = convert_currency(amt, src, dst)
                return f"{amt} {src.upper()} = {conv} {dst.upper()} {srcinfo}"
            except Exception as e:
                return f"Conversion failed: {e}"
        else:
            return "Conversion format: convert: 100 USD to INR"
    if lower.startswith("teach:"):
        # teach:tag|example|optional_response
        body = ui.split(":",1)[1].strip()
        parts = [p.strip() for p in body.split("|")]
        if len(parts) >= 2:
            tag = parts[0]
            example = parts[1]
            response = parts[2] if len(parts) >= 3 else None
            ok = teach_intent(tag, example, response)
            return f"Taught: '{example}' -> {tag}. Retrain status: {'OK' if ok else 'Failed'}"
        else:
            return "Teach format: teach: intent_tag | example phrase | optional response"
    if lower.startswith("remind:"):
        # remind: at YYYY-MM-DD HH:MM | message
        body = ui.split(":",1)[1].strip()
        if "|" in body:
            when_str, message = [b.strip() for b in body.split("|",1)]
            try:
                # parse common forms
                at_dt = parse_datetime_flexible(when_str)
                ts = at_dt.timestamp()
                rid = f"r{int(time.time()*1000)}"
                add_reminder(rid, message, ts)
                return f"Reminder set for {at_dt.strftime('%Y-%m-%d %H:%M:%S')}"
            except Exception as e:
                return f"Failed to parse date/time: {e}. Try 'remind: YYYY-MM-DD HH:MM | message' or 'remind: tomorrow 10am | message'"
        else:
            return "Remind format: remind: <when> | <message>"
    if lower == "show_intents":
        tags = [it["tag"] for it in intents_data.get("intents",[])]
        return "Known intents: " + ", ".join(tags)
    if lower == "show_memory":
        return "Memory keys: " + ", ".join(list(memory_store.keys()))
    if lower.startswith("remember:"):
        # remember:key|value
        body = ui.split(":",1)[1].strip()
        if "|" in body:
            key, val = [b.strip() for b in body.split("|",1)]
            memory_store[key] = val
            save_memory()
            return f"Saved memory {key} = {val}"
        else:
            return "Remember format: remember: key | value"
    if lower.startswith("forget:"):
        key = ui.split(":",1)[1].strip()
        if key in memory_store:
            del memory_store[key]
            save_memory()
            return f"Forgot {key}"
        else:
            return f"No such memory key: {key}"
    # No special command matched
    return None

# parse flexible datetime
def parse_datetime_flexible(s: str) -> datetime:
    s = s.strip().lower()
    now = datetime.now()
    # simple formats: "tomorrow 10am", "today 14:30", "2025-08-10 09:00", "in 2 hours", "in 30 minutes"
    if s.startswith("in "):
        m = re.match(r"in\s+(\d+)\s*(hour|hours|hr|hrs)", s)
        if m:
            hrs = int(m.group(1))
            return now + timedelta(hours=hrs)
        m = re.match(r"in\s+(\d+)\s*(minute|minutes|min|mins)", s)
        if m:
            mins = int(m.group(1))
            return now + timedelta(minutes=mins)
    if s.startswith("tomorrow"):
        rest = s[len("tomorrow"):].strip()
        dt = (now + timedelta(days=1)).replace(second=0, microsecond=0)
        if rest:
            # parse time like 10am or 10:30
            t = parse_time_string(rest)
            dt = dt.replace(hour=t.hour, minute=t.minute)
        return dt
    if s.startswith("today"):
        rest = s[len("today"):].strip()
        dt = now.replace(second=0, microsecond=0)
        if rest:
            t = parse_time_string(rest)
            dt = dt.replace(hour=t.hour, minute=t.minute)
        return dt
    # try ISO/datetime formats
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    # try time only
    try:
        t = parse_time_string(s)
        return now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
    except Exception:
        raise ValueError("Unrecognized date/time format")

def parse_time_string(s: str) -> datetime:
    s = s.strip().lower()
    m = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", s)
    if not m:
        raise ValueError("Time parse failed")
    h = int(m.group(1))
    mi = int(m.group(2) or 0)
    ampm = m.group(3)
    if ampm:
        if ampm == "pm" and h < 12:
            h += 12
        if ampm == "am" and h == 12:
            h = 0
    return datetime.now().replace(hour=h, minute=mi)

# -------------------------
# Main chat processing
# -------------------------
def process_user_input(user_input: str, ctx: ChatContext) -> str:
    user_input = user_input.strip()
    # check commands first
    cmd_res = handle_command(user_input, ctx)
    if cmd_res is not None:
        ctx.add_user(user_input)
        ctx.add_bot(cmd_res)
        logging.info(f"USER: {user_input} | BOT: {cmd_res}")
        return cmd_res

    # KB lookup
    kb_ans = kb_lookup(user_input)
    if kb_ans:
        ctx.add_user(user_input)
        ctx.add_bot(kb_ans)
        logging.info(f"USER: {user_input} | BOT: {kb_ans}")
        return kb_ans

    # predict intent
    tag, conf = predict_intent(user_input)
    # If classifier returns default or low confidence, try similarity fallback
    if conf < CONFIDENCE_THRESHOLD:
        # try similarity
        try:
            proc = preprocess_for_vector(user_input)
            vec = vectorizer.transform([proc])
            sim_score, sim_tag = find_best_similarity(vec)
            if sim_score >= SIMILARITY_THRESHOLD:
                tag = sim_tag
                conf = sim_score
        except Exception:
            pass

    if conf >= CONFIDENCE_THRESHOLD:
        resp = get_response_for_tag(tag)
        ctx.add_user(user_input)
        ctx.add_bot(resp)
        logging.info(f"USER: {user_input} | INTENT: {tag} | CONF: {conf:.2f} | BOT: {resp}")
        return resp
    else:
        # low confidence — ask clarifying or fallback default
        default_resp = get_response_for_tag("default")
        ctx.add_user(user_input)
        ctx.add_bot(default_resp)
        logging.info(f"USER: {user_input} | BOT: {default_resp} | raw_intent:{tag} conf:{conf:.2f}")
        return default_resp

# -------------------------
# CLI and GUI interfaces
# -------------------------
def run_cli():
    print("Ultra Smart Chatbot (single-file). Type 'quit' to exit.")
    print("Commands: calc:, convert:, teach:, remind:, show_intents, show_memory, remember:, forget:")
    ctx = ChatContext()
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye.")
            return
        if not user_input:
            continue
        if user_input.lower() in ("quit","exit","bye"):
            print("Bot: Goodbye.")
            return
        out = process_user_input(user_input, ctx)
        print("Bot:", out)

def run_gui():
    try:
        import tkinter as tk
        from tkinter import scrolledtext
    except Exception as e:
        print("Tkinter not available:", e)
        run_cli()
        return
    root = tk.Tk()
    root.title("Ultra Smart Chatbot")
    chat_area = scrolledtext.ScrolledText(root, state="disabled", width=90, height=30, wrap="word")
    chat_area.grid(row=0, column=0, columnspan=2, padx=8, pady=8)
    entry_var = tk.StringVar()
    entry = tk.Entry(root, textvariable=entry_var, width=74)
    entry.grid(row=1, column=0, padx=8, pady=8)
    ctx = ChatContext()
    def append(sender, text):
        chat_area.configure(state="normal")
        chat_area.insert("end", f"{sender}: {text}\n")
        chat_area.configure(state="disabled")
        chat_area.yview("end")
    def on_send():
        txt = entry_var.get().strip()
        if not txt:
            return
        append("You", txt)
        entry_var.set("")
        if txt.lower() in ("quit","exit","bye"):
            append("Bot","Goodbye.")
            root.after(200, root.destroy)
            return
        res = process_user_input(txt, ctx)
        append("Bot", res)
    btn = tk.Button(root, text="Send", command=on_send)
    btn.grid(row=1, column=1, padx=6, pady=6)
    entry.bind("<Return>", lambda e: on_send())
    root.mainloop()

# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Run simple GUI")
    args = parser.parse_args()
    print("Starting Ultra Smart Chatbot...")
    if NLTK_OK:
        print("NLTK resources available: full NLP features enabled.")
    else:
        print("NLTK resources not fully available: running in reduced offline mode.")
    if VECTOR_MODE == "fallback":
        print("TF-IDF/Classifier not available; using fallback matching.")
    try:
        if args.gui:
            run_gui()
        else:
            run_cli()
    finally:
        # ensure reminder thread is stopped gracefully
        reminder_stop.set()
        reminder_thread.join(timeout=1)
        print("Chatbot stopped.")
