# customer_helper_chatbot.py
"""
Customer Helper Chatbot (single-file)
- Self-contained intelligent assistant for customer support and business workflows.
- Auto-downloads NLTK data as possible, auto-installs missing packages (best-effort).
- Large embedded intents for customer service scenarios.
- Immediate retrain-on-teach to learn new examples on the fly.
- Persistent learned patterns, user memory, reminders & logs.
- CLI and optional Tkinter GUI.

Usage:
    python customer_helper_chatbot.py
    python customer_helper_chatbot.py --gui
"""

import os
import sys
import json
import time
import re
import threading
import random
import subprocess
import logging
from datetime import datetime, timedelta
from collections import deque, Counter
from typing import List, Tuple, Optional

# ---------------------------
# Best-effort auto-install
# ---------------------------
def try_pip_install(pkgs: List[str]):
    """Try to pip install missing packages. Best-effort: may fail due to permission/network."""
    import importlib
    to_install = []
    for p in pkgs:
        name = p.split("==")[0]
        try:
            importlib.import_module(name)
        except Exception:
            to_install.append(p)
    if not to_install:
        return True
    print("Attempting to install packages:", to_install)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *to_install])
        return True
    except Exception as e:
        print("Auto-install failed:", e)
        print("Please install the packages manually:", to_install)
        return False

# Attempt to ensure core packages exist; harmless if already installed
try_pip_install(["nltk", "numpy", "scikit-learn", "requests"])

# ---------------------------
# Imports with graceful fallback
# ---------------------------
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
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    np = None
    TfidfVectorizer = None
    LogisticRegression = None
    cosine_similarity = None
    USE_FULL_FEATURES = False

try:
    import requests
except Exception:
    requests = None
    # offline fallback for currency, weather, etc.

# ---------------------------
# NLTK data auto-download helper
# ---------------------------
def ensure_nltk_data():
    global nltk, wordnet, stopwords, word_tokenize, WordNetLemmatizer
    if not nltk:
        return False
    required = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("corpora/stopwords", "stopwords"),
    ]
    ok = True
    for path, pkg in required:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                print(f"Downloading NLTK data: {pkg} ...")
                nltk.download(pkg, quiet=True)
            except Exception as e:
                print("Download failed:", e)
                ok = False
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
        ok = False
    return ok

NLTK_OK = ensure_nltk_data()
if not NLTK_OK:
    USE_FULL_FEATURES = False

# ---------------------------
# Logging & persistence paths
# ---------------------------
LOG_FILE = "customer_chat_log.txt"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(message)s")

BASE_DIR = os.path.dirname(__file__) if "__file__" in globals() else "."
LEARNED_FILE = os.path.join(BASE_DIR, "learned_customer_intents.json")
MEMORY_FILE = os.path.join(BASE_DIR, "customer_memory.json")
REMINDERS_FILE = os.path.join(BASE_DIR, "customer_reminders.json")
MODEL_META_FILE = os.path.join(BASE_DIR, "customer_model_meta.json")

# ---------------------------
# Lightweight NLP fallbacks
# ---------------------------
def simple_tokenize(text: str) -> List[str]:
    text = re.sub(r"[^a-zA-Z0-9'\s]", " ", text)
    return [t for t in text.split() if t]

def simple_lemmatize(word: str) -> str:
    w = word.lower()
    if w.endswith("'s"): w = w[:-2]
    for suf in ("ing", "ed", "ly", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[:-len(suf)]
    return w

def get_wordnet_synonyms(word: str, max_syn=3) -> List[str]:
    if NLTK_OK and wordnet:
        syns = set()
        for s in wordnet.synsets(word):
            for l in s.lemmas():
                nm = l.name().replace("_", " ")
                if nm != word:
                    syns.add(nm)
                    if len(syns) >= max_syn:
                        break
            if len(syns) >= max_syn:
                break
        return list(syns)
    else:
        return []

# ---------------------------
# Preprocessing
# ---------------------------
if NLTK_OK:
    LEMMATIZER = WordNetLemmatizer()
    STOPWORDS = set(stopwords.words("english"))
else:
    LEMMATIZER = None
    STOPWORDS = set(["a","an","the","is","are","am","in","on","at","of","for","to","and","or","but"])

def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9'\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_and_lemmatize(text: str) -> List[str]:
    t = clean_text(text)
    if NLTK_OK and word_tokenize and LEMMATIZER:
        tokens = word_tokenize(t)
        out = []
        for w in tokens:
            lw = w.lower()
            if lw in STOPWORDS: continue
            try:
                le = LEMMATIZER.lemmatize(lw)
            except Exception:
                le = lw
            out.append(le)
        return out
    else:
        toks = simple_tokenize(t)
        return [simple_lemmatize(w) for w in toks if w.lower() not in STOPWORDS]

def preprocess_for_vector(text: str) -> str:
    return " ".join(tokenize_and_lemmatize(text))

# ---------------------------
# Embedded comprehensive customer intents
# (This is an expanded set to cover many customer scenarios.)
# ---------------------------
EMBEDDED_INTENTS = {
    "intents": [
        # Greetings & small talk
        {"tag":"greeting","patterns":["hi","hello","hey","good morning","good afternoon","good evening"],"responses":["Hello! How can I assist you today?","Hi there — how can I help?"]},
        {"tag":"goodbye","patterns":["bye","goodbye","see you","talk later","thanks bye"],"responses":["Goodbye! If you need anything else just ask.","Thanks for contacting us — have a great day!"]},
        {"tag":"thanks","patterns":["thanks","thank you","thx","appreciate it"],"responses":["You're welcome! Happy to help."]},
        {"tag":"how_are_you","patterns":["how are you","how is it going","how are you doing"],"responses":["I'm here and ready to help — what do you need?"]},

        # Account issues
        {"tag":"password_reset","patterns":["i forgot my password","reset my password","can't login","forgot password","password reset link"],"responses":["I can help with that. Do you want me to send a password reset link to your registered email?"]},
        {"tag":"account_locked","patterns":["account locked","my account is locked","can't access account","locked out"],"responses":["I'm sorry — your account might be locked for security. Can you confirm your registered email or phone so I can escalate?"]},
        {"tag":"update_account_details","patterns":["change my email","update phone number","update address","change account details"],"responses":["Tell me which detail you'd like to update and the new value (e.g., 'change email to me@example.com')."]},

        # Billing & invoices
        {"tag":"billing_issue","patterns":["billing issue","charged twice","incorrect charge","billing error","wrong amount"],"responses":["I can help with billing. Please provide the order ID or the billing date and amount charged."]},
        {"tag":"request_invoice","patterns":["send invoice","get invoice","invoice request","email invoice"],"responses":["Please provide the order ID and billing email. I will prepare an invoice request for you."]},
        {"tag":"payment_failure","patterns":["payment failed","card declined","can't pay","payment error"],"responses":["I'm sorry for the inconvenience. Which payment method did you use? If card, check expiry and CVV and try again."]},
        {"tag":"refund_request","patterns":["refund","i want a refund","how to get a refund","return my money"],"responses":["I can start a refund. Please share your order ID and reason for refund. Refunds usually take 7-10 business days."]},

        # Orders & shipping
        {"tag":"order_status","patterns":["where is my order","track my order","order status","track order"],"responses":["Please provide the order ID and I will look it up for you."]},
        {"tag":"shipping_time","patterns":["how long to ship","shipping time","delivery time","when will it arrive"],"responses":["Typical shipping time is 3-7 business days. For specific orders, provide an order ID."]},
        {"tag":"missing_item","patterns":["missing item in my order","item missing","not received item","missing product"],"responses":["I'm sorry. Please give your order ID and list the missing items. We'll investigate and offer a resolution."]},
        {"tag":"damaged_item","patterns":["item damaged","received broken item","product damaged"],"responses":["I'm sorry to hear that. Could you provide order ID and upload a photo of the damage? We'll start a replacement or refund."]},
        {"tag":"cancel_order","patterns":["cancel my order","i want to cancel","cancel order"],"responses":["Provide the order ID — I'll check whether it can be cancelled (we can cancel if it hasn't shipped)."]},
        {"tag":"change_order","patterns":["change my order","modify order","update shipping address for order"],"responses":["Tell me the order ID and the changes (e.g., change address to ...). We'll check if modification is possible."]},

        # Returns & exchanges
        {"tag":"return_policy","patterns":["how to return","return policy","return item","returns"],"responses":["You can return items within 30 days. Items must be in original condition. Would you like to start a return request?"]},
        {"tag":"exchange_item","patterns":["exchange item","i want an exchange","replace item"],"responses":["Provide your order ID and which item you want to exchange and the desired replacement."]},

        # Subscription & plans
        {"tag":"subscription_cancel","patterns":["cancel subscription","stop subscription","end my subscription"],"responses":["I'm sorry to see you go. Can you confirm your subscription ID or account email so I can process cancellation?"]},
        {"tag":"subscription_upgrade","patterns":["upgrade plan","switch to pro","upgrade subscription"],"responses":["I can help upgrade your plan. Which plan would you like to upgrade to (Pro/Enterprise)?"]},
        {"tag":"trial_help","patterns":["trial expired","extend trial","trial not working","can't start trial"],"responses":["Please share your account email and I'll check trial status and options."]},

        # Technical support
        {"tag":"app_crash","patterns":["app crashed","application crashed","app keeps crashing","it freezes"],"responses":["Sorry — which device and OS are you using? Please provide steps to reproduce or any error message."]},
        {"tag":"connectivity_issue","patterns":["can't connect","connection issue","network error","failed to connect"],"responses":["Try restarting the app and your device. Are you on Wi-Fi or mobile data? If issue persists, provide any error code shown."]},
        {"tag":"installation_help","patterns":["install help","how to install","setup","installation error"],"responses":["Which platform (Windows/macOS/Linux/iOS/Android)? I can provide step-by-step installation instructions."]},

        # Security & privacy
        {"tag":"privacy_policy","patterns":["privacy policy","data protection","gdpr","how do you use my data"],"responses":["You can view our privacy policy at example.com/privacy. We only use data per the policy and you can request deletion."]},
        {"tag":"delete_my_data","patterns":["delete my data","remove my account","erase my data"],"responses":["I can help request data deletion. Please confirm the account email to proceed and note this may be irreversible."]},

        # Promotions & coupons
        {"tag":"promo_coupon","patterns":["coupon","promo code","discount code","apply coupon"],"responses":["Provide the coupon code and the order or product you want to apply it to; I'll check eligibility."]},
        {"tag":"loyalty_points","patterns":["loyalty points","reward points","how many points do i have","redeem points"],"responses":["I can check your points if you provide your account email. Points can be redeemed at checkout."]},

        # Escalation & human agent
        {"tag":"speak_agent","patterns":["talk to human","speak to agent","agent","human agent","live agent"],"responses":["I can connect you to a human agent. Please confirm your contact preference (phone/email) and available times."]},
        {"tag":"escalate_complaint","patterns":["file a complaint","escalate my issue","complaint","manager"],"responses":["I'm sorry. Please provide a brief summary of the issue and I'll escalate to our complaint team with priority tagging."]},

        # General & utility
        {"tag":"business_hours","patterns":["what time are you open","business hours","opening hours"],"responses":["Our business hours are Monday-Friday, 9AM to 6PM (local time)."]},
        {"tag":"contact_info","patterns":["how to contact you","contact info","contact details","support contact"],"responses":["Email: support@example.com | Phone: +1-800-555-1234 | Live chat available on our website."]},
        {"tag":"product_info","patterns":["product details","tell me about the product","specs","features"],"responses":["Which product are you asking about? Provide model or name and I'll share details and pricing."]},
        {"tag":"shipping_costs","patterns":["shipping cost","how much to ship","delivery charges"],"responses":["Shipping cost depends on weight and destination. Provide the shipping country and product to estimate."]},

        # Fallback/default
        {"tag":"default","patterns":[],"responses":["Sorry, I didn't understand that. Could you rephrase or provide the order ID?","I didn't get that — would you like me to connect you to a human agent? Reply 'agent' to request a live agent."]}
    ]
}

# ---------------------------
# Persisted learned intents & merge function
# ---------------------------
def load_json_file(path, default=None):
    if default is None: default = {}
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print("Failed to load", path, ":", e)
    return default

def save_json_file(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print("Failed to save", path, ":", e)

learned_data = load_json_file(LEARNED_FILE, {"intents": []})
memory_store = load_json_file(MEMORY_FILE, {})
reminders_store = load_json_file(REMINDERS_FILE, {})

def merge_intents(embedded, learned):
    tag_map = {}
    for it in embedded.get("intents", []):
        tag_map[it["tag"]] = {"tag": it["tag"], "patterns": list(it.get("patterns", [])), "responses": list(it.get("responses", []))}
    for it in learned.get("intents", []):
        tag = it.get("tag")
        if not tag: continue
        if tag in tag_map:
            tag_map[tag]["patterns"].extend(it.get("patterns", []))
            tag_map[tag]["responses"].extend(it.get("responses", []))
        else:
            tag_map[tag] = {"tag": tag, "patterns": list(it.get("patterns", [])), "responses": list(it.get("responses", []))}
    return {"intents": list(tag_map.values())}

intents_data = merge_intents(EMBEDDED_INTENTS, learned_data)

# ---------------------------
# Build training dataset with light synonym augmentation
# ---------------------------
def build_dataset(intents):
    X = []
    y = []
    for it in intents.get("intents", []):
        tag = it["tag"]
        for p in it.get("patterns", []):
            proc = preprocess_for_vector(p)
            if proc.strip():
                X.append(proc)
                y.append(tag)
            # add synonyms for single words to augment
            words = [w for w in p.split() if len(w) > 1]
            for i, w in enumerate(words):
                syns = get_wordnet_synonyms(w, max_syn=2)
                for s in syns:
                    new = words[:]
                    new[i] = s
                    proc2 = preprocess_for_vector(" ".join(new))
                    X.append(proc2)
                    y.append(tag)
    return X, y

X_texts, y_tags = build_dataset(intents_data)
if len(X_texts) < 8:
    for it in intents_data.get("intents", []):
        for p in it.get("patterns", []):
            X_texts.append(preprocess_for_vector(p + " please"))
            y_tags.append(it["tag"])

# ---------------------------
# Vectorizer & classifier setup (with robust fallback)
# ---------------------------
VECTOR_MODE = "tfidf"
if np is None or TfidfVectorizer is None or LogisticRegression is None:
    VECTOR_MODE = "fallback"
    vectorizer = None
    clf = None
    X_mat = None
else:
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        X_mat = vectorizer.fit_transform(X_texts)
        clf = LogisticRegression(max_iter=600)
        clf.fit(X_mat, y_tags)
    except Exception as e:
        print("Training failed, falling back:", e)
        VECTOR_MODE = "fallback"
        vectorizer = None
        clf = None
        X_mat = None

# Save model meta
save_json_file(MODEL_META_FILE, {"vector_mode": VECTOR_MODE, "vocab_size": len(vectorizer.vocabulary_) if vectorizer else 0})

# ---------------------------
# Utilities: calculator, currency, reminders, parsing datetime
# ---------------------------
import math
SAFE_MATH = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
SAFE_MATH.update({"abs": abs, "round": round, "min": min, "max": max})

def safe_eval(expr: str):
    expr = expr.strip().replace("^", "**")
    code = compile(expr, "<string>", "eval")
    for name in code.co_names:
        if name not in SAFE_MATH:
            raise ValueError(f"Use of '{name}' not allowed")
    return eval(code, {"__builtins__": {}}, SAFE_MATH)

STATIC_RATES = {"USD":1.0,"EUR":0.92,"GBP":0.79,"INR":82.0,"JPY":155.0,"AUD":1.57,"CAD":1.34}
def convert_currency(amount: float, src: str, dst: str) -> Tuple[float,str]:
    src = src.upper(); dst = dst.upper()
    if requests:
        try:
            r = requests.get(f"https://api.exchangerate.host/convert?from={src}&to={dst}&amount={amount}", timeout=5)
            if r.status_code == 200:
                d = r.json()
                if d.get("success") and "result" in d:
                    return float(d["result"]), "(live rates)"
        except Exception:
            pass
    if src in STATIC_RATES and dst in STATIC_RATES:
        usd = amount / STATIC_RATES[src]
        converted = usd * STATIC_RATES[dst]
        return round(converted,4), "(static rates)"
    raise ValueError("Unsupported currency in offline mode")

# Reminders store thread
REM_LOCK = threading.Lock()
def save_reminders():
    save_json_file(REMINDERS_FILE, reminders_store)

def add_reminder(rid: str, text: str, at_ts: float):
    with REM_LOCK:
        reminders_store[rid] = {"text": text, "at": at_ts, "notified": False}
        save_reminders()

def reminders_check_loop(stop_event: threading.Event):
    while not stop_event.is_set():
        now = time.time()
        to_notify = []
        with REM_LOCK:
            for rid, item in list(reminders_store.items()):
                if not item.get("notified") and item.get("at",0) <= now:
                    to_notify.append((rid, item))
        for rid, item in to_notify:
            msg = f"Reminder: {item['text']}"
            print("\n*** " + msg + " ***\n")
            logging.info("REMINDER: " + item['text'])
            with REM_LOCK:
                reminders_store[rid]["notified"] = True
                save_reminders()
        time.sleep(2)

rem_stop = threading.Event()
rem_thread = threading.Thread(target=reminders_check_loop, args=(rem_stop,), daemon=True)
rem_thread.start()

# ---------------------------
# Prediction + response helpers
# ---------------------------
CONF_THRESH = 0.55
SIMILARITY_THRESH = 0.35

def find_best_similarity(vec):
    try:
        sims = cosine_similarity(vec, X_mat)
        idx = int(np.argmax(sims))
        return float(sims[0, idx]), y_tags[idx]
    except Exception:
        return 0.0, "default"

def predict_intent(user_input: str) -> Tuple[str, float]:
    proc = preprocess_for_vector(user_input)
    if VECTOR_MODE == "fallback" or vectorizer is None or clf is None:
        toks = set(proc.split())
        scores = []
        for i, txt in enumerate(X_texts):
            overlap = len(toks.intersection(set(txt.split())))
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
        conf = float(np.max(probs))
        return tag, conf
    except Exception:
        vec = vectorizer.transform([proc])
        sim, stag = find_best_similarity(vec)
        return stag, sim

def get_response_by_tag(tag: str) -> str:
    for it in intents_data.get("intents", []):
        if it["tag"] == tag:
            return random.choice(it.get("responses", ["Sorry, I don't have a response."]))
    return random.choice([it["responses"][0] for it in intents_data.get("intents", []) if it["tag"]=="default"]) if intents_data else "Sorry."

# ---------------------------
# Teach & retrain on the fly
# ---------------------------
def teach_intent(tag: str, example: str, response: Optional[str]=None):
    # persist
    ld = load_json_file(LEARNED_FILE, {"intents":[]})
    for it in ld.get("intents", []):
        if it.get("tag") == tag:
            it.setdefault("patterns", []).append(example)
            if response:
                it.setdefault("responses", []).append(response)
            break
    else:
        newit = {"tag": tag, "patterns":[example], "responses":[response] if response else []}
        ld.setdefault("intents", []).append(newit)
    save_json_file(LEARNED_FILE, ld)
    # merge and retrain
    global intents_data, X_texts, y_tags, vectorizer, clf, X_mat
    learned_local = load_json_file(LEARNED_FILE, {"intents":[]})
    intents_data = merge_intents(EMBEDDED_INTENTS, learned_local)
    X_texts, y_tags = build_dataset(intents_data)
    try:
        if VECTOR_MODE == "tfidf":
            vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
            X_mat = vectorizer.fit_transform(X_texts)
            clf = LogisticRegression(max_iter=600)
            clf.fit(X_mat, y_tags)
        # if VECTOR_MODE fallback, nothing to retrain
        print(f"Teach: Learned example for tag '{tag}'. Model retrained.")
        return True
    except Exception as e:
        print("Teach saved but retrain failed:", e)
        return False

# ---------------------------
# KB lookup (small)
# ---------------------------
KB = {
    "hours":"Mon-Fri 9am-6pm local time.",
    "shipping":"Shipping 3-7 business days; express available.",
    "refund":"Refunds processed within 7-10 business days after approval.",
    "contact":"support@example.com | +1-800-555-1234"
}

def kb_lookup(text: str) -> Optional[str]:
    t = clean_text(text)
    for k,v in KB.items():
        if k in t or any(word in t for word in k.split()):
            return v
    return None

# ---------------------------
# Context manager to hold dialog state (order id, last intent etc.)
# ---------------------------
class ChatContext:
    def __init__(self):
        self.history = deque(maxlen=200)
        self.slots = {}
    def add_user(self, text): self.history.append(("user", text))
    def add_bot(self, text): self.history.append(("bot", text))
    def set_slot(self, k, v): self.slots[k] = v
    def get_slot(self, k, default=None): return self.slots.get(k, default)
    def clear_slots(self): self.slots.clear()

# ---------------------------
# Command handling (calc, convert, teach, remind, remember)
# ---------------------------
def parse_datetime_flexible(s: str) -> datetime:
    s = s.strip().lower()
    now = datetime.now()
    if s.startswith("in "):
        m = re.match(r"in\s+(\d+)\s*(hour|hours|hr|hrs)", s)
        if m: return now + timedelta(hours=int(m.group(1)))
        m = re.match(r"in\s+(\d+)\s*(minute|minutes|min|mins)", s)
        if m: return now + timedelta(minutes=int(m.group(1)))
    if s.startswith("tomorrow"):
        rest = s[len("tomorrow"):].strip()
        dt = now + timedelta(days=1)
        if rest:
            t = parse_time_string(rest); return dt.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
        return dt
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try: return datetime.strptime(s, fmt)
        except: pass
    # parse time only
    t = parse_time_string(s); return now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)

def parse_time_string(s: str) -> datetime:
    m = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", s.strip().lower())
    if not m: raise ValueError("time parse failed")
    h = int(m.group(1)); mi = int(m.group(2) or 0); ampm = m.group(3)
    if ampm == "pm" and h < 12: h += 12
    if ampm == "am" and h == 12: h = 0
    return datetime.now().replace(hour=h, minute=mi, second=0, microsecond=0)

def handle_command(user_input: str, ctx: ChatContext) -> Optional[str]:
    ui = user_input.strip()
    lower = ui.lower()
    if lower.startswith("calc:") or lower.startswith("calculate:"):
        expr = ui.split(":",1)[1].strip()
        try:
            val = safe_eval(expr)
            return f"Result: {val}"
        except Exception as e:
            return f"Calculation error: {e}"
    if lower.startswith("convert:"):
        body = ui.split(":",1)[1].strip()
        m = re.search(r"([0-9,.]+)\s*([a-zA-Z]{3})\s*(to|in)?\s*([a-zA-Z]{3})", body)
        if m:
            amt = float(m.group(1).replace(",","")); src = m.group(2); dst = m.group(4)
            try:
                conv, note = convert_currency(amt, src, dst)
                return f"{amt} {src.upper()} = {conv} {dst.upper()} {note}"
            except Exception as e:
                return f"Conversion failed: {e}"
        else:
            return "Conversion format: convert: 100 USD to INR"
    if lower.startswith("teach:"):
        body = ui.split(":",1)[1].strip()
        parts = [p.strip() for p in body.split("|")]
        if len(parts) >= 2:
            tag, example = parts[0], parts[1]
            response = parts[2] if len(parts) >= 3 else None
            ok = teach_intent(tag, example, response)
            return f"Taught: '{example}' -> {tag}. Retrain status: {'OK' if ok else 'Failed'}"
        else:
            return "Teach format: teach: tag | example phrase | optional response"
    if lower.startswith("remind:"):
        body = ui.split(":",1)[1].strip()
        if "|" in body:
            when, message = [b.strip() for b in body.split("|",1)]
            try:
                dt = parse_datetime_flexible(when)
                ts = dt.timestamp()
                rid = f"r{int(time.time()*1000)}"
                add_reminder(rid, message, ts)
                return f"Reminder set for {dt.strftime('%Y-%m-%d %H:%M')}"
            except Exception as e:
                return f"Failed to parse date/time: {e}"
        else:
            return "Remind format: remind: <when> | <message>"
    if lower.startswith("remember:"):
        body = ui.split(":",1)[1].strip()
        if "|" in body:
            key, val = [b.strip() for b in body.split("|",1)]
            memory_store[key] = val; save_json_file(MEMORY_FILE, memory_store)
            return f"Saved memory: {key} -> {val}"
        else:
            return "Remember format: remember: key | value"
    if lower.startswith("forget:"):
        key = ui.split(":",1)[1].strip()
        if key in memory_store:
            del memory_store[key]; save_json_file(MEMORY_FILE, memory_store)
            return f"Forgot memory: {key}"
        else:
            return f"No such memory key: {key}"
    if lower == "show_intents":
        return "Known intents: " + ", ".join([it["tag"] for it in intents_data.get("intents",[])])
    if lower == "show_memory":
        return "Memory: " + json.dumps(memory_store)
    return None

# ---------------------------
# High-level process_user function (handles dialog flows & slots)
# ---------------------------
def process_user(user_input: str, ctx: ChatContext) -> str:
    cmd = handle_command(user_input, ctx)
    if cmd is not None:
        ctx.add_user(user_input); ctx.add_bot(cmd); logging.info(f"USER:{user_input} BOT:{cmd}")
        return cmd
    # KB
    kb_ans = kb_lookup(user_input)
    if kb_ans:
        ctx.add_user(user_input); ctx.add_bot(kb_ans); logging.info(f"USER:{user_input} BOT:{kb_ans}")
        return kb_ans
    # If context expecting order id, handle specialized flows
    last_intent = ctx.get_slot("last_intent")
    if last_intent == "order_status" and not ctx.get_slot("order_checked"):
        # if user provided order id
        m = re.search(r"(?:order\s?id[:#]?\s*)([A-Za-z0-9\-]+)", user_input, re.I)
        if m:
            order_id = m.group(1)
            ctx.set_slot("order_id", order_id); ctx.set_slot("order_checked", True)
            reply = f"I found order {order_id}. Current status: Shipped (example). Estimated delivery 3 days from now."
            ctx.add_user(user_input); ctx.add_bot(reply); logging.info(f"USER:{user_input} BOT:{reply}")
            return reply
    # Predict intent
    tag, conf = predict_intent(user_input)
    # try similarity fallback if low confidence
    if conf < CONF_THRESH and VECTOR_MODE != "fallback":
        try:
            proc = preprocess_for_vector(user_input); vec = vectorizer.transform([proc])
            sim_score, sim_tag = find_best_similarity(vec)
            if sim_score >= SIMILARITY_THRESH:
                tag = sim_tag; conf = sim_score
        except Exception:
            pass
    # If intent requires slots, prompt
    if tag in ("order_status", "refund_request", "cancel_order", "change_order", "damaged_item", "missing_item"):
        # ask for order id if not provided
        m = re.search(r"(?:order\s?id[:#]?\s*)([A-Za-z0-9\-]+)", user_input, re.I)
        if m:
            order_id = m.group(1); ctx.set_slot("order_id", order_id)
            reply = get_response_by_tag(tag) + f" (Received order ID {order_id} — processing...)"
            ctx.set_slot("last_intent", tag); ctx.add_user(user_input); ctx.add_bot(reply); logging.info(f"USER:{user_input} BOT:{reply}")
            return reply
        else:
            ctx.set_slot("last_intent", tag)
            reply = "Sure — please provide your order ID so I can look it up (e.g. Order ID: 12345-ABC)."
            ctx.add_user(user_input); ctx.add_bot(reply); logging.info(f"USER:{user_input} BOT:{reply}")
            return reply
    # If confident enough
    if conf >= CONF_THRESH:
        resp = get_response_by_tag(tag)
        ctx.set_slot("last_intent", tag)
        ctx.add_user(user_input); ctx.add_bot(resp)
        logging.info(f"USER:{user_input} INTENT:{tag} CONF:{conf:.2f} BOT:{resp}")
        return resp
    # low confidence -> ask clarifying or offer agent
    default_resp = get_response_by_tag("default")
    # offer escalation suggestion if multiple failed tries
    attempts = ctx.get_slot("failed_attempts") or 0
    attempts += 1; ctx.set_slot("failed_attempts", attempts)
    if attempts >= 2:
        default_resp += " If you'd like, I can connect you to a human agent — reply 'agent' to request."
    ctx.add_user(user_input); ctx.add_bot(default_resp); logging.info(f"USER:{user_input} BOT:{default_resp}")
    return default_resp

# ---------------------------
# CLI & GUI
# ---------------------------
def run_cli():
    print("Customer Helper Chatbot — type 'quit' to exit.")
    print("Commands: calc:, convert:, teach:, remind:, remember:, forget:, show_intents, show_memory")
    ctx = ChatContext()
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye.")
            rem_stop.set(); rem_thread.join(timeout=1)
            break
        if not user_input: continue
        if user_input.lower() in ("quit","exit","bye"):
            print("Bot: Goodbye."); rem_stop.set(); rem_thread.join(timeout=1); break
        # quick agent connect
        if user_input.lower() in ("agent","speak to agent","human"):
            reply = "I'll connect you to a human agent. Please share a brief summary and your preferred contact method (phone/email)."
            print("Bot:", reply); logging.info(f"USER:{user_input} BOT:{reply}"); continue
        res = process_user(user_input, ctx)
        print("Bot:", res)

def run_gui():
    try:
        import tkinter as tk
        from tkinter import scrolledtext
    except Exception:
        print("Tkinter not available — starting CLI")
        run_cli(); return
    root = tk.Tk(); root.title("Customer Helper Chatbot")
    chat_area = scrolledtext.ScrolledText(root, state="disabled", wrap="word", width=100, height=30)
    chat_area.grid(row=0, column=0, columnspan=2, padx=8, pady=8)
    entry_var = tk.StringVar()
    entry = tk.Entry(root, textvariable=entry_var, width=80)
    entry.grid(row=1, column=0, padx=8, pady=8)
    ctx = ChatContext()
    def append(sender, text):
        chat_area.configure(state="normal")
        chat_area.insert("end", f"{sender}: {text}\n")
        chat_area.configure(state="disabled")
        chat_area.yview("end")
    def on_send():
        txt = entry_var.get().strip()
        if not txt: return
        append("You", txt); entry_var.set("")
        if txt.lower() in ("quit","exit","bye"):
            append("Bot","Goodbye."); root.after(1000, root.destroy); return
        if txt.lower() in ("agent","speak to agent","human"):
            append("Bot","I'll connect you to a human agent. Please provide a brief summary and contact preference."); return
        res = process_user(txt, ctx)
        append("Bot", res)
    btn = tk.Button(root, text="Send", command=on_send); btn.grid(row=1, column=1, padx=4, pady=4)
    entry.bind("<Return>", lambda e: on_send())
    root.mainloop()

# ---------------------------
# Entry
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Run GUI (Tkinter)")
    args = parser.parse_args()
    print("Starting Customer Helper Chatbot...")
    print("NLTK full features available." if NLTK_OK else "NLTK not fully available — running in reduced mode.")
    print("Vector mode:", VECTOR_MODE)
    if args.gui:
        run_gui()
    else:
        run_cli()
