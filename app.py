"""Campus Pulse AI: a transparent, rule-based VADER feedback dashboard."""
import html
import re
from collections import Counter
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from vaderSentiment.vaderSentiment import BOOSTER_DICT, NEGATE, SentimentIntensityAnalyzer

st.set_page_config(page_title="Campus Pulse AI", page_icon="🎓", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=Space+Grotesk:wght@600;700&display=swap');
.stApp{background:radial-gradient(circle at 8% 0%,#e9eaff,transparent 28%),radial-gradient(circle at 92% 8%,#dcfaf3,transparent 25%),#f6f8fc;font-family:'DM Sans',sans-serif}
.block-container{max-width:1500px;padding-top:3rem}h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;color:#172554}
.hero{padding:28px 32px;border-radius:24px;color:white;background:linear-gradient(115deg,#111c4e,#4338ca 48%,#0f9f9a);box-shadow:0 18px 45px #312e8133;margin-bottom:18px}
.hero h1{color:white!important;margin:0}.hero p{color:#e8e8ff;margin:.4rem 0}.pill{display:inline-block;border:1px solid #ffffff55;background:#ffffff22;padding:6px 11px;border-radius:99px;font-size:.8rem}
div[data-testid="stMetric"]{background:#ffffffdf;border:1px solid #e5eaf4;border-radius:18px;padding:16px 18px;box-shadow:0 6px 18px #1e293b09}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#f0f4ff,#f8fbff 55%,#effcf9);border-right:1px solid #e1e8f5}
div[data-testid="stTabs"] button{font-weight:700;border-radius:12px 12px 0 0}.chip{display:inline-flex;flex-direction:column;align-items:center;padding:6px 9px;margin:3px;border-radius:10px;font-size:.82rem;font-weight:700}.chip small{font-size:.7rem}
</style><div class="hero"><div style="letter-spacing:2px;font-size:.75rem;color:#bffcf3">CAMPUS FEEDBACK INTELLIGENCE</div><h1>🎓 Campus Pulse AI</h1><p>Turn student feedback into clearer campus decisions.</p><span class="pill">VADER lexicon · Hand-written rules · Service insights</span></div>""", unsafe_allow_html=True)
st.caption("Academic prototype · Session data only · Human review recommended")

SERVICES = ["Canteen", "Wi-Fi", "Classrooms", "Transport", "Laboratories", "Cleanliness", "Campus", "Library", "Other"]
KEYWORDS = {
    "Canteen": ["canteen", "cafeteria", "food", "meal", "lunch", "snack", "queue"],
    "Wi-Fi": ["wi-fi", "wifi", "internet", "network", "connection", "bandwidth"],
    "Classrooms": ["classroom", "classrooms", "lecture hall", "projector", "desk", "fan", "lecture"],
    "Transport": ["bus", "transport", "driver", "route", "late", "punctual", "vehicle"],
    "Laboratories": ["lab", "laboratory", "computer", "equipment", "experiment"],
    "Cleanliness": ["clean", "dirty", "cleaning", "toilet", "washroom", "water", "hygiene"],
    "Campus": ["garden", "campus", "college", "playground", "parking", "sports"],
    "Library": ["library", "book", "reading room", "study space"],
}
ASPECTS = {
    "Canteen": {"Food quality": ["food", "meal", "taste", "tasty", "stale", "lunch", "snack", "freshness"], "Price": ["price", "cost", "expensive", "cheap", "affordable", "overpriced", "value for money"], "Hygiene": ["hygiene", "hygienic", "unhygienic", "clean", "dirty"], "Queue / speed": ["queue", "wait", "waiting", "waiting time", "long line", "service", "slow", "fast"]},
    "Wi-Fi": {"Speed": ["speed", "slow", "fast", "lag", "lagging", "buffering", "bandwidth"], "Connectivity": ["connection", "disconnect", "disconnected", "signal drops", "network", "internet", "wifi", "wi-fi"], "Coverage": ["coverage", "signal", "hostel", "range", "dead zone", "weak signal"]},
    "Transport": {"Punctuality": ["late", "delay", "delayed", "punctual", "on time"], "Crowding": ["crowd", "crowded", "overcrowded", "seat"], "Driver behaviour": ["driver", "driving", "rude", "polite"]},
    "Classrooms": {"Furniture": ["desk", "chair", "furniture", "bench"], "Projector / equipment": ["projector", "screen", "board", "equipment"], "Ventilation": ["fan", "ventilation", "air", "hot", "cramped"]},
    "Laboratories": {"Equipment": ["computer", "equipment", "machine", "experiment", "system"], "Staff": ["staff", "assistant", "technician", "helpful"]},
    "Cleanliness": {"Washrooms": ["washroom", "toilet", "restroom"], "Hygiene": ["clean", "dirty", "hygiene", "unhygienic", "water"]},
    "Campus": {"General campus": ["campus", "college"], "Grounds": ["garden", "ground", "playground"], "Facilities": ["parking", "sports", "lighting", "security"]},
    "Library": {"Books": ["book", "books", "catalogue", "collection"], "Noise": ["noise", "noisy", "quiet", "peaceful"], "Seating": ["seat", "seats", "seating", "chair", "chairs", "space"]},
    "Other": {"General": ["service", "facility", "college"]},
}
SUGGESTIONS = {
    ("Wi-Fi", "Speed"): "check bandwidth and access-point load in hostels",
    ("Wi-Fi", "Connectivity"): "inspect connection failures and access points",
    ("Canteen", "Food quality"): "review food freshness and preparation",
    ("Canteen", "Queue / speed"): "review staffing at peak meal times",
    ("Transport", "Punctuality"): "audit bus schedules and delayed routes",
    ("Classrooms", "Ventilation"): "inspect fans and ventilation",
    ("Library", "Noise"): "reinforce quiet study areas",
}
CUSTOM_LEXICON = {"lag": -1.5, "buffering": -2.0, "stale": -2.0, "overcrowded": -2.0, "unhygienic": -2.5, "hygienic": 2.0, "punctual": 2.0, "delayed": -1.5, "cramped": -1.5, "spacious": 1.8, "tasty": 1.8, "expensive": -1.6, "late": -1.5, "slow": -1.2, "polite": 1.5, "poor": -1.8, "quiet": 1.3}
# Stop words are used only for charts. VADER always receives the full raw text.
STOPWORDS = set("a an the and or but if then than to of in on at for from with is are was were be been being it its this that these those i we they you he she my our your their me us very really just too so as not no do does did have has had can could should would will please need needs more most after before today always".split())
SLANG_OR_VARIANTS = {"laggy": "lag", "laggin": "lag", "wifi": "wi-fi", "gr8": "great", "luv": "love", "awsm": "awesome"}
STUDENT_ABBREVIATIONS = {"clg": "college", "nt": "not", "nvr": "never", "vry": "very", "gud": "good", "gr8": "great", "bt": "but", "bcz": "because"}
LOW_INFORMATION = {"ok", "okay", "fine", "meh", "good", "bad", "nice", "👍", "👎"}
SARCASM_PHRASES = ["oh great", "yeah right", "thanks for nothing", "great, another", "fantastic, another", "excellent, another"]
COMPLAINT_CUES = {"down": 2.0, "failed": 2.0, "failure": 2.0, "crashed": 2.2, "broken": 2.0, "late": 1.5, "delayed": 1.5, "slow": 1.2, "cold": 1.1, "stale": 1.7, "no seats": 1.5, "not working": 1.8, "disconnecting": 1.5}
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF]|[:;]-?[)(DPp]")
CLAUSE_RE = re.compile(r"\b(?:but|however|although|though|yet|whereas|and)\b|[,;.!?]+", re.I)
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
ENGINE_NAMES = ["VADER + campus lexicon", "Campus Context Score (experimental)"]
EMOJI_VALENCE = {"😀": 2.0, "😊": 2.0, "🙂": 1.0, "👍": 1.5, "😞": -1.5, "😡": -2.5, "🙄": -1.0, "👎": -1.5, ":)": 1.5, ":(": -1.5}


class CampusContextAnalyzer:
    """Independent weighted lexicon scorer; no VADER polarity_scores call.

    It borrows the visible VADER/campus word dictionary, then applies its own
    negation window, boosters, contrast weights, caps, and punctuation.
    The score formula is a project choice.
    """

    def __init__(self, lexicon):
        self.lexicon = dict(lexicon)

    def explain(self, text):
        tokens = re.findall(r"[A-Za-z][A-Za-z'-]*|[\U0001F300-\U0001FAFF]|[:;]-?[)(DPp]", str(text))
        lowered = [token.lower() for token in tokens]
        contrast = next((i for i, word in enumerate(lowered) if word in {"but", "however"}), None)
        mixed_caps = any(word.isupper() and len(word) > 1 for word in tokens) and any(word.islower() for word in tokens)
        results = []
        for i, token in enumerate(tokens):
            mapped = SLANG_OR_VARIANTS.get(lowered[i], lowered[i])
            base = self.lexicon.get(mapped, EMOJI_VALENCE.get(token, 0.0))
            value, rules = float(base), []
            if value:
                context = lowered[max(0, i-3):i]
                if any(word in NEGATE or word.endswith("n't") for word in context):
                    value *= -0.74
                    rules.append("negation window")
                for word in context[-2:]:
                    if word in BOOSTER_DICT:
                        value += BOOSTER_DICT[word] * (1 if value > 0 else -1)
                        rules.append(f"booster: {word}")
                if mixed_caps and token.isupper() and len(token) > 1:
                    value *= 1.25
                    rules.append("ALL CAPS")
                if contrast is not None:
                    value *= 0.5 if i < contrast else 1.5 if i > contrast else 1
                    rules.append("contrast weight")
            results.append({"token":token,"base":round(float(base),3),"adjusted":round(value,3),"rules":", ".join(rules)})
        return results

    def polarity_scores(self, text):
        details = self.explain(text)
        values = [item["adjusted"] for item in details]
        exclamations = min(str(text).count("!"), 4)
        total = sum(values) * (1 + .08 * exclamations)
        compound = total / (abs(total) + 4) if total else 0.0
        positive = sum(max(value, 0) for value in values)
        negative = sum(max(-value, 0) for value in values)
        neutral = sum(value == 0 for value in values)
        denominator = positive + negative + neutral
        if not denominator:
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
        return {"neg": round(negative/denominator, 3), "neu": round(neutral/denominator, 3), "pos": round(positive/denominator, 3), "compound": round(compound, 4)}


def contains_term(text, term):
    """Match complete words or phrases; fan must not match fantastic."""
    return bool(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", str(text), re.I))


def normalize_comment(text):
    """Expand a small, explicit list of student abbreviations for scoring."""
    value = str(text)
    for short, full in STUDENT_ABBREVIATIONS.items():
        value = re.sub(r"(?<!\w)" + re.escape(short) + r"(?!\w)", full, value, flags=re.I)
    return value


def sarcasm_evidence(text):
    """Require both a sarcasm phrase and an explicit complaint before adjusting."""
    low = normalize_comment(text).lower()
    cue = next((phrase for phrase in SARCASM_PHRASES if contains_term(low, phrase)), None)
    if not cue and contains_term(low, "as usual") and any(contains_term(low, word) for word in ["great", "fantastic", "excellent"]):
        cue = "praise + as usual"
    complaints = [(term, weight) for term, weight in COMPLAINT_CUES.items() if contains_term(low, term)]
    if not cue or not complaints:
        return None
    term, weight = max(complaints, key=lambda pair: pair[1])
    return {"cue": cue, "complaint": term, "target": -min(.85, .25 + .15 * weight)}


class SarcasmAwareAnalyzer:
    """Apply a conservative dictionary rule on top of either base scorer."""

    def __init__(self, engine):
        self.engine = engine
        self.lexicon = engine.lexicon

    def polarity_scores(self, text):
        scoring_text = normalize_comment(text)
        raw = self.engine.polarity_scores(scoring_text)
        evidence = sarcasm_evidence(text)
        if evidence is None:
            return raw
        compound = round(min(raw["compound"], evidence["target"]), 4)
        if compound == raw["compound"]:
            return raw
        negative = max(raw["neg"], abs(compound))
        positive = min(raw["pos"], max(0, 1-negative) * .25)
        return {"neg": round(negative, 3), "neu": round(max(0, 1-negative-positive), 3), "pos": round(positive, 3), "compound": compound}


def detect_service(text):
    text = normalize_comment(text)
    hits = [(service, sum(contains_term(text, word) for word in words)) for service, words in KEYWORDS.items()]
    best = max(hits, key=lambda item: item[1])
    return best[0] if best[1] else "Other"


def mentioned_services(text):
    text = normalize_comment(text)
    return [service for service, words in KEYWORDS.items() if any(contains_term(text, word) for word in words)] or ["Other"]


def clean_tokens(text):
    return [w for w in re.findall(r"[a-z][a-z'-]{2,}", str(text).lower()) if w not in STOPWORDS]


def label3(compound):
    return "Positive" if compound >= .05 else "Negative" if compound <= -.05 else "Neutral"


def label5(compound, strong_negative, strong_positive):
    if compound <= strong_negative: return "Very Negative"
    if compound <= -.05: return "Negative"
    if compound >= strong_positive: return "Very Positive"
    if compound >= .05: return "Positive"
    return "Neutral"


def split_clauses(text):
    return [part.strip() for part in CLAUSE_RE.split(str(text)) if part.strip()]


def mixed_clause_evidence(text, analyzer):
    clauses = split_clauses(normalize_comment(text))
    scored = [(clause, analyzer.polarity_scores(clause)["compound"]) for clause in clauses]
    signs = {label3(score) for _, score in scored}
    return scored if "Positive" in signs and "Negative" in signs else []


def aspect_rows(text, service, analyzer):
    found = []
    sarcasm = sarcasm_evidence(text)
    adjusted_whole_score = analyzer.polarity_scores(text)["compound"] if sarcasm else None
    active_service = service
    for clause in split_clauses(normalize_comment(text)):
        clause_service = detect_service(clause)
        if clause_service != "Other":
            active_service = clause_service
        clause_service = active_service
        for aspect, terms in ASPECTS.get(clause_service, {}).items():
            if any(contains_term(clause, term) for term in terms):
                score = analyzer.polarity_scores(clause)["compound"]
                if sarcasm and contains_term(clause, sarcasm["complaint"]):
                    score = min(score, adjusted_whole_score)
                found.append({"service": clause_service, "aspect": aspect, "clause": clause, "score": score, "sentiment": label3(score)})
    return found


def aspect_review_reasons(found):
    """Explain when a clause-based aspect assignment deserves review."""
    reasons = []
    by_clause = {}
    for item in found:
        by_clause.setdefault(item["clause"], set()).add(item["aspect"])
    if any(len(aspects) > 1 for aspects in by_clause.values()):
        reasons.append("Multiple aspects in one clause")
    if len(found) > 1 and len({item["sentiment"] for item in found}) > 1:
        reasons.append("Opposing aspect sentiment")
    return reasons


def confidence_flags(text, analyzer):
    low = normalize_comment(text).lower()
    clauses = split_clauses(normalize_comment(text))
    signs = {label3(analyzer.polarity_scores(c)["compound"]) for c in clauses}
    flags = []
    if sarcasm_evidence(text):
        flags.append("Possible sarcasm: review")
    if "Positive" in signs and "Negative" in signs:
        flags.append("Mixed polarity")
    if re.search(r"\b(?:not bad|not terrible|not great|not sure|hardly|never|wasn't|isn't|wouldn't)\b", low):
        flags.append("Ambiguous negation")
    if re.search(r"[^\W\d_][^\x00-\x7F]", text) or re.search(r"[\u0900-\u097F\u0600-\u06FF\u4E00-\u9FFF]", text):
        flags.append("Non-English text: review language")
    words = re.findall(r"[a-z]+", low)
    if len(words) <= 2 or low.strip(".!? ") in LOW_INFORMATION:
        flags.append("Very short / limited context")
    if any(contains_term(low, variant) for variant in SLANG_OR_VARIANTS if variant != "wifi"):
        flags.append("Slang or spelling variant")
    return flags


def evaluation_examples():
    """60 hand-labelled teaching examples, six per service plus six challenge cases."""
    labels = {
        "Canteen": ["Positive", "Mixed", "Negative", "Mixed", "Negative", "Positive"],
        "Wi-Fi": ["Positive", "Negative", "Negative", "Negative", "Mixed", "Positive"],
        "Classrooms": ["Positive", "Mixed", "Negative", "Negative", "Positive", "Negative"],
        "Transport": ["Positive", "Negative", "Mixed", "Negative", "Positive", "Negative"],
        "Laboratories": ["Positive", "Negative", "Mixed", "Positive", "Negative", "Positive"],
        "Cleanliness": ["Positive", "Negative", "Mixed", "Negative", "Positive", "Negative"],
        "Campus": ["Positive", "Negative", "Positive", "Mixed", "Negative", "Positive"],
        "Library": ["Positive", "Mixed", "Negative", "Mixed", "Positive", "Negative"],
        "Other": ["Positive", "Negative", "Positive", "Mixed", "Positive", "Negative"],
    }
    challenges = [
        ("Oh great, the Wi-Fi is down again. Yeah right.", "Negative", "Sarcasm cue"),
        ("I wouldn't say the food was bad.", "Positive", "Negation"),
        ("the clg is nt great but food is very good", "Mixed", "Student shorthand + mixed services"),
        ("Fantastic, another bus delayed by an hour!!!", "Negative", "Sarcasm cue"),
        ("The library has books and chairs.", "Neutral", "Neutral wording"),
        ("Not bad at all, the classroom feels spacious.", "Positive", "Negation"),
    ]
    examples = demo_rows()[:0]
    # Use each distinct demo sentence once, so repetition does not inflate accuracy.
    demo = demo_rows()
    for service in SERVICES:
        unique = list(dict.fromkeys(r["feedback"] for r in demo if r["service"] == service))[:6]
        examples.extend({"feedback": text, "human_label": label, "category": service} for text, label in zip(unique, labels[service]))
    examples.extend({"feedback": text, "human_label": label, "category": category} for text, label, category in challenges)
    return examples


def evaluation_table(examples, analyzer):
    rows = []
    for item in examples:
        predicted = label3(analyzer.polarity_scores(item["feedback"])["compound"])
        if isinstance(analyzer, SarcasmAwareAnalyzer) and mixed_clause_evidence(item["feedback"], analyzer):
            predicted = "Mixed"
        rows.append({**item, "predicted_label": predicted, "correct": predicted == item["human_label"]})
    return pd.DataFrame(rows)


def demo_rows():
    """228 deterministic comments across 30 days, including a Wi-Fi incident."""
    examples = {
        "Canteen": ["The canteen food is tasty and affordable!", "Food is tasty but too expensive.", "The lunch was stale and the queue was LONG!!!", "Canteen staff are polite, but service is slow.", "Oh great, another cold meal as usual 🙄", "The snacks are not bad and the counters look hygienic."],
        "Wi-Fi": ["Wi-Fi is fast in the library today 😀", "The hostel signal keeps dropping and buffering!!!", "The internet is not reliable during lectures.", "Oh great, the network crashed again, yeah right.", "The connection is good but coverage near the gate is poor.", "Wi-Fi speed improved after the router reset."],
        "Classrooms": ["The classroom is spacious and comfortable.", "The projector is broken but the lecturer is helpful.", "The fan is NOT working!!!", "Desks are cramped and the room is too hot.", "The new screen looks excellent.", "Classroom chairs are not bad, though ventilation is weak."],
        "Transport": ["The bus was punctual and the driver was courteous.", "The bus was delayed AGAIN!!!", "Transport is convenient but the vehicle is overcrowded.", "The driver was rude and the route was confusing.", "The bus arrived on time today 🙂", "Oh great, the bus is late as usual."],
        "Laboratories": ["The lab computers work well and the staff are helpful.", "Lab equipment is outdated and several computers crashed.", "The experiment was exciting but the systems were slow.", "The laboratory assistant explained everything clearly.", "The lab is too cramped for our group.", "Computer access is not bad this week."],
        "Cleanliness": ["The washroom is clean and hygienic.", "The toilets are DIRTY!!!", "Drinking water is fine but the washroom needs cleaning.", "The corridor smells bad and bins are overflowing.", "Cleaning staff did an excellent job today 🙂", "The water station is not very clean."],
        "Campus": ["The campus garden looks beautiful this morning.", "Parking is crowded and the lighting is poor.", "Sports grounds are clean and spacious.", "The playground is nice but security near the gate is weak.", "Campus paths were flooded after rain.", "The new benches are GREAT!!!"],
        "Library": ["The library is peaceful and has useful books.", "The reading room is noisy but the book collection is excellent.", "Library seating is cramped during exams.", "The books are not bad, but the catalogue is outdated.", "The quiet study space is wonderful 😊", "Oh great, no seats left in the library again."],
        "Other": ["The admission office was helpful today.", "The fee portal is slow and confusing.", "Campus support answered my question promptly.", "The notice board is outdated but easy to find.", "The college event was fun 😊", "The help desk was not helpful!!!"],
    }
    now = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
    rows = []
    for service_index, service in enumerate(SERVICES):
        for i in range(24):
            text = examples[service][i % 6]
            day = (i * 7 + service_index * 3) % 30
            stamp = now - timedelta(days=day, hours=i % 8)
            rating = [5, 2, 1, 3, 4, 3][i % 6]
            rows.append({"feedback": text, "service": service, "timestamp": stamp, "rating": rating})
    for i in range(12):
        stamp = now - timedelta(days=6 + i % 2, hours=i)
        rows.append({"feedback": ["Wi-Fi is DOWN in the hostel!!! 😡", "The network is buffering and painfully slow.", "Oh great, another internet failure as usual."][i % 3], "service": "Wi-Fi", "timestamp": stamp, "rating": 1})
    return rows


def analyze(rows, analyzer, baseline, strong_negative, strong_positive):
    records, aspects = [], []
    for index, source in enumerate(rows):
        row = dict(source)
        row["feedback"] = str(row.get("feedback", ""))
        row["scoring_text"] = normalize_comment(row["feedback"])
        row["service"] = row.get("service") if row.get("service") in SERVICES else "Other"
        row["mentioned_services"] = ", ".join(mentioned_services(row["feedback"]))
        row["timestamp"] = pd.to_datetime(row.get("timestamp"), errors="coerce")
        score = analyzer.polarity_scores(row["feedback"])
        raw_score = analyzer.engine.polarity_scores(row["scoring_text"])
        sarcasm = sarcasm_evidence(row["feedback"])
        mixed_clauses = mixed_clause_evidence(row["feedback"], analyzer)
        old = baseline.polarity_scores(row["feedback"])
        row.update({"score": score["compound"], "score_label": label3(score["compound"]), "sentiment": "Mixed" if mixed_clauses else label3(score["compound"]), "fine_label": "Mixed" if mixed_clauses else label5(score["compound"], strong_negative, strong_positive), "derived_stars": None if mixed_clauses else round(((score["compound"] + 1) / 2) * 4 + 1), "baseline_score": old["compound"], "baseline_label": label3(old["compound"])})
        row["mixed_clauses"] = "; ".join(f"{clause}: {label3(value)} ({value:+.3f})" for clause, value in mixed_clauses)
        row["pre_sarcasm_score"] = raw_score["compound"]
        row["sarcasm_cue"] = sarcasm["cue"] if sarcasm else ""
        row["complaint_cue"] = sarcasm["complaint"] if sarcasm else ""
        row["sarcasm_adjusted"] = bool(sarcasm and score["compound"] != raw_score["compound"])
        rating = pd.to_numeric(row.get("rating"), errors="coerce")
        row["rating"] = int(rating) if pd.notna(rating) and 1 <= rating <= 5 and rating % 1 == 0 else None
        row["rating_mismatch"] = row["rating"] is not None and row["derived_stars"] is not None and abs(row["rating"] - row["derived_stars"]) >= 2
        flags = confidence_flags(row["feedback"], analyzer)
        found = aspect_rows(row["feedback"], row["service"], analyzer)
        flags.extend(aspect_review_reasons(found))
        row["confidence"] = "Low confidence" if flags else "Standard"
        row["confidence_flags"] = ", ".join(flags)
        row["aspects"] = "; ".join(f"{item['service']} / {item['aspect']}: {item['sentiment']}" for item in found)
        review = st.session_state.get("reviews", {}).get(index, {})
        row["reviewed_service"] = review.get("service")
        row["reviewed_sentiment"] = review.get("sentiment")
        row["reviewed_aspect"] = review.get("aspect")
        row["review_note"] = review.get("note", "")
        row["review_status"] = "Reviewed" if review else "Pending"
        row["priority"] = "High" if row["sentiment"] == "Negative" and row["score"] <= -.5 else "Medium" if row["sentiment"] == "Negative" else "Low"
        records.append(row)
        for item in found:
            aspects.append({"row_id": index, "service": row["service"], "feedback": row["feedback"], **item})
    return pd.DataFrame(records), pd.DataFrame(aspects)


if "rows" not in st.session_state:
    st.session_state.rows = demo_rows()
if "reviews" not in st.session_state:
    st.session_state.reviews = {}
baseline = SentimentIntensityAnalyzer()
vader_custom = SentimentIntensityAnalyzer()
with st.sidebar:
    st.header("🎛️ Control center")
    with st.form("add_feedback", clear_on_submit=True):
        comment = st.text_area("Student comment", placeholder="The Wi-Fi is slow near the library...")
        service_choice = st.selectbox("Campus service", ["Auto-detect"] + SERVICES)
        student_rating = st.selectbox("Student rating (optional)", ["Not provided", 1, 2, 3, 4, 5])
        submitted = st.form_submit_button("Analyze & add feedback", width="stretch")
    if submitted:
        if comment.strip():
            st.session_state.rows.append({"feedback": comment.strip(), "service": detect_service(comment) if service_choice == "Auto-detect" else service_choice, "timestamp": datetime.now(), "rating": None if student_rating == "Not provided" else student_rating})
            st.rerun()
        st.warning("Enter a comment first.")
    st.divider()
    upload = st.file_uploader("Import feedback CSV", type="csv", help="Required: feedback. Optional: service, timestamp, rating (1–5).")
    if upload is not None and st.button("Import uploaded CSV", width="stretch"):
        try:
            incoming = pd.read_csv(upload)
            incoming.columns = [str(c).strip().lower() for c in incoming.columns]
            if "feedback" not in incoming: st.error("CSV needs a feedback column.")
            else:
                added = 0
                for _, item in incoming.iterrows():
                    value = item.get("feedback")
                    if pd.isna(value) or not str(value).strip(): continue
                    text = str(value).strip()
                    service = item.get("service")
                    stamp = pd.to_datetime(item.get("timestamp"), errors="coerce")
                    if pd.isna(stamp): stamp = datetime.now()
                    st.session_state.rows.append({"feedback": text, "service": service if service in SERVICES else detect_service(text), "timestamp": stamp, "rating": item.get("rating")})
                    added += 1
                st.success(f"Imported {added} comments.")
                st.rerun()
        except Exception as exc: st.error(f"Could not import CSV: {exc}")
    if st.button("↩️ Restore demo dataset", width="stretch"):
        st.session_state.rows = demo_rows()
        st.session_state.reviews = {}
        st.rerun()
    st.divider()
    st.subheader("Fine-grained thresholds")
    strong_negative = st.slider("Very Negative at or below", -.95, -.10, -.60, .05)
    strong_positive = st.slider("Very Positive at or above", .10, .95, .60, .05)
    st.caption("Three-way labels use the same ±0.05 cutoffs for either scoring method.")
    st.subheader("Campus lexicon")
    custom_text = st.text_area("Add word: score pairs", placeholder="printer: -1.5\nhelpdesk: 1.0", help="One word and score from -4 to +4 per line.")
    additions = {}
    errors = []
    for line in custom_text.splitlines():
        if not line.strip(): continue
        match = re.fullmatch(r"\s*([A-Za-z][A-Za-z-]*)\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*", line)
        if not match or not -4 <= float(match.group(2)) <= 4: errors.append(line)
        else: additions[match.group(1).lower()] = float(match.group(2))
    if errors: st.warning(f"Ignored invalid pairs: {', '.join(errors[:3])}")
    vader_custom.lexicon.update(CUSTOM_LEXICON | additions)
    st.caption("Changes apply immediately to all analyses. Entries stay in this session.")
    st.subheader("Scoring algorithm")
    engine_name = st.selectbox("Choose analysis method", ENGINE_NAMES)
    st.caption("Campus Context Score is an experimental second rule algorithm. VADER remains the default.")

context_analyzer = CampusContextAnalyzer(vader_custom.lexicon)
vader_aware = SarcasmAwareAnalyzer(vader_custom)
context_aware = SarcasmAwareAnalyzer(context_analyzer)
analyzer = vader_aware if engine_name == ENGINE_NAMES[0] else context_aware
st.caption(f"Active scoring method: {engine_name}")

base, aspect_base = analyze(st.session_state.rows, analyzer, baseline, strong_negative, strong_positive)
base["derived_stars"] = base["derived_stars"].astype("Int64")
base["algorithm"] = engine_name
f1, f2, f3 = st.columns([1.1, 1.1, 1.5])
with f1: services = st.multiselect("Campus services", SERVICES, default=SERVICES)
with f2: sentiments = st.multiselect("Sentiment labels", ["Positive", "Neutral", "Negative", "Mixed"], default=["Positive", "Neutral", "Negative", "Mixed"])
with f3: search = st.text_input("🔎 Search comments", placeholder="Search for Wi-Fi, late bus, clean...")
filtered = base[base.service.isin(services) & base.sentiment.isin(sentiments)].copy()
if search.strip(): filtered = filtered[filtered.feedback.str.contains(search.strip(), case=False, regex=False, na=False)]
aspect_filtered = aspect_base[aspect_base.row_id.isin(filtered.index)] if not aspect_base.empty else aspect_base
total = len(filtered)
pos = int((filtered.sentiment == "Positive").sum())
neg = int((filtered.sentiment == "Negative").sum())
neu = int((filtered.sentiment == "Neutral").sum())
mixed = int((filtered.sentiment == "Mixed").sum())
mood = round(100 * (pos - neg) / total) if total else 0
for col, name, value in zip(st.columns(6), ["💬 Feedback analyzed", "😊 Positive", "😐 Neutral", "🚨 Negative", "🔀 Mixed", "🌡️ Mood score"], [total, pos, neu, neg, mixed, f"{mood:+d}/100"]):
    col.metric(name, value)

tabs = st.tabs(["✨ Overview", "🏫 Service insights", "🗂️ Feedback explorer", "🔬 Pipeline Inspector", "🧩 Aspects & actions", "📊 Comparison", "🔤 Terms & associations", "📈 Trends & warnings", "🧪 Limitations", "✅ Evaluation & review", "📚 Method & project notes"])

with tabs[0]:
    a, b = st.columns([.8, 1.2])
    with a:
        st.subheader("Campus mood snapshot")
        st.metric("Mood", f"{mood:+d}/100")
        st.caption("(Positive − negative) ÷ all comments × 100. A project indicator, not a validated wellbeing measure.")
        st.info("Review recurring negative topics in the service and aspect tabs." if neg else "No negative feedback in the current selection.")
    with b:
        st.subheader("Sentiment distribution")
        if total:
            counts = filtered.sentiment.value_counts().reindex(["Positive", "Neutral", "Negative", "Mixed"], fill_value=0)
            fig = px.pie(names=counts.index, values=counts.values, hole=.67, color=counts.index, color_discrete_map={"Positive":"#10b981","Neutral":"#94a3b8","Negative":"#f43f5e","Mixed":"#8b5cf6"})
            fig.update_layout(showlegend=True, margin=dict(t=10,b=10,l=10,r=10))
            st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
    st.subheader("Feedback volume over time")
    if total and filtered.timestamp.notna().any():
        daily = filtered.assign(Date=filtered.timestamp.dt.date).groupby(["Date", "sentiment"]).size().reset_index(name="Comments")
        st.plotly_chart(px.area(daily, x="Date", y="Comments", color="sentiment", color_discrete_map={"Positive":"#10b981","Neutral":"#94a3b8","Negative":"#f43f5e","Mixed":"#8b5cf6"}), width="stretch", config=PLOT_CONFIG)
    st.subheader("Most repeated words")
    common = Counter(w for text in filtered.feedback for w in clean_tokens(text)).most_common(12)
    if common:
        chart = pd.DataFrame(common, columns=["Word", "Frequency"])
        st.plotly_chart(px.bar(chart.sort_values("Frequency"), x="Frequency", y="Word", orientation="h", color="Frequency", color_continuous_scale=["#c7d2fe","#6366f1","#0d9488"]), width="stretch", config=PLOT_CONFIG)
    else: st.info("No matching comments.")

with tabs[1]:
    st.subheader("Service health comparison")
    if total:
        stats = filtered.groupby("service").agg(Total=("feedback","size"), Negative=("sentiment",lambda s:(s=="Negative").sum()), Positive=("sentiment",lambda s:(s=="Positive").sum()), Mixed=("sentiment",lambda s:(s=="Mixed").sum()), Average_score=("score","mean"), Average_stars=("derived_stars","mean")).reset_index()
        stats["Negative %"] = 100 * stats.Negative / stats.Total
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(stats.sort_values("Negative %"), x="Negative %", y="service", orientation="h", color="Negative %", color_continuous_scale=["#a7f3d0","#fbbf24","#fb7185"]), width="stretch", config=PLOT_CONFIG)
        c2.plotly_chart(px.bar(stats.sort_values("Total"), x="Total", y="service", orientation="h", color="Average_score", color_continuous_scale=["#fb7185","#cbd5e1","#34d399"]), width="stretch", config=PLOT_CONFIG)
        st.subheader("🚦 Follow-up queue")
        alerts = stats[(stats.Negative >= 2) | ((stats.Negative >= 1) & (stats["Negative %"] >= 50))]
        for _, row in alerts.sort_values("Negative %", ascending=False).iterrows():
            st.warning(f"{row.service}: {int(row.Negative)} negative of {int(row.Total)} ({row['Negative %']:.1f}%).")
        st.caption("Rule: ≥2 negative comments, or ≥1 negative and ≥50% negative share.")
        st.dataframe(stats.rename(columns={"service":"Service"}).round(2), width="stretch", hide_index=True)
    else: st.info("Select comments to compare services.")

with tabs[2]:
    st.subheader("Searchable feedback table")
    st.caption("Final sentiment uses clause context. A Mixed comment can retain a positive or negative score-only label; its single derived star rating is left blank.")
    low_only = st.checkbox("Show only low-confidence comments")
    shown = filtered[filtered.confidence.eq("Low confidence")] if low_only else filtered
    columns = ["timestamp","algorithm","service","mentioned_services","feedback","scoring_text","sentiment","fine_label","score","score_label","mixed_clauses","pre_sarcasm_score","sarcasm_cue","complaint_cue","sarcasm_adjusted","derived_stars","rating","rating_mismatch","aspects","confidence","confidence_flags","review_status","reviewed_service","reviewed_sentiment","reviewed_aspect","priority"]
    if len(shown):
        display = shown[columns].copy()
        display["timestamp"] = display.timestamp.dt.strftime("%Y-%m-%d %H:%M").fillna("")
        display["score"] = display.score.round(3)
        st.dataframe(display.sort_values("timestamp", ascending=False), width="stretch", hide_index=True, column_config={
            "sentiment": st.column_config.TextColumn("Final sentiment"),
            "score": st.column_config.NumberColumn("Lexicon compound", format="%.3f"),
            "score_label": st.column_config.TextColumn("Score-only label"),
            "mixed_clauses": st.column_config.TextColumn("Opposing clauses"),
            "scoring_text": st.column_config.TextColumn("Expanded scoring text"),
        })
        st.download_button("⬇️ Export filtered feedback (CSV)", display.to_csv(index=False).encode("utf-8-sig"), "campus_pulse_feedback.csv", "text/csv")
        report = f"Campus Pulse AI - Summary\nGenerated: {datetime.now():%Y-%m-%d %H:%M}\nAlgorithm: {engine_name}\nFeedback analyzed: {total}\nPositive: {pos}\nNeutral: {neu}\nNegative: {neg}\nMixed: {mixed}\nMood score: {mood}/100\nRule-based lexicon analysis."
        st.download_button("📄 Export short summary (TXT)", report, "campus_pulse_summary.txt", "text/plain")
    else: st.info("No rows match the explorer filters.")
    export = base.drop(columns=[], errors="ignore").copy()
    if not export.empty: export["timestamp"] = export.timestamp.dt.strftime("%Y-%m-%d %H:%M")
    st.download_button("⬇️ Export fully analyzed dataset (CSV)", export.to_csv(index=False).encode("utf-8-sig"), "campus_pulse_full_analysis.csv", "text/csv")

with tabs[3]:
    st.subheader("Pipeline Inspector")
    st.markdown("#### Rule effect comparison")
    contrast_groups = {
        "Emphasis and punctuation": ["good", "VERY good!!!", "good!!!"],
        "Negation": ["good", "not good", "not bad"],
        "Contrast": ["The food is good.", "The food is good but the price is terrible.", "The price is terrible but the food is good."],
        "Sarcasm cue": ["The Wi-Fi is great.", "Oh great, the Wi-Fi is down again.", "The Wi-Fi is down again."],
    }
    comparison_name = st.selectbox("Compare a rule", list(contrast_groups))
    comparison = []
    for sentence in contrast_groups[comparison_name]:
        result = analyzer.polarity_scores(sentence)
        cues = []
        if "not" in sentence.lower(): cues.append("negation")
        if "VERY" in sentence: cues.append("ALL-CAPS and booster")
        if "!" in sentence: cues.append("punctuation")
        if contains_term(sentence, "but"): cues.append("but contrast")
        if sarcasm_evidence(sentence): cues.append("sarcasm phrase + complaint")
        comparison.append({"Text": sentence, "Compound": result["compound"], "Label": label3(result["compound"]), "Rule cues": ", ".join(cues) or "none"})
    st.dataframe(pd.DataFrame(comparison), width="stretch", hide_index=True)
    st.caption(f"Scores come from {engine_name}. Differences reflect combined rules, not isolated numerical contributions.")
    st.divider()
    choices = ["Type my own"] + [f"{i+1}. {text[:75]}" for i, text in enumerate(base.feedback.head(30))]
    pick = st.selectbox("Choose a sample comment", choices)
    raw = st.text_area("Raw text", value="" if pick == "Type my own" else base.iloc[choices.index(pick)-1].feedback, key=f"inspector_{pick}")
    if raw.strip():
        tokens = re.findall(r"[A-Za-z][A-Za-z'-]*|[\U0001F300-\U0001FAFF]|[:;]-?[)(DPp]", raw)
        st.markdown("#### 1. Raw text")
        st.write(raw)
        st.markdown("#### 2. Preprocessing for term display")
        st.write({"normalized for scoring": normalize_comment(raw), "lowercase": raw.lower(), "tokens": tokens, "without chart stop words": clean_tokens(raw)})
        st.caption("The original is preserved. Only listed abbreviations are expanded for scoring; stop-word removal is for term charts only.")
        st.markdown("#### 3. Lexicon matching")
        chips = []
        for token in tokens:
            value = analyzer.lexicon.get(token.lower(), 0)
            bg, fg = ("#dcfce7", "#166534") if value > 0 else (("#ffe4e6", "#9f1239") if value < 0 else ("#e2e8f0", "#475569"))
            chips.append(f'<span class="chip" style="background:{bg};color:{fg}">{html.escape(token)}<small>{value:+.1f}</small></span>')
        st.markdown("".join(chips), unsafe_allow_html=True)
        st.caption("Chip values are raw dictionary valences; final scores also apply context rules.")
        st.markdown("#### 4. Rule application")
        words_lower = [w.lower() for w in tokens]
        fired = []
        if any(w in NEGATE or w.endswith("n't") for w in words_lower): fired.append("Negation")
        if any(w in BOOSTER_DICT for w in words_lower): fired.append("Intensifier / booster")
        if any(w.isupper() and len(w) > 1 and w.lower() in analyzer.lexicon for w in tokens) and any(w.islower() for w in tokens): fired.append("ALL-CAPS emphasis")
        if "!" in raw: fired.append("! punctuation")
        if EMOJI_RE.search(raw): fired.append("Emoji / emoticon")
        if contains_term(raw, "but"): fired.append("But contrast")
        evidence = sarcasm_evidence(raw)
        if evidence: fired.append("Possible sarcasm")
        st.write(", ".join(fired) if fired else "No listed rule cue detected")
        st.caption("These are rule cues. The experimental scorer also provides token-level adjustments below.")
        if isinstance(analyzer.engine, CampusContextAnalyzer):
            st.dataframe(pd.DataFrame(analyzer.engine.explain(raw)), width="stretch", hide_index=True)
        st.markdown("#### 5. Score aggregation")
        score = analyzer.polarity_scores(raw)
        mixed_evidence = mixed_clause_evidence(raw, analyzer)
        if evidence:
            before = analyzer.engine.polarity_scores(normalize_comment(raw))["compound"]
            st.warning(f"Possible sarcasm: '{evidence['cue']}' plus complaint '{evidence['complaint']}'. Score before rule: {before:+.4f}; after rule: {score['compound']:+.4f}. Human review advised.")
        if mixed_evidence:
            st.warning("Mixed opinions: " + "; ".join(f"{clause} → {label3(value)} ({value:+.3f})" for clause, value in mixed_evidence))
        st.json({k: round(v, 4) for k, v in score.items()})
        st.markdown("#### 6. Final label")
        if mixed_evidence:
            st.info(f"Mixed · numeric score alone would say {label3(score['compound'])} ({score['compound']:+.3f}). No single derived star rating for opposing clauses.")
        else:
            st.success(f"{label3(score['compound'])} · {label5(score['compound'], strong_negative, strong_positive)} · {round(((score['compound']+1)/2)*4+1)} derived stars")

with tabs[4]:
    st.subheader("Rule-based aspect sentiment")
    st.caption("Split at but, however, although, commas, and and; score each clause containing an aspect keyword.")
    st.caption("Added phrases include waiting time, value for money, signal drops, weak signal, and more. Clauses with multiple aspects are flagged for review.")
    if not aspect_filtered.empty:
        heat = aspect_filtered.pivot_table(index="aspect", columns="service", values="score", aggfunc="mean")
        fig = px.imshow(heat, zmin=-1, zmax=1, color_continuous_scale=["#f43f5e","#f8fafc","#10b981"], aspect="auto", labels={"color":"Avg compound"})
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
        st.dataframe(aspect_filtered[["service","aspect","clause","sentiment","score","feedback"]].round(3), width="stretch", hide_index=True)
        st.subheader("Priority Action List")
        priorities = aspect_filtered.groupby(["service","aspect"]).agg(volume=("score","size"), negative=("sentiment",lambda s:(s=="Negative").sum())).reset_index()
        priorities["negative_share"] = priorities.negative / priorities.volume
        priorities["priority_score"] = priorities.negative_share * priorities.volume
        priorities = priorities[priorities.negative > 0].sort_values("priority_score", ascending=False)
        for _, row in priorities.head(10).iterrows():
            suggestion = SUGGESTIONS.get((row.service, row.aspect), f"review student comments about {row.aspect.lower()} and inspect the service")
            st.warning(f"{row.service} / {row.aspect}: {row.negative_share:.0%} negative ({int(row.negative)}/{int(row.volume)}) → {suggestion}.")
        st.caption("Priority = negative share × mention volume = number of negative aspect mentions. Repeated comments count as separate mentions.")
    else: st.info("No aspect keywords found in this selection.")

with tabs[5]:
    st.subheader("Campus services as brands")
    if total:
        lead = filtered.groupby("service").agg(Comments=("feedback","size"), Positive=("sentiment",lambda s:(s=="Positive").sum()), Negative=("sentiment",lambda s:(s=="Negative").sum()), Avg_compound=("score","mean"), Avg_stars=("derived_stars","mean")).reset_index().rename(columns={"service":"Service"})
        lead["Share of voice %"] = 100 * lead.Comments / total
        lead["Net sentiment"] = (lead.Positive - lead.Negative) / lead.Comments
        lead = lead.sort_values(["Net sentiment","Avg_compound"], ascending=False)
        st.dataframe(lead.round(3), width="stretch", hide_index=True)
        radar = go.Figure()
        metrics = ["Share of voice", "Net sentiment", "Avg compound", "Avg stars"]
        for _, row in lead.iterrows():
            values = [row["Share of voice %"] / 100, (row["Net sentiment"]+1)/2, (row.Avg_compound+1)/2, (row.Avg_stars-1)/4 if pd.notna(row.Avg_stars) else .5]
            radar.add_trace(go.Scatterpolar(r=values+[values[0]], theta=metrics+[metrics[0]], name=row.Service, fill="toself", opacity=.65))
        radar.update_layout(polar=dict(radialaxis=dict(range=[0,1], visible=True)), title="Normalized comparison (0–1)")
        st.plotly_chart(radar, width="stretch", config=PLOT_CONFIG)
        st.caption("Radar scales: share ÷100; net/compound mapped from −1…1 to 0…1; stars mapped from 1…5 to 0…1.")
        st.subheader("Student rating versus derived stars")
        paired = filtered.dropna(subset=["rating", "derived_stars"])
        if len(paired):
            matrix = pd.crosstab(paired.rating.astype(int), paired.derived_stars.astype(int)).reindex(index=range(1,6), columns=range(1,6), fill_value=0)
            st.plotly_chart(px.imshow(matrix, text_auto=True, labels={"x":"Derived stars","y":"Student rating","color":"Comments"}, color_continuous_scale="Blues"), width="stretch", config=PLOT_CONFIG)
            st.metric("Rating mismatches (≥2 stars)", int(paired.rating_mismatch.sum()))
    else: st.info("Select comments to compare services.")
    st.subheader("Baseline VADER versus selected method")
    changed = filtered[filtered.score.ne(filtered.baseline_score)]
    if len(changed):
        comparison = changed[["feedback","service","baseline_score","baseline_label","score","sentiment"]].copy()
        st.dataframe(comparison.round(3), width="stretch", hide_index=True)
        averages = pd.DataFrame({"Version":["Baseline VADER",engine_name], "Average compound":[filtered.baseline_score.mean(),filtered.score.mean()]})
        st.plotly_chart(px.bar(averages, x="Version", y="Average compound", color="Version"), width="stretch", config=PLOT_CONFIG)
    else: st.info("No selected comment changes compared with baseline VADER.")

with tabs[6]:
    st.subheader("Frequent terms and associations")
    term_service = st.selectbox("Service for term analysis", ["All selected"] + SERVICES)
    subset = filtered if term_service == "All selected" else filtered[filtered.service.eq(term_service)]
    positive, negative, bigrams = Counter(), Counter(), Counter()
    for item in subset.feedback:
        tokens = clean_tokens(item)
        for token in tokens:
            valence = analyzer.lexicon.get(token, 0)
            if valence > 0: positive[token] += 1
            elif valence < 0: negative[token] += 1
        bigrams.update(zip(tokens, tokens[1:]))
    c1, c2 = st.columns(2)
    for target, heading, counts, color in [(c1,"Positive lexicon terms",positive,"#10b981"),(c2,"Negative lexicon terms",negative,"#f43f5e")]:
        target.subheader(heading)
        if counts:
            frame = pd.DataFrame(counts.most_common(12), columns=["Term","Count"])
            target.plotly_chart(px.bar(frame.sort_values("Count"), x="Count", y="Term", orientation="h", color_discrete_sequence=[color]), width="stretch", config=PLOT_CONFIG)
        else: target.info("No matching lexicon terms.")
    st.subheader("Top 15 adjacent word pairs")
    if bigrams:
        frame = pd.DataFrame([(" ".join(pair), count) for pair,count in bigrams.most_common(15)], columns=["Bigram","Count"])
        st.plotly_chart(px.bar(frame.sort_values("Count"), x="Count", y="Bigram", orientation="h", color_discrete_sequence=["#6366f1"]), width="stretch", config=PLOT_CONFIG)
    else: st.info("No word pairs in this selection.")
    st.caption("Simple counts after chart-only stop-word removal; adjacent pairs do not imply causation.")

with tabs[7]:
    st.subheader("Daily sentiment trend")
    dated = filtered.dropna(subset=["timestamp"]).copy()
    if len(dated):
        dated["date"] = dated.timestamp.dt.normalize()
        daily = dated.groupby("date").score.mean().asfreq("D")
        trend = pd.DataFrame({"Date":daily.index,"Daily average":daily.values,"3-day moving average":daily.rolling(3,min_periods=1).mean().values})
        st.plotly_chart(px.line(trend.melt("Date",var_name="Series",value_name="Compound"), x="Date", y="Compound", color="Series", markers=True), width="stretch", config=PLOT_CONFIG)
        st.subheader("Early Warnings")
        multiplier = st.slider("Baseline standard deviations", 0.5, 4.0, 2.0, .25)
        min_volume = st.slider("Minimum comments in a two-day window", 3, 30, 6)
        min_negative = st.slider("Minimum negative comments for a warning", 2, 20, 4)
        st.caption(f"Rule: after at least 7 earlier windows, flag a two-day window with ≥{min_volume} total and ≥{min_negative} negative comments if its negative count exceeds historical mean + {multiplier:g} × population SD or is at least 3× a positive historical mean.")
        warnings = []
        warning_rows = []
        for service, group in dated.groupby("service"):
            dates = pd.date_range(dated.date.min(), dated.date.max(), freq="D")
            negatives = group[group.sentiment.eq("Negative")].groupby("date").size().reindex(dates, fill_value=0)
            counts = negatives.rolling(2,min_periods=2).sum()
            volumes = group.groupby("date").size().reindex(dates, fill_value=0).rolling(2,min_periods=2).sum()
            for i in range(8, len(counts)):
                current = counts.iloc[i]
                volume = volumes.iloc[i]
                history = counts.iloc[1:i].dropna()
                if len(history) < 7: continue
                mean, std = history.mean(), history.std(ddof=0)
                threshold = mean + multiplier * std
                meets_rule = current > threshold or (mean > 0 and current >= 3*mean)
                flagged = volume >= min_volume and current >= min_negative and meets_rule
                warning_rows.append({"Service":service,"Window start":dates[i-1].date(),"Window end":dates[i].date(),"Total comments":int(volume),"Negative":int(current),"Historical mean":round(mean,2),"Historical SD":round(std,2),"Mean + SD threshold":round(threshold,2),"3× baseline":round(3*mean,2),"Warning":bool(flagged)})
                if flagged:
                    warnings.append((service, dates[i-1].date(), dates[i].date(), int(current), round(threshold,2), int(volume)))
        # Show the strongest warning per service so one burst does not flood the page.
        strongest = {}
        for warning in warnings:
            if warning[0] not in strongest or warning[3] > strongest[warning[0]][3]: strongest[warning[0]] = warning
        if strongest:
            for service, start, end, count, threshold, volume in sorted(strongest.values(), key=lambda w:w[3], reverse=True):
                st.error(f"Early Warning · {service} · {start} to {end}: {count} negative of {volume} comments (mean + SD threshold {threshold}).")
        else: st.success("No service exceeds the current spike rule.")
        if warning_rows:
            st.subheader("Warning calculation by service and window")
            warning_table = pd.DataFrame(warning_rows)
            service_windows = st.selectbox("Inspect service baseline", SERVICES)
            st.dataframe(warning_table[warning_table.Service.eq(service_windows)].sort_values("Window end", ascending=False), width="stretch", hide_index=True)
    else: st.info("Valid timestamps are needed for trend analysis.")

with tabs[8]:
    st.subheader("Limitations demo")
    st.caption("Low confidence is a hand-written review flag, not a calibrated probability.")
    examples = [
        ("Oh great, the Wi-Fi is down again.", "A sarcasm phrase plus an explicit failure triggers a cautious negative adjustment and human-review flag."),
        ("I wouldn't say the food was bad.", "Negation can reverse a negative word in a subtle way."),
        ("The food is delicious but the price is terrible.", "One overall score hides opposite aspect opinions."),
        ("Fine.", "A one-word comment has too little context for a reliable interpretation."),
    ]
    for sentence, explanation in examples:
        result = analyzer.polarity_scores(sentence)["compound"]
        with st.container(border=True):
            st.write(f"**{sentence}**")
            st.write(f"{engine_name}: {result:+.3f} ({label3(result)}). {explanation}")
    flagged = filtered[filtered.confidence.eq("Low confidence")]
    st.metric("Low-confidence comments in selection", len(flagged))
    if len(flagged): st.dataframe(flagged[["feedback","confidence_flags","sentiment","score"]], width="stretch", hide_index=True)

with tabs[9]:
    st.subheader("Evaluation against 60 hand-labelled examples")
    st.caption("This small teaching set contains one copy of each selected demo sentence plus six challenge cases. Labels were assigned by a person; it is not an independent campus survey or a claim of real-world accuracy.")
    reference_examples = evaluation_examples()
    method_results = {
        "Baseline VADER": evaluation_table(reference_examples, baseline),
        ENGINE_NAMES[0]: evaluation_table(reference_examples, vader_aware),
        ENGINE_NAMES[1]: evaluation_table(reference_examples, context_aware),
    }
    comparison_results = pd.DataFrame([
        {"Method":name,"Correct":int(result.correct.sum()),"Examples":len(result),"Agreement %":round(100*result.correct.mean(),1)}
        for name, result in method_results.items()
    ])
    st.dataframe(comparison_results, width="stretch", hide_index=True)
    st.caption("This is a small, synthetic teaching set. Compare disagreements rather than assuming the highest agreement will generalize.")
    evaluated = method_results[engine_name]
    correct = int(evaluated.correct.sum())
    st.metric(f"{engine_name}: agreement with reference labels", f"{correct}/{len(evaluated)} ({correct/len(evaluated):.1%})")
    matrix = pd.crosstab(evaluated.human_label, evaluated.predicted_label).reindex(index=["Negative","Neutral","Positive","Mixed"], columns=["Negative","Neutral","Positive","Mixed"], fill_value=0)
    st.plotly_chart(px.imshow(matrix, text_auto=True, labels={"x":"Selected method","y":"Human reference","color":"Examples"}, color_continuous_scale="Blues"), width="stretch", config=PLOT_CONFIG)
    per_class = []
    for label in ["Negative", "Neutral", "Positive", "Mixed"]:
        true_positive = int(((evaluated.human_label == label) & (evaluated.predicted_label == label)).sum())
        predicted = int((evaluated.predicted_label == label).sum())
        actual = int((evaluated.human_label == label).sum())
        precision = true_positive / predicted if predicted else 0
        recall = true_positive / actual if actual else 0
        per_class.append({"Label":label,"Examples":actual,"Precision":round(precision,3),"Recall":round(recall,3),"F1":round(2*precision*recall/(precision+recall),3) if precision+recall else 0})
    st.dataframe(pd.DataFrame(per_class), width="stretch", hide_index=True)
    st.markdown("**Disagreements to study**")
    disagreements = evaluated[~evaluated.correct]
    st.dataframe(disagreements[["feedback","human_label","predicted_label","category"]], width="stretch", hide_index=True)
    st.download_button("Download evaluation results", evaluated.to_csv(index=False).encode("utf-8-sig"), "campus_pulse_evaluation.csv", "text/csv")
    st.divider()
    st.subheader("Human review queue")
    st.caption("Corrections are stored in this Streamlit session and exported beside the selected method's original result. They do not silently change its score.")
    queue_mode = st.selectbox("Review comments", ["Unreviewed low-confidence or rating mismatch", "All comments", "Reviewed comments"])
    if queue_mode == "Unreviewed low-confidence or rating mismatch":
        queue = base[(base.review_status == "Pending") & ((base.confidence == "Low confidence") | base.rating_mismatch)]
    elif queue_mode == "Reviewed comments":
        queue = base[base.review_status == "Reviewed"]
    else:
        queue = base
    st.metric("Comments in queue", len(queue))
    if len(queue):
        selected_id = st.selectbox("Select comment", queue.index.tolist(), format_func=lambda i:f"#{i+1} · {base.loc[i,'service']} · {base.loc[i,'feedback'][:90]}")
        selected = base.loc[selected_id]
        previous = st.session_state.reviews.get(selected_id, {})
        st.info(f"{engine_name}: {selected.sentiment} ({selected.score:+.3f}) · Service: {selected.service} · Flags: {selected.confidence_flags or 'none'}")
        st.write(selected.feedback)
        review_service = st.selectbox("Correct service", SERVICES, index=SERVICES.index(previous.get("service", selected.service)), key=f"review_service_{selected_id}")
        with st.form(f"review_form_{selected_id}"):
            labels = ["Negative", "Neutral", "Positive", "Mixed"]
            review_sentiment = st.selectbox("Correct overall sentiment", labels, index=labels.index(previous.get("sentiment", selected.sentiment)))
            aspect_options = ["No single aspect / not sure"] + list(ASPECTS[review_service])
            old_aspect = previous.get("aspect", "No single aspect / not sure")
            review_aspect = st.selectbox("Main aspect", aspect_options, index=aspect_options.index(old_aspect) if old_aspect in aspect_options else 0)
            review_note = st.text_input("Reason for correction", value=previous.get("note", ""), placeholder="e.g. the comment mixes two opinions")
            save_review = st.form_submit_button("Save review")
        if save_review:
            st.session_state.reviews[selected_id] = {"service":review_service,"sentiment":review_sentiment,"aspect":review_aspect,"note":review_note.strip()}
            st.rerun()
    else:
        st.info("No comments match this review queue.")
    reviewed = base[base.review_status.eq("Reviewed")]
    if len(reviewed):
        st.subheader("Saved corrections")
        st.dataframe(reviewed[["feedback","service","sentiment","aspects","reviewed_service","reviewed_sentiment","reviewed_aspect","review_note"]], width="stretch", hide_index=True)
        st.download_button("Download reviewed comments", reviewed.to_csv(index=False).encode("utf-8-sig"), "campus_pulse_reviews.csv", "text/csv")

with tabs[10]:
    st.subheader("Method & project notes")
    st.markdown("""**Pipeline:** collect comments → keep the original wording → expand a small, explicit list of student abbreviations for scoring → match VADER and campus lexicon entries → apply the selected method's context rules → preserve its numeric score → score clauses separately → label opposite positive and negative clauses Mixed → derive stars only when one overall polarity is appropriate → compare, trend, flag, and export.

**Two selectable algorithms:** `VADER + campus lexicon` is the default. `Campus Context Score` is an independent experimental scorer that reuses the visible word dictionary but calculates its own token scores: three-token negation window, nearby boosters, ALL-CAPS emphasis, pre/post `but` weights, exclamation scaling, and a small emoji dictionary. It normalizes the weighted sum as `sum / (abs(sum) + 4)`. It never calls VADER's `polarity_scores`; it is still rule-based, not trained. A shared sarcasm rule then checks for a phrase such as `oh great` plus a complaint such as `down` and can lower the compound score. The original score and rule evidence remain visible. Both methods are compared against the 60-example teaching set.

**Labels and ratings:** The numeric score uses ±0.05 for Positive/Negative and adjustable ±0.6 defaults for Very Positive/Very Negative. A separate clause rule assigns **Mixed** when at least one clause is positive and another is negative. The raw score and its numeric label remain visible, but Mixed comments receive no single derived star rating. Otherwise stars = `round(((compound + 1) / 2) * 4 + 1)`.

**Technology:** Python, Streamlit, Pandas, Plotly, vaderSentiment. No trained classifier is used.""")
    st.dataframe(pd.DataFrame([
        ("Pipeline inspector, abbreviations, and Mixed label", "Sentiment Analysis; Negation; Multipolarity"),
        ("Derived stars and rating mismatch", "User Reviews and Rating Analysis; Rating Derivation"),
        ("Aspect sentiment and priority actions", "Opinion Mining; Multipolarity"),
        ("Service leaderboard and radar", "Brand Monitoring and Comparison Analysis"),
        ("Term and bigram counts", "Frequent Terms and Associations"),
        ("Daily trends and early warnings", "Trend Detection; Brand Monitoring"),
        ("Low-confidence review flags and sarcasm adjustment", "Challenges: Sarcasm, Negation and Multipolarity"),
        ("Campus lexicon before/after", "Lexicon-Based Approach; Rule-Based Sentiment Analysis"),
        ("Campus Context Score versus VADER", "Rule-Based versus Lexicon-Based; Sentiment Evaluation"),
        ("Hand-labelled evaluation and review queue", "Sentiment Analysis; Challenges and human interpretation"),
    ], columns=["Feature", "Project area"]), width="stretch", hide_index=True)
    st.markdown("""**Limitations and mitigations:** Possible sarcasm, ambiguous negation, mixed aspect sentiment, short comments, non-English text, and slang are flagged for human review. The sarcasm adjustment requires a known phrase and explicit complaint; genuine praise with the same wording or unseen sarcastic phrasing can still be misread. Its weights were chosen for teaching, not learned from data. Expanded dictionaries cover selected campus phrases, while multi-aspect clauses are marked uncertain. Alerts display their baseline and require minimum volume. Both methods are English-oriented, phrase splitting can lose context, keyword detection selects one service, and the synthetic demo and hand-labelled teaching set do not establish accuracy on real campus feedback. Review original comments before decisions.""")
st.caption("🎓 Campus Pulse AI · Campus feedback intelligence · Academic prototype")
