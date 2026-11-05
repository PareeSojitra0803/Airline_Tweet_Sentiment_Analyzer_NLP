import re
import string

POSITIVE_WORDS = {
    "amazing", "awesome", "best", " love", "love", "great", "excellent", "good",
    "wonderful", "fantastic", "helpful", "friendly", "thanks", "thank", "thankyou",
    "smooth", "quick", "easy", "comfortable", "appreciate", "impressed", "on time",
    "pleasant", "happy", "professional", "kind", "brilliant", "perfect", "enjoy",
    "awesome", "outstanding"
}
NEGATIVE_WORDS = {
    "bad", "worst", "terrible", "awful", "hate", "horrible", "delay", "delayed",
    "cancelled", "canceled", "rude", "lost", "missing", "late", "poor", "angry",
    "disappointed", "disappointing", "frustrated", "broken", "unhelpful", "waiting",
    "refund", "cancel", "failure", "unacceptable", "never", "stuck", "problem",
    "issue", "worse", "missed"
}
NEGATIONS = {"no", "not", "nor", "never", "n't"}

def clean_text(text):
    """Normalize a tweet while preserving hashtag words and sentiment negation."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # URLs
    text = re.sub(r"@\w+", " ", text)                   # mentions
    text = re.sub(r"#(\w+)", r"\1", text)              # keep hashtag word
    text = re.sub(r"[^a-z\s]", " ", text)              # punctuation, digits, emoji
    text = re.sub(r"\s+", " ", text).strip()
    return text

def heuristic_sentiment(text):
    """Tiny transparent lexicon baseline; negation flips nearby polarity."""
    tokens = clean_text(text).split()
    pos = neg = 0
    for i, token in enumerate(tokens):
        score = 1 if token in POSITIVE_WORDS else (-1 if token in NEGATIVE_WORDS else 0)
        if score and any(t in NEGATIONS for t in tokens[max(0, i-2):i]):
            score *= -1
        if score > 0:
            pos += 1
        elif score < 0:
            neg += 1
    label = "positive" if pos > neg else ("negative" if neg > pos else "neutral")
    return label, pos, neg
