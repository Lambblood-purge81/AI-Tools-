import re
import textstat
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Analyze real-time spoken transcript for feedback
def analyze_response(transcript):
    try:
        word_count = len(transcript.split())
        reading_level = textstat.flesch_reading_ease(transcript)
        sentiment = sia.polarity_scores(transcript)

        filler_words = ["um", "uh", "like", "you know", "so", "actually", "basically"]
        filler_count = sum(transcript.lower().count(word) for word in filler_words)

        long_pauses = len(re.findall(r"\.\.\.", transcript))

        feedback = {
            "word_count": word_count,
            "reading_ease": reading_level,
            "sentiment": sentiment,
            "filler_words_used": filler_count,
            "long_pauses_detected": long_pauses,
            "overall_score": round((reading_level + (sentiment['compound'] * 100) - (filler_count * 2)), 2)
        }
        return feedback

    except Exception as e:
        return {"error": str(e)}
