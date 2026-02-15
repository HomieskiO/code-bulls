import pandas as pd
import re
import json

# 1. טען את ה-CSV שלך
df = pd.read_csv('Lone User Tweets.csv')


def clean_tweet_text(text):
    if not isinstance(text, str):
        return ""
    # הסרת קישורים
    text = re.sub(r'http\S+', '', text)
    # הסרת טיקרים ($SPY) כדי לא לבלבל את המודל
    text = re.sub(r'\$[A-Za-z]+', '', text)
    # ניקוי שורות חדשות ורווחים כפולים
    text = text.replace('\\n', ' ').replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def categorize_tweet(text):
    text_lower = text.lower()

    # סדר הבדיקה חשוב! קודם בודקים פונדמנטלי כדי למנוע טעויות
    fund_keywords = ['fundamental', 'earnings', 'sales', 'eps', 'revenue', 'growth', 'valuation', 'quarter']
    if any(k in text_lower for k in fund_keywords):
        return "Fundamental Analysis"

    risk_keywords = ['risk', 'stop', 'loss', 'cut', 'protect', 'capital', 'position size', 'exposure', 'max loss',
                     'fail']
    if any(k in text_lower for k in risk_keywords):
        return "Risk Management"

    tech_keywords = ['pivot', 'breakout', 'setup', 'set up', 'chart', 'support', 'resistance',
                     'volume', 'base', 'tight', 'consolidation', '10-week', '20-day', '50-day',
                     '200-day', 'line', 'trend', 'gap', 'wedge', 'flag', 'candle', 'moving average',
                     'ma', 'rsi', 'macd', 'technical', 'pattern', 'extension', 'extended', 'cup', 'handle', 'pocket']
    if any(k in text_lower for k in tech_keywords):
        return "Technical Analysis"

    psych_keywords = ['patience', 'wait', 'fear', 'greed', 'mindset', 'emotion', 'discipline', 'fomo', 'mental',
                      'psychology', 'conviction']
    if any(k in text_lower for k in psych_keywords):
        return "Psychology"

    return "Market Philosophy"


def score_impact(text):
    text_lower = text.lower()
    # מילים שמעידות על חוק ברזל
    high_impact = ['never', 'always', 'must', 'rule', 'critical', 'key', 'don\'t', 'do not', 'only', 'avoid', 'huge',
                   'major']

    if any(w in text_lower for w in high_impact):
        return 3
    elif len(text.split()) > 15:
        return 2
    else:
        return 1


# --- ביצוע התהליך ---
labeled_data = []

for index, row in df.iterrows():
    raw_text = row['full_text']
    clean_text = clean_tweet_text(raw_text)

    # סינון ציוצים ריקים או קצרים מדי
    if len(clean_text) < 5:
        continue

    category = categorize_tweet(clean_text)
    impact = score_impact(clean_text)

    entry = {
        "quote": clean_text,  # הטקסט הנקי למודל
        "category": category,  # הקטגוריה לסינון
        "impact": impact,  # חשיבות הציטוט
        "original_date": row['created_at'],
        "source_url": row['url']  # לינק לציוץ המקורי
    }
    labeled_data.append(entry)

# שמירה לקובץ
output_filename = 'labeled_trading_quotes.json'
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(labeled_data, f, ensure_ascii=False, indent=4)

print(f"בוצע בהצלחה! {len(labeled_data)} ציטוטים נשמרו בקובץ {output_filename}")