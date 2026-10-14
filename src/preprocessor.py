"""
Text Preprocessing Module
Handles all text cleaning and normalization for spam detection.
"""

import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer


def download_nltk_resources():
    """Download required NLTK datasets (safe to call multiple times)."""
    resources = ['stopwords', 'punkt', 'punkt_tab', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass


# Initialise once at module level so objects are reused across calls
download_nltk_resources()
_stemmer = PorterStemmer()
_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words('english'))
_punctuation = set(string.punctuation)


def preprocess_text(text: str, method: str = 'stem') -> str:
    """
    Clean and normalise raw SMS text.

    Steps
    -----
    1. Lowercase
    2. Tokenise
    3. Remove punctuation tokens
    4. Remove English stop-words
    5. Stem  (method='stem')  or  Lemmatise  (method='lemma')

    Parameters
    ----------
    text   : Raw input string.
    method : 'stem' uses PorterStemmer; 'lemma' uses WordNetLemmatizer.

    Returns
    -------
    Cleaned string joined by spaces.
    """
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _punctuation]
    tokens = [t for t in tokens if t not in _stop_words]

    if method == 'lemma':
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    else:
        tokens = [_stemmer.stem(t) for t in tokens]

    return ' '.join(tokens)
