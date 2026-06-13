import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Auto-verify and secure essential NLTK asset compliance
def _initialize_assets():
    for asset in ['stopwords', 'punkt', 'wordnet', 'omw-1.4']:
        try:
            nltk.data.find(f'tokenizers/{asset}' if asset=='punkt' else f'corpora/{asset}')
        except LookupError:
            nltk.download(asset, quiet=True)

_initialize_assets()

class TextProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Retain critical sentiment-carrying modifiers within the corpus
        negation_exceptions = {"not", "no", "never", "neither", "nor", "barely", "hardly"}
        self.stop_words = self.stop_words - negation_exceptions

    def clean_text(self, text):
        """Executes structural structural sanitization over raw textual inputs."""
        if not isinstance(text, str):
            return ""
        # Lowercase and strip HTML tags/line breaks
        text = re.sub(r'<[^>]+>', ' ', text.lower())
        # Strip structural special characters but preserve punctuation anchors
        text = re.sub(r'[^a-zA-Z\s\?!\']', '', text)
        
        tokens = word_tokenize(text)
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        return " ".join(cleaned_tokens)