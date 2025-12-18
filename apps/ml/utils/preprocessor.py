import re
import string

import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


nltk.download("stopwords")
nltk.download("punkt_tab")
nltk.download("wordnet")


class TextPreprocessor:
    def __init__(
        self,
        language: str = "english",
        *,
        use_lemmatization: bool = True,
        custom_stopwords: list[str] | None = None,
    ) -> None:
        self.stop_words = set(stopwords.words(language))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)

        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        self.stemmer = PorterStemmer() if not use_lemmatization else None

        # Extending delete-templates
        self.patterns_to_remove = [
            r"{{.*?}}",  # Template figure brackets
            r"\[.*?\]",  # Template square brackets
            r"<.*?>",  # HTML tags
            r"http\S+",  # URL
            r"@\w+",  # Mentions
            r"#\w+",  # HashTags
        ]

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = text.lower()

        for pattern in self.patterns_to_remove:
            text = re.sub(pattern, "", text)

        # Deleting punctuation and digits
        text = text.translate(str.maketrans("", "", string.punctuation + string.digits))

        tokens = word_tokenize(text)

        # Filtering and normalizing
        filtered_tokens = []
        for token in tokens:
            if (
                len(token) > 1  # Removing single symbols
                and token.isalpha()
                and token not in self.stop_words
            ):
                if self.lemmatizer:
                    modified_token = self.lemmatizer.lemmatize(token)
                elif self.stemmer:
                    modified_token = self.stemmer.stem(token)

                filtered_tokens.append(modified_token)

        return " ".join(filtered_tokens)


preprocessor = TextPreprocessor()
