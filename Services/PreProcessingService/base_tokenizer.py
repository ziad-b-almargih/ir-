import nltk
from typing import List
from nltk.tokenize import word_tokenize

from Services.PreProcessingService.i_text_processor import ITextProcessor

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


class BaseTokenizer(ITextProcessor):
    def process(self, text: str) -> List[str]:
        if not text or not isinstance(text, str):
            return []
        return word_tokenize(text)
