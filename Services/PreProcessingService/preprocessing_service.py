from typing import List

from Services.PreProcessingService.base_tokenizer import BaseTokenizer
from Services.PreProcessingService.lowercase_decorator import LowerCaseDecorator
from Services.PreProcessingService.remove_punctuation_decorator import RemovePunctuationDecorator
from Services.PreProcessingService.remove_stopwords_decorator import RemoveStopWordsDecorator
from Services.PreProcessingService.stemming_decorator import StemmingDecorator


class PreprocessingService:
    def __init__(self):
        processor = BaseTokenizer()
        processor = LowerCaseDecorator(processor)
        processor = RemovePunctuationDecorator(processor)
        processor = RemoveStopWordsDecorator(processor)
        self._pipeline = StemmingDecorator(processor)

    def process_text(self, text: str) -> List[str]:
        return self._pipeline.process(text)
