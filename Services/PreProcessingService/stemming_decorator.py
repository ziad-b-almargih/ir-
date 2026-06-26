from typing import List
from nltk.stem import PorterStemmer

from Services.PreProcessingService.base_decorator import TextProcessorDecorator


class StemmingDecorator(TextProcessorDecorator):
    def __init__(self, wrapped_processor):
        super().__init__(wrapped_processor)
        self.stemmer = PorterStemmer()

    def process(self, text: str) -> List[str]:
        tokens = super().process(text)
        return [self.stemmer.stem(token) for token in tokens]
