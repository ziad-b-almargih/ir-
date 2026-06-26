import nltk
from typing import List
from nltk.corpus import stopwords

from Services.PreProcessingService.base_decorator import TextProcessorDecorator

nltk.download('stopwords', quiet=True)


class RemoveStopWordsDecorator(TextProcessorDecorator):
    def __init__(self, wrapped_processor):
        super().__init__(wrapped_processor)
        self.stop_words = set(stopwords.words('english'))

    def process(self, text: str) -> List[str]:
        tokens = super().process(text)
        return [token for token in tokens if token not in self.stop_words]
