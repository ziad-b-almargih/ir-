import string
from typing import List

from Services.PreProcessingService.base_decorator import TextProcessorDecorator


class RemovePunctuationDecorator(TextProcessorDecorator):
    def __init__(self, wrapped_processor):
        super().__init__(wrapped_processor)
        self.punctuations = set(string.punctuation).union({"''", "``", "...", ".."})

    def process(self, text: str) -> List[str]:
        tokens = super().process(text)

        return [
            token for token in tokens
            if not all(char in self.punctuations for char in token)
        ]
