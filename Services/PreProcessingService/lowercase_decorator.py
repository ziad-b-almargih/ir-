from typing import List

from Services.PreProcessingService.base_decorator import TextProcessorDecorator


class LowerCaseDecorator(TextProcessorDecorator):
    def process(self, text: str) -> List[str]:
        tokens = super().process(text)
        return [token.lower() for token in tokens]
