from typing import List

from Services.PreProcessingService.i_text_processor import ITextProcessor


class TextProcessorDecorator(ITextProcessor):
    def __init__(self, wrapped_processor: ITextProcessor):
        self._wrapped_processor = wrapped_processor

    def process(self, text: str) -> List[str]:
        return self._wrapped_processor.process(text)
