from abc import ABC, abstractmethod
from typing import List


class ITextProcessor(ABC):
    @abstractmethod
    def process(self, text: str) -> List[str]:
        pass
