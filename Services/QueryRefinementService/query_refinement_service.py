import re
from dataclasses import dataclass, field
from typing import Dict, List

import nltk
from nltk.corpus import stopwords, wordnet
from spellchecker import SpellChecker

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)

_WORD_RE = re.compile(r"[A-Za-z]+")


@dataclass
class RefinementResult:
    original: str
    refined: str
    corrections: Dict[str, str] = field(default_factory=dict)   # misspelled -> corrected
    added_synonyms: Dict[str, List[str]] = field(default_factory=dict)  # word -> synonyms added


class QueryRefinementService:
    """Improves a raw query before retrieval via spelling correction and synonym expansion.

    Both steps run on natural words (no stemming here); the downstream strategies still
    apply their own preprocessing afterwards.
    """

    def __init__(self):
        self._spell = SpellChecker()
        self._stop_words = set(stopwords.words('english'))

    def correct_spelling(self, query: str):
        # Replace each unknown alphabetic word with its most likely correction.
        words = query.split()
        corrections: Dict[str, str] = {}
        corrected_words = []
        for word in words:
            lower = word.lower()
            if _WORD_RE.fullmatch(lower) and lower in self._spell.unknown([lower]):
                suggestion = self._spell.correction(lower)
                if suggestion and suggestion != lower:
                    corrections[lower] = suggestion
                    corrected_words.append(suggestion)
                    continue
            corrected_words.append(word)
        return " ".join(corrected_words), corrections

    def expand_synonyms(self, query: str, max_per_word: int = 2):
        # Append up to `max_per_word` WordNet synonyms for each content word.
        added: Dict[str, List[str]] = {}
        extra_terms: List[str] = []
        seen = {w.lower() for w in _WORD_RE.findall(query)}

        for word in _WORD_RE.findall(query):
            lower = word.lower()
            if lower in self._stop_words:
                continue
            synonyms = self._synonyms_for(lower, seen, max_per_word)
            if synonyms:
                added[lower] = synonyms
                extra_terms.extend(synonyms)
                seen.update(synonyms)

        expanded = query if not extra_terms else f"{query} {' '.join(extra_terms)}"
        return expanded, added

    def suggest(self, query: str, max_suggestions: int = 6) -> List[str]:
        # Return alternative queries the user can pick from (instead of auto-replacing).
        suggestions: List[str] = []
        seen = {query.strip().lower()}

        def add(candidate: str):
            candidate = candidate.strip()
            if candidate and candidate.lower() not in seen:
                seen.add(candidate.lower())
                suggestions.append(candidate)

        # 1) Spelling-corrected version (if anything changed).
        corrected, corrections = self.correct_spelling(query)
        if corrections:
            add(corrected)

        # 2) Per-word synonym substitutions (natural rephrasings, e.g. "automobile insurance").
        words = corrected.split()
        lower_words = [w.lower() for w in words]
        for i, word in enumerate(words):
            if len(suggestions) >= max_suggestions:
                break
            lower = word.lower()
            if not _WORD_RE.fullmatch(lower) or lower in self._stop_words:
                continue
            synonyms = self._synonyms_for(lower, set(lower_words), 1)
            if synonyms:
                variant = words.copy()
                variant[i] = synonyms[0]
                add(" ".join(variant))

        # 3) One synonym-expanded version (original terms plus synonyms appended).
        expanded, added = self.expand_synonyms(corrected, max_per_word=1)
        if added:
            add(expanded)

        return suggestions[:max_suggestions]

    def refine(self, query: str, correct: bool = True, expand: bool = True,
               max_synonyms_per_word: int = 2) -> RefinementResult:
        result = RefinementResult(original=query, refined=query)

        if correct:
            result.refined, result.corrections = self.correct_spelling(result.refined)
        if expand:
            result.refined, result.added_synonyms = self.expand_synonyms(
                result.refined, max_synonyms_per_word)

        return result

    def _synonyms_for(self, word: str, seen: set, max_per_word: int) -> List[str]:
        synonyms: List[str] = []
        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                candidate = lemma.name().replace("_", " ").lower()
                # Skip the word itself, duplicates, and multi-word lemmas for simplicity.
                if candidate in seen or " " in candidate:
                    continue
                synonyms.append(candidate)
                seen.add(candidate)
                if len(synonyms) >= max_per_word:
                    return synonyms
        return synonyms
