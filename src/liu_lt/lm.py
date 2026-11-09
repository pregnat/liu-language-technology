# A simple n-gram language model
# ---
# Copyright –2023 by Marco Kuhlmann <marco.kuhlmann@liu.se>
# Modified 2025 by Marcel Bollmann <marcel.bollmann@liu.se>

import math
import random
import sys

from collections import Counter, deque
from typing import overload, Generator, Iterable, Optional

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"


def make_deque(n: int) -> deque:
    return deque([BOS] * (n - 1), maxlen=(n - 1))


def ngrams(
    sentence: Iterable[str], n: int
) -> Generator[tuple[tuple[str, ...], str], None, None]:
    history = make_deque(n)
    for word in sentence + [EOS]:
        yield tuple(history), word
        history.append(word)


class OutOfVocabularyError(Exception):
    def __init__(self, word: str) -> None:
        super().__init__(word)


class CachedCounter(Counter):
    def __init__(self) -> None:
        super().__init__()
        self._total = None

    def total(self) -> int:
        if self._total is None:
            self._total = sum(self.values())
        return self._total


class NgramLM:
    def __init__(self, order: int) -> None:
        """A simple n-gram language model.

        Arguments:
          order: The order of the model (i.e., the value for n).
        """
        self._n: int = order
        self._vocab: set[str] = set([BOS, EOS])
        self._counts: tuple[str, ...] = {}

    @property
    def order(self) -> int:
        """The model's order."""
        return self._n

    @property
    def vocab(self) -> set[str]:
        """The model's vocabulary, as a set of strings."""
        return self._vocab

    @property
    def has_unk(self) -> bool:
        return UNK in self._vocab

    @overload
    def _mask_or_raise(self, words: str) -> str:  # pragma: no cover
        ...

    @overload
    def _mask_or_raise(
        self, words: tuple[str, ...]
    ) -> tuple[str, ...]:  # pragma: no cover
        ...

    def _mask_or_raise(self, words: str | tuple[str, ...]) -> str | tuple[str, ...]:
        if isinstance(words, str):
            if words in self._vocab:
                return words
            elif self.has_unk:
                return UNK
            raise OutOfVocabularyError(words)
        masked = []
        for word in words:
            if word in self._vocab:
                masked.append(word)
            elif self.has_unk:
                masked.append(UNK)
            else:
                raise OutOfVocabularyError(word)
        return tuple(masked)

    def freq(self, context: tuple[str, ...], word: str) -> int:
        """Retrieve the absolute frequency of a word in a given context."""
        assert (
            len(context) == self.order - 1
        ), "Context length must be one less than the model's order"
        context = self._mask_or_raise(context)
        word = self._mask_or_raise(word)
        if context in self._counts:
            return self._counts[context].get(word, 0)
        else:
            return 0

    def total(self, context: tuple[str, ...]) -> int:
        """Retrieve the absolute frequency of a given context."""
        assert (
            len(context) == self.order - 1
        ), "Context length must be one less than the model's order"
        context = self._mask_or_raise(context)
        if context in self._counts:
            return self._counts[context].total()
        else:
            return 0

    def fit(self, sentences: Iterable[Iterable[str]]) -> Self:
        """Fit this n-gram model on a training corpus.

        Arguments:
          sentences: A list of sentences, where each sentence is represented as a list of tokens.
        """
        for sentence in sentences:
            for context, word in ngrams(sentence, self.order):
                if context not in self._counts:
                    self._counts[context] = CachedCounter()
                self._counts[context][word] += 1
                self._vocab.add(word)
        return self

    def prob(self, context: tuple[str, ...], word: str) -> float:
        """Retrieve the probability of seeing a word given a context."""
        a = self.freq(context, word)
        b = self.total(context)
        return a / b

    def logprob(self, context: tuple[str, ...], word: str) -> float:
        """Retrieve the log-probability of seeing a word given a context."""
        return -math.log(self.prob(context, word), 2)

    def generate(self, start: Optional[Iterable[str]] = None) -> list[str]:
        """Generate a sentence from this n-gram model.

        Arguments:
          start: A list of context words to condition the sentence on.
        """
        history = make_deque(self.order)
        sentence = []
        if start is not None:
            for word in start:
                sentence.append(word)
                history.append(word)
        while True:
            context = tuple(history)
            r = random.random()
            p = 0
            for word in self._counts[context]:
                p += self.prob(context, word)
                if p > r:
                    break
            if word == EOS:
                return sentence
            sentence.append(word)
            history.append(word)

    def perplexity(self, sentences: Iterable[Iterable[str]]) -> float:
        """Compute the perplexity on a list of sentences."""
        logprob, m = 0, 0
        for sentence in sentences:
            for context, word in ngrams(sentence, self.order):
                logprob += self.logprob(context, word)
                m += 1
        entropy = logprob / m
        return math.pow(2.0, entropy)
