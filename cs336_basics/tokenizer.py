from typing import Iterable, Iterator

import regex as re


class Word:

    def __init__(self, word: str, is_special = False):
        self.is_special = is_special
        if is_special:
            self.word_bytes = [bytes(list(word.encode("utf-8")))]
        else:
            self.word_bytes = Word.word2bytes(word)
            self.byte_size = len(self.word_bytes)

    def is_special(self):
        return self.is_special
    
    @staticmethod
    def word2bytes(word: str) -> tuple[bytes,...]:
        "Convert word string to tuple of bytes"
        a = list(word.encode("utf-8"))
        return tuple(bytes([i]) for i in a)
    
    def merge(self, pair: tuple[bytes, bytes]):
        if self.is_special:
            return
        if len(pair[0]) + len(pair[1]) > self.byte_size:
            return
        assert len(pair) == 2
        merged = pair[0] + pair[1]
        i = 0
        new_word_bytes = []
        while i < len(self.word_bytes):
            # Check for match
            if i < len(self.word_bytes) - 1 and self.word_bytes[i] == pair[0] and self.word_bytes[i + 1] == pair[1]:
                new_word_bytes.append(merged)
                i += 2
            else:
                new_word_bytes.append(self.word_bytes[i])
                i += 1
        self.word_bytes = new_word_bytes

    def apply_merges(self, pairs: list[tuple[bytes, bytes]]):
        if self.is_special:
            return
        for pair in pairs:
            self.merge(pair)

    def as_ids(self, vocab: dict[bytes, int]) ->list[int]:
        return [vocab[bs] for bs in self.word_bytes]
            

class Tokenizer:

    def __init__(self, vocab: dict[int, bytes],merges: list[tuple[bytes, bytes]],special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab_by_word = {v: k for k,v in vocab.items()}

    def encode(self, text: str) -> list[int]:
        return list(self.encode_iterable([text]))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        for text in iterable:
            chunks = self._split_by_special(text, self.special_tokens, False)
            for chunk in chunks:
                if self.special_tokens is not None and chunk in self.special_tokens:
                    yield self.vocab_by_word[bytes(list(chunk.encode("utf-8")))]
                else:
                    for m in PAT.finditer(chunk):
                        word = Word(m.group(0))
                        word.apply_merges(self.merges)
                        for i in word.as_ids(self.vocab_by_word):
                            yield i

    def decode(self, ids: list[int]) -> str:
        tokens = []
        for i in ids:
            if i in self.vocab.keys():
                tokens.append(self.vocab[i])
            else:
                tokens.append(b"\xef\xbf\xbd")
        return b"".join(tokens).decode("utf-8", "replace")
    
    def _split_by_special(cls,text, special_tokens, drop_special=True):
        if not special_tokens:
            return [text]

        # Sort by descending length to prioritize longer tokens (e.g., "<|endoftext|><|endoftext|>" before "<|endoftext|>")
        special_tokens = sorted(special_tokens, key=len, reverse=True)

        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        if not drop_special:
            pattern = f"({pattern})"

        pattern = re.compile(pattern)
        chunks = pattern.split(text)
        return [c for c in chunks if c]  # remove empty strings

