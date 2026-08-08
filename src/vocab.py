"""
Tokenization and vocabulary construction for the LSTM seq2seq model.

We use simple whitespace tokenization after light normalization. This is
intentionally simple (no subword/BPE) so the whole pipeline -- including
vocabulary construction -- is implemented by us, as required by the
assignment (no HuggingFace tokenizers / prebuilt seq2seq pipelines).
"""
import re
import unicodedata
from collections import Counter

PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<pad>", "<sos>", "<eos>", "<unk>"
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


def unicode_to_ascii_preserve_accents(s: str) -> str:
    """
    Normalize unicode form (NFC) without stripping accents -- Spanish accents
    (á, é, í, ó, ú, ñ, ü) are meaningful and must be preserved.
    """
    return unicodedata.normalize("NFC", s)


def normalize_string(s: str) -> str:
    """
    Lowercase, normalize unicode, and add spaces around punctuation so that
    punctuation tokens are separated from words by whitespace tokenization.
    Keeps letters (incl. accented), digits, and basic sentence punctuation.
    """
    s = unicode_to_ascii_preserve_accents(s.strip().lower())
    # Add space before/after punctuation we care about
    s = re.sub(r"([.!?,;:¿¡])", r" \1 ", s)
    # Collapse anything that isn't a letter/number/accented char/punctuation into a space
    s = re.sub(r"[^a-zA-Z0-9áéíóúüñÁÉÍÓÚÜÑ.!?,;:¿¡'\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str):
    return normalize_string(s).split(" ") if s else []


class Vocab:
    def __init__(self, min_freq: int = 2, max_size: int = None):
        self.min_freq = min_freq
        self.max_size = max_size
        self.word2idx = {}
        self.idx2word = {}
        self.freqs = Counter()

    def build(self, tokenized_sentences):
        """Build vocabulary ONLY from the given sentences (e.g. train split)."""
        for tokens in tokenized_sentences:
            self.freqs.update(tokens)

        # Start with special tokens at fixed indices
        self.word2idx = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.idx2word = {i: tok for tok, i in self.word2idx.items()}

        # Sort by frequency (desc), then alphabetically for determinism
        items = sorted(self.freqs.items(), key=lambda kv: (-kv[1], kv[0]))
        for word, freq in items:
            if freq < self.min_freq:
                continue
            if self.max_size and len(self.word2idx) >= self.max_size:
                break
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        return self

    def __len__(self):
        return len(self.word2idx)

    def encode(self, tokens, add_sos_eos=True):
        ids = [self.word2idx.get(t, UNK_IDX) for t in tokens]
        if add_sos_eos:
            ids = [SOS_IDX] + ids + [EOS_IDX]
        return ids

    def decode(self, ids, strip_special=True):
        words = []
        for i in ids:
            tok = self.idx2word.get(int(i), UNK_TOKEN)
            if strip_special and tok in (PAD_TOKEN, SOS_TOKEN):
                continue
            if strip_special and tok == EOS_TOKEN:
                break
            words.append(tok)
        return " ".join(words)

    def to_dict(self):
        return {"word2idx": self.word2idx, "min_freq": self.min_freq, "max_size": self.max_size}

    @classmethod
    def from_dict(cls, d):
        v = cls(min_freq=d.get("min_freq", 2), max_size=d.get("max_size"))
        v.word2idx = d["word2idx"]
        v.idx2word = {int(i): w for w, i in v.word2idx.items()}
        return v
