import pickle
from abc import ABC, abstractmethod
from typing import List, Dict

import torch


class TokenizerBase(ABC):
    """
    base class for text tokenizers that convert text to numerical tokens and back.

    provides methods for encoding text to token ids, decoding ids to text, and handling special tokens (start, end, padding, unknown).

    attributes:
        special_tokens (List[str]): special tokens used by the tokenizer
        idx_to_token (Dict[int, str]): token index to string mapping
        token_to_idx (Dict[str, int]): token string to index mapping
    """

    def __init__(self) -> None:
        self.special_tokens: List[str] = []
        self.idx_to_token: Dict[int, str] = {}
        self.token_to_idx: Dict[str, int] = {}

    @property
    @abstractmethod
    def start_token(self) -> str:
        """beginning of the sequence token"""
        pass

    @property
    @abstractmethod
    def end_token(self) -> str:
        """end of sequence token"""
        pass

    @property
    @abstractmethod
    def pad_token(self) -> str:
        """padding sequences"""
        pass

    @property
    @abstractmethod
    def unk_token(self) -> str:
        """unknown or out-of-vocabulary characters token"""
        pass

    @property
    def num_tokens(self) -> int:
        """
        get the total number of unique tokens in the tokenizer's vocabulary

        returns:
            int: the total number of tokens in the vocabulary, including special tokens
        """
        return len(self.idx_to_token)

    @abstractmethod
    def train(self, file_path: str, n_tokens: int = -1, start_token: str = "<s>",
              end_token: str = "</s>", pad_token: str = "<p>", unk_token: str = "<unk>") -> None:
        """
        train the tokenizer on a dataset to build its vocabulary.

        args:
            file_path: path to the training data file
            n_tokens: maximum number of tokens to include in the vocabulary (-1 for no limit)
            start_token: token to represent the start of a sequence
            end_token: token to represent the end of a sequence
            pad_token: token to use for padding sequences to equal length
            unk_token: token to use for unknown or out-of-vocabulary characters

        note:
            the implementation should populate idx_to_token and token_to_idx mappings
            and set up special tokens based on the training data.
        """
        pass

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        encode a text string into a list of token IDs.

        args:
            text: the input text to encode
            add_special_tokens: whether to add start and end tokens to the encoded sequence

        returns:
            a list of integer token ids representing the encoded text
        """
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """
        decode a list of token IDs back into a text string.

        args:
            tokens: list of integer token IDs to decode

        returns:
            the decoded text string
        """
        pass

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        encode a batch of text strings into token IDs.

        args:
            list of input text strings to encode

        returns:
        lList of lists containing token ids for each input text
        """
        pass

    def decode_batch(self, batch_tokens: List[List[int]]) -> List[str]:
        """
        decode a batch of token ID sequences back into text strings.

        args:
            batch_tokens: list of lists containing token ids to decode

        returns:
            list of decoded text strings
        """
        pass

    def save(self, filepath: str) -> None:
        """
        save the tokenizer instance to a file using pickle serialization.

        args:
            filepath: path where the serialized tokenizer should be saved

        note:
            this method uses pickle for serialization, which may not be secure for
            untrusted data sources.
        """
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'TokenizerBase':
        """
        load a tokenizer instance from a file using pickle deserialization.

        args:
            filepath: Path to the saved tokenizer file

        returns:
            the loaded tokenizer instance

        note:
            this method uses pickle for deserialization, which may not be secure for
            untrusted data sources.
        """
        with open(filepath, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded from {filepath}")
        return model

    def tokenize(self, text: str) -> List[str]:
        """
        split text into tokens without converting to token IDs.

        args:
            text: input text to tokenize

        returns:
            list of token strings
        """
        return [self.decode(self.encode(i, add_special_tokens=False)) for i in text.split()]


class CharacterTokenizer(TokenizerBase):
    """
    a character-level tokenizer that treats each character as a separate token.

    this tokenizer creates a vocabulary from individual characters in the training data
    and handles text encoding/decoding at the character level. It includes special tokens
    for sequence start, end, padding, and unknown characters.
    """

    def __init__(self) -> None:
        """initialize the character tokenizer with default special tokens."""
        super().__init__()
        self._start_token: str = "<s>"
        self._end_token: str = "</s>"
        self._pad_token: str = "<p>"
        self._unk_token: str = "<unk>"

    @property
    def start_token(self) -> str:
        return self._start_token

    @property
    def end_token(self) -> str:
        return self._end_token

    @property
    def pad_token(self) -> str:
        return self._pad_token

    @property
    def unk_token(self) -> str:
        return self._unk_token

    def train(self, file_path: str, n_tokens: int = -1, start_token: str = "<s>",
              end_token: str = "</s>", pad_token: str = "<p>", unk_token: str = "<unk>") -> None:
        """
        train the character tokenizer on a text file.

        args:
            file_path: path to the training data file
            n_tokens: maximum number of tokens to include (not used in character tokenizer)
            start_token: token to represent sequence start
            end_token: token to represent sequence end
            pad_token: token to use for padding
            unk_token: token to use for unknown characters

        note:
            the character tokenizer creates a vocabulary from all unique characters
            in the training data plus the special tokens.
        """
        with open(file_path, 'r') as f:
            data = ''.join(f.readlines())
        self.special_tokens = [start_token, end_token, pad_token, unk_token]
        self._start_token = start_token
        self._end_token = end_token
        self._pad_token = pad_token
        self._unk_token = unk_token

        tokens = self.special_tokens + list(set(data))
        self.idx_to_token = {idx: token for idx, token in enumerate(tokens)}
        self.token_to_idx = {token: idx for idx, token in enumerate(tokens)}

    def encode(self, text: str, add_special_tokens: bool = True) -> torch.Tensor:
        """
        encode text into character-level token IDs.

        args:
            text: Input text to encode
            add_special_tokens: Whether to add start and end tokens

        returns:
            list of integer token ids
        """
        token_ids = [self.token_to_idx.get(i, self.token_to_idx[self.unk_token]) for i in text]
        if add_special_tokens:
            token_ids.insert(0, self.token_to_idx[self.start_token])
            token_ids.append(self.token_to_idx[self.end_token])
        return torch.tensor(token_ids)

    def pad(self, token_ids: List[List[int]]) -> List[List[int]]:
        """
        pad a batch of token ID sequences to the same length.

        args:
            token_ids: List of token id sequences to pad

        returns:
            padded token id sequences of equal length
        """
        max_length = max(len(sublist) for sublist in token_ids)
        pad_token_id = self.token_to_idx[self.pad_token]
        return [batch + [pad_token_id] * (max_length - len(batch)) for batch in token_ids]

    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """
        encode a batch of texts into padded token id sequences.

        args:
            texts: list of input texts to encode

        returns:
            list of padded token id sequences
        """
        return torch.tensor(self.pad([self.encode(text).tolist() for text in texts]))

    def decode(self, token_ids: torch.Tensor) -> str:
        """
        decode a sequence of token ids back into text.

        args:
            token_ids: list of token iss to decode

        returns:
            decoded text string
        """
        return ''.join([self.idx_to_token[i.item()] for i in token_ids])

    def decode_batch(self, batch_tokens: torch.Tensor) -> List[str]:
        """
        decode a batch of token id sequences.

        args:
            batch_tokens - list of token id sequences to decode

        returns:
            list of decoded text strings
        """
        return [self.decode(i) for i in batch_tokens]