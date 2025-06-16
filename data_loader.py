import torch
from tqdm import tqdm
from tokenizer import CharacterTokenizer


class DataLoader:
    def __init__(self, train_input_ids, valid_input_ids, test_input_ids, block_size, tokenizer: CharacterTokenizer):
        # store dataset splits as tokenized sequences
        self.train_input_ids = train_input_ids
        self.valid_input_ids = valid_input_ids
        self.test_input_ids = test_input_ids

        # block size determines sequence length for training
        self.block_size = block_size
        self.tokenizer = tokenizer

        # precompute valid starting indices for each dataset to ensure clean text boundaries
        self.valid_starts = {
            'train': self._get_valid_starts(self.train_input_ids),
            'valid': self._get_valid_starts(self.valid_input_ids),
            'test': self._get_valid_starts(self.test_input_ids),
        }

    def _get_valid_starts(self, data):
        # find indices that start after a space or newline to avoid splitting words
        valid_indices = [
            i + 1 for i in range(len(data) - self.block_size)
            if data[i].item() in {self.tokenizer.token_to_idx[' '], self.tokenizer.token_to_idx['\n']}
        ]
        return valid_indices

    def get_batch(self, split: str, batch_size: int):
        # get random batch of sequences from specified dataset split
        valid_indices = self.valid_starts[split]

        # randomly sample starting positions
        ix = torch.tensor(valid_indices)[torch.randint(len(valid_indices), (batch_size,))]

        # create input sequences (x) with start token prepended
        x = torch.stack([
            torch.cat((
                torch.tensor([self.tokenizer.token_to_idx[self.tokenizer.start_token]]),
                self.train_input_ids[i:i + self.block_size - 1]  # note: uses train_input_ids regardless of split
            )) for i in ix
        ])

        # create target sequences (y) - same as input but without start token
        y = torch.stack([self.train_input_ids[i: i + self.block_size] for i in
                         ix])  # note: uses train_input_ids regardless of split

        return x, y

    def iterator(self, split: str, batch_size: int):
        # iterate through entire dataset split in batches (for evaluation)
        valid_indices = self.valid_starts[split]

        # process data in chunks with progress bar
        for j in tqdm(range(0, len(valid_indices), batch_size)):
            batch_indices = valid_indices[j:j + batch_size]

            # create input and target sequences (standard language modeling setup)
            x = torch.stack([self.train_input_ids[i:i + self.block_size] for i in
                             batch_indices])  # note: uses train_input_ids regardless of split
            y = torch.stack([self.train_input_ids[i + 1:i + self.block_size + 1] for i in
                             batch_indices])  # targets are shifted by 1

            yield x, y