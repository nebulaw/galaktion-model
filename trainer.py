import torch

from config import Config, ModelConfig
from data_loader import DataLoader
from nn import BigramModel, DecoderModel
from tokenizer import CharacterTokenizer


class Trainer:
    def __init__(self, tokenizer_dir, input_data_dir, model_dir, model='bigram'):
        self.tokenizer = CharacterTokenizer.load(tokenizer_dir)
        self.input_data_dir = input_data_dir
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = Config
        self.dataloader = self.load_data()
        pad_token_idx = self.tokenizer.token_to_idx[self.tokenizer.pad_token]
        if model == 'bigram':
            self.model = BigramModel(self.tokenizer.num_tokens, pad_token_idx).to(self.device)
        if model == 'decoder':
            self.model = DecoderModel(self.tokenizer.num_tokens,
                                      ModelConfig.d_model,
                                      ModelConfig.head_dim,
                                      ModelConfig.block_size,
                                      ModelConfig.n_head,
                                      ModelConfig.ffn_dim,
                                      ModelConfig.layers,
                                      ModelConfig.dropout,
                                      pad_token_idx).to(self.device)

    def load_data(self):
        with open(self.input_data_dir) as f:
            data = ''.join(f.readlines())
        data = data.split('\n')
        data = [i for i in data if i.strip().isdigit() == False]
        train_data = data[:int(len(data) * 0.9)]
        valid_data = data[int(len(data) * 0.9):int(len(data) * 0.95)]
        test_data = data[int(len(data) * 0.95):]

        train_data = '\n'.join(train_data)
        valid_data = '\n'.join(valid_data)
        test_data = '\n'.join(test_data)
        train_data = self.tokenizer.encode(train_data, add_special_tokens=False)
        valid_data = self.tokenizer.encode(valid_data, add_special_tokens=False)
        test_data = self.tokenizer.encode(test_data, add_special_tokens=False)

        dataLoader = DataLoader(train_data, valid_data, test_data, ModelConfig.block_size, self.tokenizer)
        return dataLoader

    @torch.no_grad()
    def estimate_val(self):
        out=dict()
        self.model.eval()
        for split in ['train', 'valid']:
            losses = torch.zeros(self.config.valid_steps)
            for step in range(self.config.valid_steps):
                x, y = self.dataloader.get_batch('train', self.config.valid_batch_size)
                x = x.to(self.device)
                y = y.to(self.device)
                logits, loss = self.model(x, y)
                losses[step] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out


    def run_training(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.train_steps)

        lossi=[]
        val_lossi=[]
        print("Starts Training!")
        for step in range(self.config.train_steps):
            x, y = self.dataloader.get_batch('train', self.config.train_batch_size)
            x=x.to(self.device)
            y=y.to(self.device)
            logits, loss = self.model(x, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % self.config.logging_steps == 0:
                out=self.estimate_val()
                val_lossi.append(out['valid'])
                lossi.append(out['train'])
                print(f"Step: {step}/{self.config.train_steps}, train loss: {out['train']:.4f}, valid loss: {out['valid']:.4f}")

        out = self.estimate_val()
        val_lossi.append(out['valid'])
        lossi.append(out['train'])
        print(f"Finally, train loss: {out['train']:.4f}, valid loss: {out['valid']:.4f}")
        torch.save(self.model.state_dict(), self.model_dir)
        return lossi, val_lossi, self.model