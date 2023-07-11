import argparse
import math
import os
import time
import uuid

import datasets
import numpy as np
from datasets import load_from_disk
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm
from datetime import datetime
import torch
import torchtext
import torch.distributed as dist
import torch.multiprocessing as mp

import tools


class TransformerModel(torch.nn.Module):

    def __init__(self, vocab_size, emb_size, nhead, nhid, nlayers, dropout=0.5, maxlen=5000):
        super().__init__()
        self.model_type = 'Transformer'

        tmp0 = np.exp(-np.arange(0, emb_size, 2) * np.log(10000) / emb_size)
        tmp1 = np.arange(maxlen)[:, np.newaxis] * tmp0
        pos_embedding = np.stack([np.sin(tmp1), np.cos(tmp1)], axis=2).reshape(maxlen, 1, emb_size)
        self.register_buffer('pos_embedding', torch.tensor(pos_embedding, dtype=torch.float32))
        self.dropout = torch.nn.Dropout(dropout)

        encoder_layers = torch.nn.TransformerEncoderLayer(emb_size, nhead, nhid, dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        self.src_tok_emb = torch.nn.Embedding(vocab_size, emb_size)
        self.decoder = torch.nn.Linear(emb_size, vocab_size)
        self.emb_factor = np.sqrt(emb_size)

        self.init_weights()

    def init_weights(self):
        initrange = 1 / np.sqrt(self.src_tok_emb.weight.shape[1])
        self.src_tok_emb.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        N0 = src.shape[0]
        device = self.decoder.weight.device
        tmp0 = torch.triu(torch.ones((N0, N0), dtype=torch.bool, device=device), 1)
        src_mask = torch.zeros((N0, N0), dtype=torch.float32, device=device)
        src_mask.masked_fill_(tmp0, float('-inf'))

        src = self.dropout(self.src_tok_emb(src) * self.emb_factor + self.pos_embedding[:src.shape[0]])
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-c', '--cpus', default=1, type=int,
                        help='number of cpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--tracedir', default="/mnt/", type=str, )
    parser.add_argument('--sfdir', type=str, )

    args = parser.parse_args()

    os.sched_setaffinity(0, [i for i in range(args.cpus)])
    os.environ['GOMP_CPU_AFFINITY'] = "0-" + str(args.cpus - 1)
    args.world_size = args.gpus * args.nodes

    for _ in range(2):
        start = datetime.now()
        args.sharefile = str(uuid.uuid1())
        args.start = start
        mp.spawn(worker, nprocs=args.gpus, args=(args,), daemon=True)
        time.sleep(10)


def worker(gpu, args):
    def print_with_rank(msg):
        print('[RANK {}]: {}'.format(rank, msg))

    os.sched_setaffinity(0, [i for i in range(args.cpus)])
    os.environ['GOMP_CPU_AFFINITY'] = "0-" + str(args.cpus - 1)
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl',
                            init_method='file://' + args.sfdir + args.sharefile, world_size=args.world_size,
                            rank=rank)
    torch.manual_seed(0)

    os.environ['HF_DATASETS_OFFLINE'] = '1'
    dataset = load_from_disk("./datasets")
    train_iter = dataset['train']
    val_iter = dataset['validation']
    print("data ready")
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    tmp0 = (tokenizer(x['text']) for x in train_iter)
    vocab = torchtext.vocab.build_vocab_from_iterator(tmp0, specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    print("vocab ready")

    parameter = {
        'vocab_size': len(vocab),
        'emb_size': 512,
        'nhead': 8,
        'nhid': 512,
        'nlayers': 12,
        'dropout': 0.2,
    }
    model = TransformerModel(**parameter).cuda(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    criterion = torch.nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001*args.gpus)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    def data_process(raw_text_iter):
        data = [torch.tensor(vocab(tokenizer(item['text'])), dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    train_data = data_process(train_iter)
    val_data = data_process(val_iter)

    device = torch.device('cuda:{}'.format(gpu))

    def batchify(data, bsz, rank, world_size, is_train=False):
        # Divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        # Divide the data across the ranks only for training data.
        if is_train:
            data_per_rank = data.size(0) // world_size
            data = data[rank * data_per_rank: (rank + 1) * data_per_rank]
        return data.to(device)

    batch_size = 256
    train_data = batchify(train_data, batch_size, rank, args.world_size, True)
    val_data = batchify(val_data, batch_size, rank, args.world_size)

    bptt = 64

    def get_batch(source, i):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].view(-1)
        # Need batch dimension first for pipeline parallelism.
        return data.t(), target

    def train():
        start_time = time.time()
        ntokens = len(vocab)
        nbatches = min(50 * bptt, train_data.size(0) - 1)
        log_interval = 10
        total_loss = 0.
        model.train()

        for batch, i in enumerate(range(0, nbatches, bptt)):
            data, targets = get_batch(train_data, i)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.view(-1, ntokens), targets.cuda(gpu))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print_with_rank('| epoch {:3d} | {:5d}/{:5d} batches | '
                                'lr {:02.2f} | ms/batch {:5.2f} | '
                                'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, nbatches // bptt, scheduler.get_last_lr()[0],
                                  elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

    def evaluate(eval_model, data_source):
        eval_model.eval()  # Turn on the evaluation mode
        total_loss = 0.
        ntokens = len(vocab)
        # Evaluate only for 50 batches to keep script execution time low.
        nbatches = min(50 * bptt, data_source.size(0) - 1)
        with torch.no_grad():
            for i in range(0, nbatches, bptt):
                data, targets = get_batch(data_source, i)
                output = eval_model(data)
                output_flat = output.view(-1, ntokens)
                # Need to move targets to the device where the output of the
                # pipeline resides.
                total_loss += len(data) * criterion(output_flat, targets.cuda(gpu)).item()
        return total_loss / (len(data_source) - 1)

    best_val_loss = float("inf")
    epochs = 15  # The number of epochs
    best_model = None

    logs = []
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(model, val_data)
        print_with_rank('-' * 89)
        print_with_rank('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                   val_loss, math.exp(val_loss)))
        print_with_rank('-' * 89)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        logs.append(((datetime.now() - args.start).total_seconds(), val_loss))
        if val_loss <= 44.47:
            break
        scheduler.step()

    if gpu == 0:
        tools.append_list_as_row(args.tracedir + "trace.csv",
                                 ["lm", args.cpus, args.gpus, (datetime.now() - args.start).total_seconds()] + logs)


if __name__ == '__main__':
    main()
