"""
Speech Recognition with Wav2Vec2
================================
**Author**: `Moto Hira <moto@fb.com>`__
This tutorial shows how to perform speech recognition using using
pre-trained models from wav2vec 2.0
[`paper <https://arxiv.org/abs/2006.11477>`__].
"""


import argparse
import time
from datetime import datetime
import os
import sys
import tools
import requests
import torch
import torchaudio

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

    for _ in range(2):
        start = datetime.now()
        inference()
        tools.append_list_as_row(args.tracedir + "trace.csv",
                                 ["audio", args.cpus, args.gpus, (datetime.now() - start).total_seconds()])
        time.sleep(5)

def inference():
    torch.random.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

    model = bundle.get_model(dl_kwargs={"model_dir": "./model"}).to(device)

    cmu_arctic = torchaudio.datasets.CMUARCTIC('./', download=True)
    data_loader = torch.utils.data.DataLoader(cmu_arctic,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=1)
    with torch.inference_mode():
        decoder = GreedyCTCDecoder(labels=bundle.get_labels())
        for i, (waveform, sample_rate, transcript, utterance_id) in enumerate(data_loader):
            waveform = waveform.to(device)
            waveform = waveform.squeeze(0)
            if sample_rate != bundle.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
            features, _ = model.extract_features(waveform)
            emission, _ = model(waveform)
            for e in emission:
                transcript = decoder(e)
            print(transcript)


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.
        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

if __name__ == '__main__':
    main()
