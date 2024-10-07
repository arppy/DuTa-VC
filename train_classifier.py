# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import os
import numpy as np
from tqdm import tqdm

import torch
from model.vc import FwdDiffusion, FwdDiffusionWithDurationPredictor
from torch.utils.data import DataLoader

import params
from data import ATYDecDataset, ATYDecBatchCollate
from model.vc import DiffVC

n_mels = params.n_mels
sampling_rate = params.sampling_rate
n_fft = params.n_fft
hop_size = params.hop_size

channels = params.channels
filters = params.filters
layers = params.layers
kernel = params.kernel
dropout = params.dropout
heads = params.heads
window_size = params.window_size
enc_dim = params.enc_dim

dec_dim = params.dec_dim
spk_dim = params.spk_dim
use_ref_t = params.use_ref_t
beta_min = params.beta_min
beta_max = params.beta_max

random_seed = params.seed
test_size = params.test_size

dim = params.enc_dim
device = torch.device('cuda:' + str(params.gpu))
num_of_phoneme_class = params.num_of_phoneme_class
filters_dp = params.filters_dp

data_dir = '../data/UASpeech/'
log_dir = 'logs_dec_aty'
encoder_path = 'logs_enc/enc.pt'
allspks = ['F02','F03','F04','F05','CM01','CM02','M10','M11','M12','M14','M16']

epochs = 40
batch_size = 32
learning_rate = 5e-5
save_every = 1


def main(dys):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    log_dir_dys = os.path.join(log_dir, dys)
    os.makedirs(log_dir_dys, exist_ok=True)

    print('Initializing data loaders...')
    train_set = ATYDecDataset(data_dir, dys)
    collate_fn = ATYDecBatchCollate()
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              collate_fn=collate_fn, num_workers=16, drop_last=True)
    print(len(train_set))
    print('Initializing and loading models...')
    pretrained_model = FwdDiffusionWithDurationPredictor(n_mels, channels, filters, heads, layers, kernel, dropout, window_size,
                                              dim, filters_dp, num_of_phoneme_class).to(device)
    pretrained_model = torch.load("./logs_enc/enc.pt", map_location=device)

    print('Initializing optimizers...')
    optimizer = torch.optim.Adam(params=model.decoder.parameters(), lr=learning_rate)

    print('Start training.')
    torch.backends.cudnn.benchmark = True
    iteration = 0
    for epoch in range(1, epochs + 1):
        print(f'Epoch: {epoch} [iteration: {iteration}]')
        model.train()
        losses = []
        for batch in tqdm(train_loader, total=len(train_set) // batch_size):
            mel, mel_ref = batch['mel1'].cuda(), batch['mel2'].cuda()
            c, mel_lengths = batch['c'].cuda(), batch['mel_lengths'].cuda()
            model.zero_grad()
            loss = model.compute_loss(mel, mel_lengths, mel_ref, c)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1)
            optimizer.step()
            losses.append(loss.item())
            iteration += 1

        losses = np.asarray(losses)
        msg = 'Epoch %d: loss = %.4f\n' % (epoch, np.mean(losses))
        print(msg)
        with open(f'{log_dir_dys}/train_dec.log', 'a') as f:
            f.write(msg)

        if epoch % save_every > 0:
            continue

        print('Saving model...\n')
        ckpt = model.state_dict()
        torch.save(ckpt, f=f"{log_dir_dys}/vc.pt")

if __name__ == "__main__":
    for spk in allspks:
        main(spk)