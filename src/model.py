"""
LSTM Seq2Seq with Bahdanau (additive) attention -- implemented from scratch
using only nn.LSTM / nn.Embedding / nn.Linear building blocks, per the
assignment's requirement (no Fairseq / OpenNMT / HF Seq2SeqTrainer).

Architecture (matches course slides, Week 11 - RNN Encoder-Decoder + Attention):
  Embedding -> Bidirectional LSTM Encoder -> Bahdanau Attention -> LSTM Decoder -> Output projection

References:
  Bahdanau et al., 2015, "Neural Machine Translation by Jointly Learning to
  Align and Translate" (course slides, Week 11).
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import PAD_IDX, SOS_IDX


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hidden_dim=512, num_layers=1, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Project concatenated [forward; backward] final states down to hidden_dim
        # for the (unidirectional) decoder's initial state.
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src, src_lens):
        # src: [batch, src_len]
        embedded = self.dropout(self.embedding(src))  # [batch, src_len, emb_dim]

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lens.cpu(), batch_first=True, enforce_sorted=True
        )
        packed_outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # outputs: [batch, src_len, hidden_dim*2]  (concat of fwd/bwd per layer's top)

        # hidden/cell: [num_layers*2, batch, hidden_dim] -> take the last layer's
        # forward and backward states and combine.
        hidden_fwd = hidden[-2, :, :]
        hidden_bwd = hidden[-1, :, :]
        cell_fwd = cell[-2, :, :]
        cell_bwd = cell[-1, :, :]

        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=1)  # [batch, hidden_dim*2]
        cell_cat = torch.cat([cell_fwd, cell_bwd], dim=1)

        dec_hidden = torch.tanh(self.fc_hidden(hidden_cat)).unsqueeze(0)  # [1, batch, hidden_dim]
        dec_cell = torch.tanh(self.fc_cell(cell_cat)).unsqueeze(0)

        return outputs, dec_hidden, dec_cell


class Attention(nn.Module):
    """Bahdanau additive attention: e_ij = v^T tanh(W1 h_j + W2 s_{i-1})."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask):
        # decoder_hidden: [batch, hidden_dim]
        # encoder_outputs: [batch, src_len, hidden_dim*2]
        # mask: [batch, src_len] (1 for real tokens, 0 for padding)
        src_len = encoder_outputs.size(1)
        hidden_rep = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch, src_len, hidden_dim]

        energy = torch.tanh(self.attn(torch.cat([hidden_rep, encoder_outputs], dim=2)))
        scores = self.v(energy).squeeze(2)  # [batch, src_len]

        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=1)  # [batch, src_len]

        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [batch, 1, hidden_dim*2]
        return context.squeeze(1), attn_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hidden_dim=512, num_layers=1, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(
            emb_dim + hidden_dim * 2, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.fc_out = nn.Linear(emb_dim + hidden_dim * 2 + hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell, encoder_outputs, mask):
        # input_token: [batch] (previous target token id)
        input_token = input_token.unsqueeze(1)  # [batch, 1]
        embedded = self.dropout(self.embedding(input_token))  # [batch, 1, emb_dim]

        # attention uses the top-layer decoder hidden state from the previous step
        context, attn_weights = self.attention(hidden[-1], encoder_outputs, mask)  # [batch, hidden_dim*2]

        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)  # [batch, 1, emb_dim+hidden_dim*2]
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # output: [batch, 1, hidden_dim]

        output = output.squeeze(1)
        embedded = embedded.squeeze(1)
        prediction = self.fc_out(torch.cat([output, context, embedded], dim=1))  # [batch, vocab_size]

        return prediction, hidden, cell, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    @staticmethod
    def make_src_mask(src):
        return (src != PAD_IDX).to(torch.float32)  # [batch, src_len]

    def forward(self, src, src_lens, tgt=None, max_len=50, teacher_forcing_ratio=0.5):
        """
        Training mode: pass `tgt` [batch, tgt_len] -> uses teacher forcing.
        Inference mode: leave `tgt` as None -> greedy decode up to `max_len`.
        Returns logits: [batch, out_len, vocab_size]
        """
        batch_size = src.size(0)
        encoder_outputs, hidden, cell = self.encoder(src, src_lens)
        mask = self.make_src_mask(src)

        vocab_size = self.decoder.fc_out.out_features
        out_len = tgt.size(1) if tgt is not None else max_len

        outputs = torch.zeros(batch_size, out_len, vocab_size, device=self.device)
        input_token = torch.full((batch_size,), SOS_IDX, dtype=torch.long, device=self.device)

        for t in range(1, out_len):
            prediction, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs, mask)
            outputs[:, t, :] = prediction

            teacher_force = (tgt is not None) and (random.random() < teacher_forcing_ratio)
            top1 = prediction.argmax(1)
            if teacher_force:
                input_token = tgt[:, t]
            else:
                input_token = top1

        return outputs

    @torch.no_grad()
    def greedy_decode(self, src, src_lens, max_len=50):
        """Inference-time greedy decoding. Returns token id sequences (lists) per example."""
        self.eval()
        batch_size = src.size(0)
        encoder_outputs, hidden, cell = self.encoder(src, src_lens)
        mask = self.make_src_mask(src)

        input_token = torch.full((batch_size,), SOS_IDX, dtype=torch.long, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        sequences = [[] for _ in range(batch_size)]

        from .vocab import EOS_IDX

        for _ in range(max_len):
            prediction, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs, mask)
            top1 = prediction.argmax(1)
            for i in range(batch_size):
                if not finished[i]:
                    sequences[i].append(top1[i].item())
                    if top1[i].item() == EOS_IDX:
                        finished[i] = True
            input_token = top1
            if finished.all():
                break
        return sequences


def build_model(src_vocab_size, tgt_vocab_size, config, device):
    # NOTE: the encoder->decoder initial-state bridge (fc_hidden/fc_cell) currently
    # produces a single-layer initial state, so the decoder must use num_layers=1.
    # (The encoder itself may still be a multi-layer / bidirectional LSTM.)
    assert config["decoder_layers"] == 1, (
        "Decoder must have num_layers=1 with the current encoder->decoder "
        "state bridge. Extend Encoder.fc_hidden/fc_cell to support >1 if needed."
    )
    encoder = Encoder(
        src_vocab_size,
        emb_dim=config["emb_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["encoder_layers"],
        dropout=config["dropout"],
    )
    decoder = Decoder(
        tgt_vocab_size,
        emb_dim=config["emb_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["decoder_layers"],
        dropout=config["dropout"],
    )
    model = Seq2Seq(encoder, decoder, device).to(device)
    return model
