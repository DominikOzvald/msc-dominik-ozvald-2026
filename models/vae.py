import torch
import math


class CharVaeEnc(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, latent_size: int):
        super().__init__()
        self.latent_size = latent_size
        self.rnn = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc_mean = torch.nn.Linear(in_features=hidden_size, out_features=latent_size)
        self.fc_log_var = torch.nn.Linear(in_features=hidden_size, out_features=latent_size)

    def forward(self, x, lengths):
        rnn_in = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False, batch_first=True)
        output, _ = self.rnn(rnn_in)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        mean = self.fc_mean(output)
        log_var = self.fc_log_var(output)
        eps = 0.0 * torch.rand_like(mean).to(self.fc_mean.weight.device)
        z = mean + torch.exp(0.5 * log_var) * eps
        return z, mean[:lengths.min()], log_var[:lengths.min()]


class CharVaeDec(torch.nn.Module):
    def __init__(self, latent_size: int, hidden_size: int, vocab_size: int):
        super().__init__()
        self.rnn = torch.nn.LSTM(input_size=latent_size, hidden_size=hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, z, lengths):
        rnn_in = torch.nn.utils.rnn.pack_padded_sequence(z, lengths, enforce_sorted=False, batch_first=True)
        output, _ = self.rnn(rnn_in)
        h, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        logist = self.fc(h)
        return logist


class CharVae(torch.nn.Module):
    def __init__(self, matrix: torch.nn.Embedding, embedding_size: int, latent_size: int, hidden_size: int):
        super().__init__()
        self.matrix = matrix
        self.enc = CharVaeEnc(embedding_size, hidden_size, latent_size)
        self.dec = CharVaeDec(latent_size, hidden_size, self.matrix.weight.size(0))

    def forward(self, in_text, lengths):
        x = self.matrix(in_text)
        z, mean, log_var = self.enc(x, lengths)
        logist = self.dec(z, lengths)
        return logist, mean, log_var

    def get_z(self, in_text, lengths):
        x = self.matrix(in_text)
        z, mean, log_var = self.enc(x, lengths)
        return z


class LineVaeEnc(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, latent_size: int):
        super().__init__()
        self.latent_size = latent_size
        self.rnn = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc_mean = torch.nn.Linear(in_features=hidden_size, out_features=latent_size)
        self.fc_log_var = torch.nn.Linear(in_features=hidden_size, out_features=latent_size)

    def forward(self, x, lengths):
        rnn_in = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False, batch_first=True)
        output, (hn, cn) = self.rnn(rnn_in)
        mean = self.fc_mean(hn)
        log_var = self.fc_log_var(hn)
        eps = 0.01 * torch.rand_like(mean).to(self.fc_mean.weight.device)
        z = mean + torch.exp(0.5 * log_var) * eps
        return z.squeeze(0).unsqueeze(1), mean, log_var


class LineVaeDec(torch.nn.Module):
    def __init__(self, latent_size: int, hidden_size: int, vocab_size: int):
        super().__init__()
        self.rnn = torch.nn.LSTM(input_size=2 + latent_size, hidden_size=hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, z: torch.Tensor, lengths: torch.Tensor, targets: torch.Tensor):
        z_expanded = z.repeat(1, lengths.max().item(), 1)
        targets_z = torch.cat([targets, z_expanded], dim=-1)
        rnn_in = torch.nn.utils.rnn.pack_padded_sequence(targets_z, lengths.cpu(), enforce_sorted=False,
                                                         batch_first=True)
        h, _ = self.rnn(rnn_in)
        h_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        logits = self.fc(h_padded)

        return logits


class LineVae(torch.nn.Module):
    def __init__(self, matrix: torch.nn.Embedding, embedding_size: int, latent_size: int, hidden_size: int):
        super().__init__()
        self.enc_matrix = matrix
        self.dec_matrix = torch.nn.Embedding(matrix.num_embeddings, embedding_dim=2)
        self.enc = LineVaeEnc(embedding_size, hidden_size, latent_size)
        self.dec = LineVaeDec(latent_size, hidden_size, self.enc_matrix.weight.size(0))

    def forward(self, in_text, lengths):
        x = self.enc_matrix(in_text) * math.sqrt(self.enc_matrix.embedding_dim)
        z, mean, log_var = self.enc(x, lengths)
        sos = 2 * torch.ones((in_text.size(0), 1), dtype=torch.long).to(self.enc_matrix.weight.device)
        targets = self.dec_matrix(torch.cat([sos, in_text], dim=-1)[:, :-1]) * math.sqrt(
            self.enc_matrix.embedding_dim)
        recon = self.dec(z, lengths, targets)
        return recon, mean, log_var

    def get_z(self, in_text, lengths):
        x = self.enc_matrix(in_text)
        z, mean, log_var = self.enc(x, lengths)
        return z


class ConvLSTMEnc(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, latent_size: int, vocab_size: int = None,
                 max_in_len: int = 256,
                 letter_chunk: int = 8):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.max_in_len = max_in_len

        if self.vocab_size:
            self.matrix = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=input_size)
        else:
            self.matrix = None
        self.conv_step = input_size * letter_chunk
        self.conv_size = max_in_len // letter_chunk

        self.conv = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=self.conv_step, stride=self.conv_step)
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(in_features=hidden_size, out_features=self.latent_size - self.conv_size)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):

        if self.matrix:
            x = self.matrix(x)
        elif len(x.shape) == 2:
            x = x.unsqueeze(-1)
        rnn_in = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, (hn, cn) = self.lstm(rnn_in)
        fc_out = self.fc(hn).squeeze(0).unsqueeze(1)
        conv_in = x.reshape(x.size(0), 1, -1)
        conv_out = self.conv(conv_in)
        out = torch.cat([conv_out, fc_out], dim=-1)

        return out


class ConvLSTMDec(torch.nn.Module):
    def __init__(self, latent_size: int, hidden_size: int, vocab_size: int, max_in_len: int = 256):
        super().__init__()

        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_in_len = max_in_len

        self.lstm = torch.nn.LSTM(input_size=latent_size, hidden_size=hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, z: torch.Tensor, lengths: torch.Tensor):
        z_expanded = z.repeat(1, self.max_in_len, 1)
        rnn_in = torch.nn.utils.rnn.pack_padded_sequence(z_expanded, lengths.cpu(), batch_first=True,
                                                         enforce_sorted=False)
        h, _ = self.lstm(rnn_in)
        fc_in, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=self.max_in_len)
        out = self.fc(fc_in)
        return out


class ConvLSTM(torch.nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, latent_size: int, vocab_size: int,
                 use_embed_matrix: bool = False, max_in_len: int = 256,
                 letter_chunk: int = 8):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.use_embed_matrix = use_embed_matrix

        if use_embed_matrix:
            enc_vocab = vocab_size
        else:
            enc_vocab = None
        self.enc = ConvLSTMEnc(input_size=embed_size, hidden_size=hidden_size, latent_size=latent_size,
                               vocab_size=enc_vocab, max_in_len=max_in_len, letter_chunk=letter_chunk)
        self.dec = ConvLSTMDec(latent_size=latent_size, hidden_size=hidden_size, vocab_size=vocab_size,
                               max_in_len=max_in_len)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        z = self.enc(x, lengths)
        out = self.dec(z, lengths)
        return out

    def get_z(self,x:torch.Tensor,lengths:torch.Tensor):
        z = self.enc(x,lengths)
        return z

class RnnVaeEnc(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, latent_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.rnn1 = torch.nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size)
        self.fc1 = torch.nn.Linear(in_features=self.hidden_size, out_features=2 * self.latent_size)

    def forward(self, x):
        h, _ = self.rnn1(x)
        mean, log_var = torch.split(self.fc1(h), self.latent_size, dim=2)

        eps = torch.normal(0, 1, size=mean.shape).to(self.fc1.weight.device)
        z = mean + torch.exp(0.5 * log_var) * eps

        return z, mean, log_var


class RnnVaeDec(torch.nn.Module):
    def __init__(self, embedding_size: int, latent_size: int, hidden_size: int, vocab_size: int):
        super().__init__()

        self.embedding_size = embedding_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.rnn1 = torch.nn.GRU(input_size=latent_size + embedding_size, hidden_size=hidden_size)
        self.fc = torch.nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, z, targets):
        rnn_input = torch.cat([targets, z], dim=2)
        h, _ = self.rnn1(rnn_input)
        recon = self.fc(h)
        return recon


class RnnVae(torch.nn.Module):

    def __init__(self, matrix, input_size: int, hidden_size: int, latent_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.matrix = matrix
        self.enc = RnnVaeEnc(input_size=self.input_size, hidden_size=self.hidden_size, latent_size=self.latent_size)
        self.dec = RnnVaeDec(embedding_size=self.input_size, latent_size=self.latent_size, hidden_size=self.hidden_size,
                             vocab_size=self.matrix.weight.shape[0])

    def forward(self, in_log):
        x = self.matrix(in_log)
        z, mean, log_var = self.enc(x)
        sos = self.matrix(2 * torch.ones(in_log[0].shape, dtype=torch.long).to(self.enc.fc1.weight.device)).unsqueeze(0)
        targets = torch.cat([sos, x], dim=0)[:-1]
        recon = self.dec(z, targets)

        return recon, mean, log_var
