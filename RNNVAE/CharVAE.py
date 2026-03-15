import torch


class CharVaeEnc(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, latent_size: int):
        super().__init__()
        self.rnn = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        self.fc_mean = torch.nn.Linear(in_features=hidden_size, out_features=latent_size)
        self.fc_log_var = torch.nn.Linear(in_features=hidden_size, out_features=latent_size)

    def forward(self, x, lengths):
        rnn_in = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)
        output, (hn, cn) = self.rnn(rnn_in)
        output,_ = torch.nn.utils.rnn.pad_packed_sequence(output)
        mean = self.fc_mean(output)
        log_var = self.fc_log_var(output)
        eps = torch.rand_like(mean).to(self.fc_mean.weight.device)
        z = mean + torch.exp(0.5 * log_var) * eps
        return z,mean[:lengths.min()],log_var[:lengths.min()]


class CharVaeDec(torch.nn.Module):
    def __init__(self, embedding_size: int, latent_size: int, hidden_size: int, vocab_size: int):
        super().__init__()
        self.rnn = torch.nn.LSTM(input_size=latent_size, hidden_size=hidden_size)
        self.fc = torch.nn.Linear(in_features=hidden_size, out_features=vocab_size)
    def forward(self,z,lengths):
        # rnn_in = torch.cat([targets,z],dim=2)
        rnn_in = torch.nn.utils.rnn.pack_padded_sequence(z,lengths,enforce_sorted=False)
        output, (hn, cn) = self.rnn(rnn_in)
        h,_ = torch.nn.utils.rnn.pad_packed_sequence(output)
        recon = self.fc(h)
        return recon

class CharVae(torch.nn.Module):
    def __init__(self,matrix : torch.nn.Embedding, embedding_size: int, latent_size: int, hidden_size: int):
        super().__init__()
        self.matrix = matrix
        self.enc = CharVaeEnc(embedding_size,hidden_size,latent_size)
        self.dec = CharVaeDec(embedding_size,latent_size,hidden_size,self.matrix.weight.size(0))

    def forward(self,in_text,lengths):
        x = self.matrix(in_text)
        z,mean,log_var = self.enc(x,lengths)
        recon = self.dec(z,lengths)
        return recon,mean,log_var




if __name__ == "__main__":
    input_text = [torch.ones(4), torch.ones(6)]
    padded_in = torch.nn.utils.rnn.pad_sequence(input_text, batch_first=True, padding_value=0).to(torch.long)
    padded_in = torch.transpose(padded_in,dim0=0,dim1=1)
    lengths = torch.tensor([4, 6])
    matrix = torch.nn.Embedding(3, 3, padding_idx=0)
    enc_in = matrix(padded_in)
    enc = CharVaeEnc(3, 5, 4)
    dec = CharVaeDec(3,4,5,3)
    z = enc(enc_in, lengths)
    sos = matrix(2*torch.ones_like(padded_in[0]).to(torch.long)).unsqueeze(0)
    targets = torch.cat([sos,enc_in],dim=0)[:-1]

    rec = dec(z,lengths)
    print(rec)

