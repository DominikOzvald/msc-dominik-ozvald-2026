import torch.nn
from os import path
from utils.datasets import CharVocab, TransformerDataset
from models.vae import LineVae
from models.embedder import Embedder
from models.vaetransformer import DecoderTransformer
from torch.utils.data import DataLoader
from utils.data import separate_last_log
import matplotlib.pyplot as plt
import torch.nn.functional as F

if __name__ == "__main__":
    data_folders = ["../test_data/order", "../test_data/rand"]
    save_folder = "../trained_models"
    image_folder = "../test_images"

    vae_embedding = 32
    vae_hidden = 196
    vae_latent = 64

    vae_name = f"LINE_VAE_I_{vae_embedding}_H_{vae_hidden}_L_{vae_latent}"
    char_vocab = CharVocab()
    vae = LineVae(torch.nn.Embedding(len(char_vocab), vae_embedding), embedding_size=vae_embedding,
                  latent_size=vae_latent, hidden_size=vae_hidden)

    try:
        vae.load_state_dict(torch.load(path.join(save_folder, vae_name + '.pt'), weights_only=True))
    except:
        print("Can not load VAE:", vae_name)
        exit(-1)
    embedder = Embedder(vae.enc_matrix, vae.enc)

    dec_enc_layer = 2
    n_head = 2
    dim_forward = 1024
    transformer_name = f"DecTransformer_D_{dec_enc_layer}_H_{n_head}_F_{dim_forward}"
    d_model = 64
    transformer = DecoderTransformer(d_model=d_model, n_head=n_head,
                                     enc_dec_layer=dec_enc_layer, dim_forward=dim_forward)
    try:
        transformer.load_state_dict(torch.load(path.join(save_folder, transformer_name + ".pt"), weights_only=True))
    except:
        print("Can not load Transformer:", transformer_name)
        exit(-1)

    step_size = 1
    frame_size = 20
    max_len = 200
    n_steps = 5
    for k, folder in enumerate(data_folders):
        data_set = TransformerDataset(folder, step=step_size, frame_size=frame_size, max_len=max_len)
        data_loader = DataLoader(data_set, batch_size=1, shuffle=False)
        losses = []
        with torch.no_grad():
            for i, (data, lengths, masks) in enumerate(data_loader):
                if i > 2000:
                    z = embedder(data, lengths)
                    z, target, masks = separate_last_log(z, masks,n_steps=n_steps)
                    out = transformer(z, masks,n_steps=n_steps)
                    loss = F.mse_loss(out.reshape(-1,out.size(-1)), target.reshape(-1,target.size(-1)))
                    losses.append(loss.item())
                    print(f"Line {i * step_size + frame_size - n_steps}-{i * step_size + frame_size}: {loss.item()}")
                if i > 2145:
                    break

        print(f"Average loss: {sum(losses) / len(losses)}")
        plt.plot(range(len(losses)), losses)
        plt.grid()
        plt.title(folder)
        plt.savefig(path.join(image_folder, f"{transformer_name}_{k}.png"))
        plt.clf()