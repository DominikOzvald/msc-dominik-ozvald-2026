from datasets import LogCharDataSet,pad_len_collate_fn
from embeddings import create_embedding_matrix
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from CharVAE import CharVae
from torch.optim import Adam
from TrainUtils import vae_train_loop_lengths
from os import path
from torch import save

if __name__ == "__main__":
    data_folder = "../train_data"
    save_folder = "../trained_models"
    image_folder = "../train_images"
    embedding_dim = 256
    batch_size = 64
    hidden_size = 400
    latent_size = 128
    lr = 1e-4
    epochs = 200
    print_every = 20
    model_name = f"CHAR_VAE_I_{embedding_dim}_H_{hidden_size}_L_{latent_size}"
    beta = 0.01
    # ----------------------------------------------------------------------------

    data_set = LogCharDataSet(data_folder)
    matrix = create_embedding_matrix(data_set.vocab,embedding_dim)
    data_loader = DataLoader(data_set,batch_size=batch_size,shuffle=True,collate_fn=pad_len_collate_fn)
    # ----------------------------------------------------------------------------

    _, axs = plt.subplots(1, 2)
    axs[0].imshow(matrix.weight.clone().detach().cpu().numpy()[:50, :50])
    axs[0].set_title("Embedding matrix before training")

    # ----------------------------------------------------------------------------

    model = CharVae(matrix,embedding_size=embedding_dim,hidden_size=hidden_size,latent_size=latent_size)
    optimizer = Adam(model.parameters(),lr=lr)
    loss = vae_train_loop_lengths(model,data_loader,optimizer,epochs=epochs,show_every_n=print_every,beta=beta)
    # ----------------------------------------------------------------------------
    axs[1].imshow(matrix.weight.clone().detach().cpu().numpy()[:50,:50])
    axs[1].set_title("Embedding matrix after training")

    plt.savefig(path.join(image_folder, model_name + '_embedding.png'))
    plt.clf()

    plt.plot(range(50, len(loss)), loss[50:], )
    plt.title(model_name + " loss")
    plt.savefig(path.join(image_folder, model_name + "_loss.png"))

    save(model.state_dict(), path.join(save_folder, model_name + '.pt'))