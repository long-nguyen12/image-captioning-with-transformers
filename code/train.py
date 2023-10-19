from pathlib import Path
import pickle

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Compose
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

from models.poolformer import ImageEncoder
from models.IC_encoder_decoder.transformer import Transformer

from dataset.dataloader import HDF5Dataset, collate_padd
from torchtext.vocab import Vocab

from trainer import Trainer
from utils.train_utils import parse_arguments, seed_everything, load_json
from utils.gpu_cuda_helper import select_device
from data.flickr30k import Flickr30kDataset


def get_datasets(dataset_dir: str, pid_pad: float):
    DATASET_BASE_PATH = 'data/flickr30k/'

    train_dataset = Flickr30kDataset(dataset_base_path=DATASET_BASE_PATH, dist='train', device=device,
                                     return_type='tensor',
                                     load_img_to_memory=False)

    val_dataset = Flickr30kDataset(dataset_base_path=DATASET_BASE_PATH, dist='val', device=device,
                                   return_type='corpus',
                                   load_img_to_memory=False)

    test_dataset = Flickr30kDataset(dataset_base_path=DATASET_BASE_PATH, dist='test', device=device,
                                    return_type='corpus',
                                    load_img_to_memory=False)

    vocab, word2idx, idx2word, max_len = vocab_set = train_dataset.get_vocab()
    with open('vocab_set.pkl', 'wb') as f:
        pickle.dump(train_dataset.get_vocab(), f)

    vocab_size = len(vocab)

    # images transfrom
    train_transformations = transforms.Compose([
        transforms.Resize(256),  # smaller edge of image resized to 256
        transforms.RandomCrop(256),  # get 256x256 crop from random location
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))
    ])
    eval_transformations = transforms.Compose([
        transforms.Resize(256),  # smaller edge of image resized to 256
        transforms.CenterCrop(256),  # get 256x256 crop from random location
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))
    ])

    train_dataset.transformations = train_transformations
    val_dataset.transformations = eval_transformations

    return train_dataset, val_dataset, vocab, vocab_size


if __name__ == "__main__":

    # parse command arguments
    args = parse_arguments()
    dataset_dir = args.dataset_dir  # mscoco hdf5 and json files
    resume = args.resume
    if resume == "":
        resume = None

    # device
    device = select_device(args.device)
    print(f"selected device is {device}.\n")

    # load confuguration file
    config = load_json(args.config_path)

    train_dataset, val_dataset, vocab, vocab_size = get_datasets()

    # load vocab
    min_freq = config["min_freq"]

    # SEED
    SEED = config["seed"]
    seed_everything(SEED)
    BATCH_SIZE = 32

    # --------------- dataloader --------------- #
    print("loading dataset...")
    g = torch.Generator()
    g.manual_seed(SEED)
    loader_params = config["dataloader_parms"]
    max_len = config["max_len"]
    eval_collate_fn = lambda batch: (torch.stack([x[0] for x in batch]), [x[1] for x in batch], [x[2] for x in batch])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, sampler=None, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, sampler=None, pin_memory=False,
                        collate_fn=eval_collate_fn)
    print("loading dataset finished.")
    print(f"number of vocabualry is {vocab_size}\n")

    # --------------- Construct models, optimizers --------------- #
    print("constructing models")
    # prepare some hyperparameters
    image_enc_hyperparms = config["hyperparams"]["image_encoder"]
    image_seq_len = int(image_enc_hyperparms["encode_size"]**2)

    transformer_hyperparms = config["hyperparams"]["transformer"]
    transformer_hyperparms["vocab_size"] = vocab_size
    transformer_hyperparms["pad_id"] = "<pad>"
    transformer_hyperparms["img_encode_size"] = image_seq_len
    transformer_hyperparms["max_len"] = max_len - 1

    # construct models
    image_enc = ImageEncoder(**image_enc_hyperparms)
    image_enc.fine_tune(True)
    transformer = Transformer(**transformer_hyperparms)

    # load pretrained embeddings
    print("loading pretrained glove embeddings...")
    weights = vocab.vectors
    transformer.decoder.cptn_emb.from_pretrained(weights,
                                                 freeze=True,
                                                 padding_idx="<pad>")
    list(transformer.decoder.cptn_emb.parameters())[0].requires_grad = False

    # Optimizers and schedulers
    image_enc_lr = config["optim_params"]["encoder_lr"]
    parms2update = filter(lambda p: p.requires_grad, image_enc.parameters())
    image_encoder_optim = Adam(params=parms2update, lr=image_enc_lr)
    gamma = config["optim_params"]["lr_factors"][0]
    image_scheduler = StepLR(image_encoder_optim, step_size=1, gamma=gamma)

    transformer_lr = config["optim_params"]["transformer_lr"]
    parms2update = filter(lambda p: p.requires_grad, transformer.parameters())
    transformer_optim = Adam(params=parms2update, lr=transformer_lr)
    gamma = config["optim_params"]["lr_factors"][1]
    transformer_scheduler = StepLR(transformer_optim, step_size=1, gamma=gamma)

    # --------------- Training --------------- #
    print("start training...\n")
    train = Trainer(optims=[image_encoder_optim, transformer_optim],
                    schedulers=[image_scheduler, transformer_scheduler],
                    device=device,
                    pad_id="<pad>",
                    resume=resume,
                    checkpoints_path=config["pathes"]["checkpoint"],
                    **config["train_parms"])
    train.run(image_enc, transformer, [train_loader, val_loader], SEED)

    print("done")
