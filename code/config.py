from argparse import Namespace
import torch as th
import numpy as np
import torch.nn.functional as F
args = Namespace(
    raw_data_path = "./dataset/ci4000213_si_001.xlsx", # replace your raw dataset path
    train_sheet_name = "Train+Test",
    valid_sheet_name = "External validation",
    device = 'cuda' if th.cuda.is_available() else 'cpu',
    # -----------------------model hyperparameters------------------------
    model = 'GIN',
    train_batch_size = 128,
    valid_batch_size = 670,
    learning_rate = 0.005,
    in_feats =  74,
    hidden_feat = 14,
    num_layers = 8,
    n_classes = 2,
    hidden_feats = [128] * 4, 
    activations  = [F.selu] * 4,
    residuals    = [True] * 4,
    batchnorms   = [True] * 4, 
    dropouts = [0.2] * 4,
    n_epoch = 100,
    # -------------------specail hyperparameters-------------------------
    num_heads = [4] * 4,
    feat_drops = [0.2] * 4,
    attn_drops = [0.2] * 4,
    agg_modes = ['flatten'] * 4,
    seed = 2021
)

# def set_seed_everywhere(seed, cuda):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     if cuda:
#         torch.cuda.manual_seed_all(seed)
# set_seed_everywhere(args.seed, torch.cuda.is_available())
