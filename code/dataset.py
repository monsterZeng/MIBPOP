
from dgllife.data import MoleculeCSVDataset
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer, mol_to_graph,PretrainAtomFeaturizer
from dgllife.utils import AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
from dgllife.model import GCNPredictor
from dgllife.model.gnn import GCN, GAT
from dgllife.model.readout import WeightedSumAndMax
import dgl
import torch
from torch.utils.data import DataLoader


def smile_to_graph(smile, node_featurizer, edge_featurizer):
    graph = smiles_to_bigraph(smile,
                             
                             node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h'),
                             edge_featurizer = CanonicalBondFeaturizer(bond_data_field='e')
    )
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    return graph

def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.

    Parameters
    ----------
    data : list of 3-tuples or 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and optionally a binary
        mask indicating the existence of labels.

    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels.
    """
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if len(data[0]) == 3:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks

def get_dataloader(df, batch_size, shuffle = True, collate_fn = collate_molgraphs, get_dataset = False):
    dataset = MoleculeCSVDataset(df = df, 
                                 smiles_to_graph=smile_to_graph,
                                 node_featurizer = None,
                                 edge_featurizer = None,
                                 smiles_column='Smiles', 
                                 cache_file_path="./degradation_valid.bin")
    if not get_dataset:
        return DataLoader(dataset = dataset, batch_size = batch_size, shuffle = shuffle, collate_fn = collate_fn)
    else:
        return dataset, DataLoader(dataset = dataset, batch_size = batch_size, shuffle = shuffle, collate_fn = collate_fn)