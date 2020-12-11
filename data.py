import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

class Dataset(TensorDataset):
    'Characterize a dataset for PyTorch'
    def __init__(self, data, mask):
        'Initialization'
        self.data=data
        self.mask=mask
        self.n=data.shape[0]
        self.d=data.shape[1]

    def __len__(self):
        'Denotes the total number of samples'
        return self.n

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        out_data = self.data[index]
        out_mask = self.mask[index]

        return out_data, out_mask

def make_dataloader(data=None, mask=None, batchsize=100, cuda=False, shuffle=False):
    """Create a DataLoader for input of each data type
    """

    data = torch.from_numpy(data)
    mask = torch.from_numpy(mask)
    dataset = Dataset(data=data, mask=mask)

    # Create dataloader
    dataloader = DataLoader(dataset=dataset, batch_size=batchsize, drop_last=False,
                            shuffle=shuffle, num_workers=1, pin_memory=cuda)
    return dataloader
