import torch.utils.data
import torch
from data.base_data_loader import BaseDataLoader


def CreateDataset(dataroot,
                  phase,
                  pairLst,
                  dataset_mode='keypoint',
                  use_flip=False,
                  resize_or_crop='no',
                  loadSize=286,
                  fineSize=256,
                  **kwargs
                  ):
    dataset = None

    if dataset_mode == 'keypoint':
        from data.keypoint import KeyDataset
        dataset = KeyDataset()

    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(dataroot,
                       phase,
                       pairLst,
                       use_flip,
                       resize_or_crop,
                       loadSize,
                       fineSize)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, dataroot, phase, pairLst, batchSize, serial_batches, nThreads, max_dataset_size=float("inf"),
                   dataset_mode='keypoint',
                   use_flip=False, resize_or_crop='no',
                   loadSize=286, fineSize=256, **kwargs):
        self.max_dataset_size = max_dataset_size
        self.dataset = CreateDataset(dataroot, phase, pairLst, dataset_mode, use_flip, resize_or_crop, loadSize,
                                     fineSize)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=batchSize,
                                                      shuffle=not serial_batches,
                                                      num_workers=int(nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.max_dataset_size:
                break
            yield data

class GDataLoader(CustomDatasetDataLoader):

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.max_dataset_size:
                break
            BP = torch.cat((data['BP1'], data['BP2']),1)
            yield {
                'g_data': [data['P1'], BP],
                'target': data['P2'],
                'from': data['P1_path'],
                'to': data['P2_path']
            }
