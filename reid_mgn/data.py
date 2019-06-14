from torchvision import transforms
from torch.utils.data import dataset, dataloader
from torchvision.datasets.folder import default_loader
from utils.RandomErasing import RandomErasing
from utils.RandomSampler import RandomSampler
import os
import re


class Data():
    def __init__(self,
                 rand_erasing=True,
                 dataset='Market1501',
                 data_path='./datasets/market_1501',
                 batchid=16,
                 batchimage=4,
                 batchtest=16,
                 num_workers=0,
                 query_image=None,
                 mode='train',
                 input_height=384,
                 input_width=128,
                 **kwargs):
        train_transforms = [
            transforms.Resize((input_height, input_width), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if rand_erasing:
            train_transforms.append(RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0]))
        train_transform = transforms.Compose(train_transforms)

        test_transform = transforms.Compose([
            transforms.Resize((input_height, input_width), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if dataset == 'Market1501':
            self.trainset = Market1501(train_transform, 'train', data_path)
            self.testset = Market1501(test_transform, 'test', data_path)
            self.queryset = Market1501(test_transform, 'query', data_path)
        elif dataset == 'DeepFashion':
            self.trainset = DeepFashion(train_transform, 'train', data_path)
            self.testset = DeepFashion(test_transform, 'test', data_path)
            self.queryset = DeepFashion(test_transform, 'query', data_path)
        else:
            raise Exception('Dataset not implemented: %s' % dataset)

        self.train_loader = dataloader.DataLoader(self.trainset,
                                                  sampler=RandomSampler(self.trainset,
                                                                        batch_id=batchid,
                                                                        batch_image=batchimage),
                                                  batch_size=batchid * batchimage,
                                                  num_workers=num_workers,
                                                  pin_memory=True)
        self.test_loader = dataloader.DataLoader(self.testset, 
                                                 batch_size=batchtest,
                                                 num_workers=num_workers,
                                                 pin_memory=True)
        self.query_loader = dataloader.DataLoader(self.queryset, 
                                                  batch_size=batchtest,
                                                  num_workers=num_workers,
                                                  pin_memory=True)

        if mode == 'vis':
            self.query_image = test_transform(default_loader(query_image))


class Market1501(dataset.Dataset):
    def __init__(self, transform, dtype, data_path):

        self.transform = transform
        self.loader = default_loader
        self.data_path = data_path

        if dtype == 'train':
            self.data_path += '/train'
        elif dtype == 'test':
            self.data_path += '/test'
        else:
            self.data_path += '/query'

        self.imgs = [path for path in self.list_pictures(self.data_path) if self.id(path) != -1]

        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self._id2label[self.id(path)]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(file_path.split('/')[-1].split('_')[0])

    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        """
        return int(file_path.split('/')[-1].split('_')[1][1])

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))

    @property
    def cameras(self):
        """
        :return: camera id list corresponding to dataset image paths
        """
        return [self.camera(path) for path in self.imgs]

    @staticmethod
    def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm|npy'):
        assert os.path.isdir(directory), 'dataset is not exists!{}'.format(directory)

        return sorted([os.path.join(root, f)
                       for root, _, files in os.walk(directory) for f in files
                       if re.match(r'([\w]+\.(?:' + ext + '))', f)])



class DeepFashion(dataset.Dataset):
    def __init__(self, transform, dtype, data_path):

        self.transform = transform
        self.loader = default_loader
        self.data_path = data_path

        if dtype == 'train':
            self.data_path += '/train'
        elif dtype == 'test':
            self.data_path += '/test'
        else:
            self.data_path += '/test'

        self.imgs = [path for path in self.list_pictures(self.data_path) if self.id(path) != -1]

        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self._id2label[self.id(path)]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        filename = file_path.split('/')[-1]
        sku = ''.join(filename.split('_')[:-1])
        id = sku.split('id')[1]
        return int(id)

    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        """
        return int(file_path.split('/')[-1].split('_')[-1][0])

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))

    @property
    def cameras(self):
        """
        :return: camera id list corresponding to dataset image paths
        """
        return [self.camera(path) for path in self.imgs]

    @staticmethod
    def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm|npy'):
        assert os.path.isdir(directory), 'dataset is not exists!{}'.format(directory)

        return sorted([os.path.join(root, f)
                       for root, _, files in os.walk(directory) for f in files
                       if re.match(r'([\w]+\.(?:' + ext + '))', f)])
