import os
import numbers

import torch
import cv2
import mxnet as mx
import numpy as np
from torchvision import transforms



class MXFaceDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, q_func, num_colors):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        if q_func is None:
            self.qfunc = None
        else:
            self.qfunc = lambda img: q_func(img, num_colors)

        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')

        if not os.path.exists(path_imgrec):
            raise FileNotFoundError(f"{os.path.abspath(path_imgrec)}")
        if not os.path.exists(path_imgidx):
            raise FileNotFoundError(f"{os.path.abspath(path_imgidx)}")

        # if file not found does not raise descriptive exception
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.qfunc is not None:
            sample = self.qfunc(sample)
        sample = np.uint8(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)


class FaceDatasetFolder(torch.utils.data.Dataset):
    def __init__(self, root_dir, local_rank):
        super(FaceDatasetFolder, self).__init__()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.root_dir = os.path.join( root_dir)
        self.local_rank = local_rank
        self.imgidx, self.labels=self.scan(self.root_dir)

    def scan(self,root):
        imgidex=[]
        labels=[]
        lb=0
        list_dir=os.listdir(root)
        #list_dir.sort()
        for img in list_dir:
                imgidex.append(os.path.join(root,img))
                labels.append(lb)
                lb = lb+1
        return imgidex, labels
    
    def readImage(self, path):
        return cv2.imread(os.path.join(self.root_dir,path))

    def __getitem__(self, index):
        path = self.imgidx[index]
        img = self.readImage(path)
        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)
        sample = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)
