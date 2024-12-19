#!/root/miniconda3/bin/python
import os.path
import mindspore as ms
from mindspore.dataset import  ImageFolderDataset
from sklearn.tree._splitter import RandomSplitter
from PIL import Image
import numpy as np

class DataLoader():
    def __init__(self):
        super().__init__()

    def load_train_eval(self):
        data_path='/root/dataset_storage/data/'

        # for filename in os.listdir(f'{data_path}train/'):
        #     i = 0
        #     file_path = os.path.join(data_path, 'train', filename)
        #     for img_path in os.listdir(file_path):
        #         with Image.open(os.path.join(data_path, 'train', filename,img_path)) as img:
        #             print(f'image_processed_{i}')
        #             i=i+1
        #             img_array = np.array(img)
        #             mean_red = np.mean(img_array[:, :, 0])
        #             mean_green = np.mean(img_array[:, :, 1])
        #             mean_blue = np.mean(img_array[:, :, 2])
        #             mean.append([mean_red, mean_green, mean_blue])
        #             std.append([np.std(img_array[:, :, 0]), np.std(img_array[:, :, 1]), np.std(img_array[:, :, 2])])
        #
        # # 计算所有图片的RGB通道均值和标准差的平均值
        # mean = np.array(mean).mean(axis=0)
        # std = np.array(std).mean(axis=0)
        dataset_train = ImageFolderDataset(dataset_dir=os.path.join(data_path, "train"),
                                        class_indexing={"falciparum":0, "uninfected":1,"vivax":2},
                                        extensions=[".tiff", ".jpg"],
                                        shuffle=False)

        total_train = len(dataset_train)
        val_size = int(0.2 * total_train)  # 20%的数据用作验证
        train_size = total_train - val_size

        train_dataset, val_dataset = dataset_train.split([train_size,val_size],randomize=True)#二八分训练和验证
        train_dataset=train_dataset.shuffle(train_dataset.get_dataset_size())
        val_dataset=val_dataset.shuffle(val_dataset.get_dataset_size())
        return train_dataset, val_dataset

    def load_test(self):
        data_path='/root/dataset_storage/data/'
        dataset_test = ImageFolderDataset(dataset_dir=os.path.join(data_path, "test"),class_indexing={"falciparum":0, "uninfected":1,"vivax":2},
                                        extensions=[".tiff", ".jpg"],
                                        shuffle=False)
        test_dataset = dataset_test.shuffle(dataset_test.get_dataset_size())
        return test_dataset

    def load_infer(self):
        data_path='/root/dataset_storage/data/'
        dataset_infer = ImageFolderDataset(dataset_dir=os.path.join(data_path, "infer"),class_indexing={"falciparum":0, "uninfected":1,"vivax":2},
                                        extensions=[".tiff", ".jpg"],
                                        shuffle=False)
        return dataset_infer