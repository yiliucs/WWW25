import math
import numpy as np
import torch
import os
import random


class Multimodal_dataset():
    """Build dataset from motion sensor data."""
    def __init__(self, node_id):

        # user_id = node2user[node_id]
        self.folder_path = "/AD-data/node_{}/".format(node_id)

        y = np.load(self.folder_path + "label.npy")

        self.labels = y.tolist() #tolist
        self.labels = torch.tensor(self.labels)
        self.labels = (self.labels).long()


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        x1 = np.load(self.folder_path + "audio/" + "{}.npy".format(idx))
        x2 = np.load(self.folder_path + "depth/" + "{}.npy".format(idx))
        x3 = np.load(self.folder_path + "radar/" + "{}.npy".format(idx))



        self.data1 = x1.tolist() #concate and tolist
        self.data2 = x2.tolist() #concate and tolist
        self.data3 = x3.tolist()

        sensor_data1 = torch.tensor(self.data1) # to tensor
        sensor_data2 = torch.tensor(self.data2) # to tensor
        sensor_data3 = torch.tensor(self.data3) # to tensor

        sensor_data2 = torch.unsqueeze(sensor_data2, 0)

        activity_label = self.labels[idx]

        return sensor_data1, sensor_data2, sensor_data3, activity_label

class Unimodal_dataset():
    """Build dataset from audio data."""

    def __init__(self, node_id, modality_str):

        # user_id = node2user[node_id]
        self.folder_path = "/AD-data/node_{}/".format(node_id)
        self.modality = modality_str
        self.data_path = self.folder_path + "{}/".format(self.modality)

        y = np.load(self.folder_path + "label.npy")

        self.labels = y.tolist()
        self.labels = torch.tensor(self.labels)
        self.labels = (self.labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        x = np.load(self.data_path + "{}.npy".format(idx))

        self.data = x.tolist()
        self.data = torch.tensor(self.data)

        sensor_data = self.data
        if self.modality == 'depth':
            sensor_data = torch.unsqueeze(sensor_data, 0)

        activity_label = self.labels[idx]

        return sensor_data, activity_label


class Multimodal_dataset_train():
    """Build dataset from motion sensor data."""
    def __init__(self, node_id):

        # user_id = node2user[node_id]
        self.folder_path = "/AD-data/"

        y = np.load(self.folder_path + "label_fin_train.npy")
        self.x1 = np.load(self.folder_path + "audio_fin_train.npy")
        self.x2 = np.load(self.folder_path + "depth_fin_train.npy")
        self.x3 = np.load(self.folder_path + "radar_fin_train.npy")

        self.labels = y.tolist() #tolist
        self.labels = torch.tensor(self.labels)
        self.labels = (self.labels).long()


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        x1 = self.x1[idx]
        x2 = self.x2[idx]
        x3 = self.x3[idx]

        x1 = np.squeeze(x1)
        x2 = np.squeeze(x2)
        x3 = np.squeeze(x3)

        self.data1 = x1.tolist() #concate and tolist
        self.data2 = x2.tolist() #concate and tolist
        self.data3 = x3.tolist()

        sensor_data1 = torch.tensor(self.data1) # to tensor
        sensor_data2 = torch.tensor(self.data2) # to tensor
        sensor_data3 = torch.tensor(self.data3) # to tensor

        sensor_data2 = torch.unsqueeze(sensor_data2, 0)

        activity_label = self.labels[idx]

        return sensor_data1, sensor_data2, sensor_data3, activity_label


class Multimodal_dataset_test():
    """Build dataset from motion sensor data."""

    def __init__(self):
        # user_id = node2user[node_id]
        self.folder_path = "/AD-data/"

        y = np.load(self.folder_path + "label_fin_test.npy")
        self.x1 = np.load(self.folder_path + "audio_fin_test.npy")
        self.x2 = np.load(self.folder_path + "depth_fin_test.npy")
        self.x3 = np.load(self.folder_path + "radar_fin_test.npy")

        self.labels = y.tolist()  # tolist
        self.labels = torch.tensor(self.labels)
        self.labels = (self.labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x1 = self.x1[idx]
        x2 = self.x2[idx]
        x3 = self.x3[idx]

        x1 = np.squeeze(x1)
        x2 = np.squeeze(x2)
        x3 = np.squeeze(x3)

        self.data1 = x1.tolist()  # concate and tolist
        self.data2 = x2.tolist()  # concate and tolist
        self.data3 = x3.tolist()

        sensor_data1 = torch.tensor(self.data1)  # to tensor
        sensor_data2 = torch.tensor(self.data2)  # to tensor
        sensor_data3 = torch.tensor(self.data3)  # to tensor

        sensor_data2 = torch.unsqueeze(sensor_data2, 0)

        activity_label = self.labels[idx]

        return sensor_data1, sensor_data2, sensor_data3, activity_label



class Multimodal_imdataset():

    def __init__(self, node_id, accumulation_count=100):

        self.folder_path = "/AD-data/node_{}/".format(node_id)

        self.labels = torch.load(self.folder_path + "label.pt")
        self.labels = self.labels.long()

        self.x1_data = []
        self.y_data = []

        # 记录每个类别的累计数据
        self.accumulated_data = {}

        # 设置最大累计数量
        self.max_accumulation_count = accumulation_count

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        activity_label = self.labels[idx]
        x2 = np.load(self.folder_path + "depth/" + "{}.npy".format(idx))
        x3 = np.load(self.folder_path + "radar/" + "{}.npy".format(idx))

        if idx <= self.max_accumulation_count:

            # 当累计数量未超过最大值时，将 x1 和 y 数据添加到列表中
            x1 = np.load(self.folder_path + "audio/" + "{}.npy".format(idx))
            self.x1_data.append(x1)
            self.y_data.append(activity_label.item())

            # 检查累计数量是否超过最大值，如果是，则清空 x1 和 y 的历史数据，并开始累计新的数据
            if idx == self.max_accumulation_count:
                # self.accumulated_data = {}
                for x1_value, y_value in zip(self.x1_data, self.y_data):
                    if y_value in self.accumulated_data:
                        self.accumulated_data[y_value].append(x1_value)
                    else:
                        self.accumulated_data[y_value] = [x1_value]
                self.x1_data = []
                self.y_data = []

        else:
            # 当累计数量超过最大值时，从已累计的数据中获取 x1 样本
            x1 = self.get_x1_by_y(activity_label.item())

            # 将 x1、x2 和 x3 数据转换为张量
        sensor_data1 = torch.tensor(x1)
        sensor_data2 = torch.tensor(x2)
        sensor_data3 = torch.tensor(x3)

        # 将 x2 和 x3 数据增加一维，以与 x1 数据匹配
        sensor_data2 = torch.unsqueeze(sensor_data2, 0)
        sensor_data3 = torch.unsqueeze(sensor_data3, 0)

        return sensor_data1, sensor_data2, sensor_data3, activity_label

    def get_x1_by_y(self, y):
        # 从已累计的数据中获取 y 对应的 x1 样本
        x1_list = self.accumulated_data.get(y, [])
        if len(x1_list) == 0:
            return None
        else:
            return random.choice(x1_list).tolist()

