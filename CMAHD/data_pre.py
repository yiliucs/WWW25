import numpy as np
import torch
import random


random.seed(42)

class Multimodal_dataset():
	"""Build dataset from motion sensor data."""
	def __init__(self, x1, x2, y):

		# self.data1 = x1.tolist() #concate and tolist
		# self.data2 = x2.tolist() #concate and tolist
		# self.labels = y.tolist() #tolist

		self.data1 = torch.tensor(x1) # to tensor
		self.data2 = torch.tensor(x2) # to tensor
		self.labels = torch.tensor(y)
		self.labels = (self.labels).long()


	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):

		sensor_data1 = self.data1[idx]
		# sensor_data1 = torch.unsqueeze(sensor_data1, 0)

		sensor_data2 = self.data2[idx]
		# sensor_data2 = torch.unsqueeze(sensor_data2, 0)

		activity_label = self.labels[idx]

		return sensor_data1, sensor_data2, activity_label