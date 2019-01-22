import torch 
import torchvision
import torchvision.transforms as transforms


def get_dataset(file):
	
	data_transform = transforms.Compose([
			transforms.ToTensor()
		])

	dataset = torchvision.datasets.ImageFolder(file, data_transform)

	return dataset


def get_augmented_dataset(file):
	bright = 0.6
	cont = 0.6
	satu = 0.6
	hu = 0.25

	data_transform = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.ColorJitter(brightness=bright, contrast=cont, saturation=satu, hue=hu),
			transforms.RandomCrop(size=64, padding=4),
			transforms.ToTensor()
		])

	dataset = torchvision.datasets.ImageFolder(file, data_transform)

	return dataset