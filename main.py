import torch 
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

from settings import use_cuda

from cnn import CNN
from datamanager import get_dataset, get_augmented_dataset

from time import time


def test(data_loader):
	correct = 0.0
	total = 0.0
	for images, labels in data_loader:
		if use_cuda:
			images = images.cuda()
			labels = labels.cuda()
		outputs = cnn(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum()

	accuracy =  100 * correct / total

	print('Test Accuracy of the model on the %d test images: %d %%' % ( total , accuracy))
	
	return accuracy


# Hyper Parameters
num_epochs = 100
batch_size = 100
learning_rate = 1e-5
wt_decay = 1e-7
lr_decay = 0.99


cnn = CNN()


print("Fetching data...")

train_dataset = get_augmented_dataset("./tiny-imagenet-200/train")
test_dataset = get_dataset("./tiny-imagenet-200/val")


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=wt_decay)


# Train the Model
print("Begin training...")

for epoch in range(num_epochs):
	t1 = time()

	for i, (images, labels) in enumerate(train_loader):
		if use_cuda:
			images = images.cuda()
			labels = labels.cuda()
		# Forward + Backward + Optimize
		optimizer.zero_grad()
		outputs = cnn(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		
	t2 = time()
	
	print("Epoch time : %0.3f s" % ((t2-t1) ))

	print("Testing data")
	test(test_loader)

	learning_rate *= lr_decay
	optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=wt_decay)

	# Save the Trained Model
	torch.save(cnn, "cnn.pt")