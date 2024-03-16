import io
from PIL import Image
from ipfs_dataset.ipfs_dataset import IPFSIterableDataset
from ipfs_dataset.types import RemoteFileType, IPFSObject

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from lenet import LeNet

LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
BATCH_SIZE = 64
LR = 1e-2
EPOCH = 1


class Cifar10IPFS(IPFSIterableDataset):
    def __init__(self, ipfs_objs_list, ipfs_api_address, shuffle_urls=False, transform=None):
        self.ipfs_dataset = IPFSIterableDataset(ipfs_objs_list, ipfs_api_address, shuffle_urls)
        self.transform = transform

    def data_generator(self):
        try:
            while True:
                fname, fobj = next(self.ipfs_dataset_iter)

                if fname.startswith("__MACOSX"):
                    continue
                fname = fname.split("/")[-1]
                if fname is None:
                    continue
                if not fname.endswith(".png"):
                    continue

                label = LABELS.index(fname.split("_")[0])
                image_np = Image.open(io.BytesIO(fobj)).convert('RGB')

                # Apply torch visioin transforms if provided
                if self.transform is not None:
                    image_np = self.transform(image_np)
                yield image_np, label
        except StopIteration:
            return

    def __iter__(self):
        self.ipfs_dataset_iter = iter(self.ipfs_dataset)
        return self.data_generator()


ipfs_api_address = "/ip4/127.0.0.1/tcp/5001"
train_ipfs_objs = [
    IPFSObject("QmdPw1vze5nwVM1fsCjygXCs8WyzDsf3hn4uq3MpMLvw9w", RemoteFileType.ZIP)
]
test_ipfs_objs = [
    IPFSObject("ipfs://QmUXbPZvz4NuM3StmxLJdSjoGWTJLxF2iTe3Afd4uLhPa1", RemoteFileType.ZIP)
]

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = Cifar10IPFS(train_ipfs_objs, transform=transform_train)
test_dataset = Cifar10IPFS(test_ipfs_objs, ipfs_api_address, transform=transform_test)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

net = LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

for epoch in range(EPOCH):
    net.train()
    total_loss = 0.
    for i, (inputs, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch {epoch}, Batch {i}, Loss: {total_loss / 100}")
            total_loss = 0.

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch}, Accuracy: {100 * correct / total}%")
