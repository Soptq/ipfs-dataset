# IPFS Dataset for Pytorch

`ipfs-dataset` is a high-performance PyTorch dataset library designed to provide direct and efficient access to datasets stored in the InterPlanetary File System (IPFS). This library allows interacting with large datasets with streaming data, which eliminates the need to provision local storage capacity.

- [IPFS Dataset for Pytorch](#ipfs-dataset-for-pytorch)
  * [Background](#background)
  * [Features](#features)
  * [Installation](#installation)
  * [Usage](#usage)
    + [Getting started: export and import your Pytorch dataset](#getting-started--export-and-import-your-pytorch-dataset)
    + [Upload your dataset](#upload-your-dataset)
      - [Upload your dataset to IPFS as a `tar` file, a `zip` file, or a plain file](#upload-your-dataset-to-ipfs-as-a--tar--file--a--zip--file--or-a-plain-file)
      - [Upload your dataset to IPFS as a folder](#upload-your-dataset-to-ipfs-as-a-folder)
      - [Upload your dataset with any format](#upload-your-dataset-with-any-format)
    + [Define your IPFS object for retrieval](#define-your-ipfs-object-for-retrieval)
    + [Load the dataset using the IPFS object](#load-the-dataset-using-the-ipfs-object)
      - [Regular Pytorch dataset](#regular-pytorch-dataset)
      - [Iterable Pytorch dataset](#iterable-pytorch-dataset)
    + [Shuffle the dataset](#shuffle-the-dataset)
    + [Iterate over the dataset using Pytorch dataloader](#iterate-over-the-dataset-using-pytorch-dataloader)
  * [Examples](#examples)
  * [License](#license)

## Background

In the current age of big data and machine learning, the need for larger and more complex datasets is increasing. We are now dealing with petabytes or even larger sizes of datasets. These datasets are crucial for training complex machine learning models, which require a vast amount of data to learn effectively. However, managing and accessing these large datasets can be a challenge due to the limitations of local storage capacity.

This is where `ipfs-dataset` comes into play. It leverages the power of IPFS, a protocol and network designed to create a content-addressable, peer-to-peer method of storing and sharing hypermedia in a distributed file system. IPFS allows data to be distributed across a network, reducing the need for centralized data storage and providing resilience against network failures.

The `ipfs-dataset` library provides a seamless interface for PyTorch, a popular open-source machine learning library, to access these distributed datasets. It allows data to be streamed directly from the IPFS network, eliminating the need for local storage and providing efficient access to large datasets.

In essence, `ipfs-dataset` is a powerful tool that addresses the challenges of handling large datasets in the era of big data and machine learning. It offers a solution to the storage limitations and provides efficient and resilient access to large datasets, thereby facilitating the development of more complex and powerful machine learning models.

## Features

1. Fully decentralized: `ipfs-dataset` does **not** (although can) rely on centralized entities.
2. Flexible: `ipfs-dataset` supports accessing datasets stored in IPFS using either IPFS API or IPFS gateway. Accessed datasets can be modified and processed as needed.
3. No local storage: datasets are streamed directly from IPFS, eliminating the need for local storage. **Thus, it can handle datasets that are larger than the available local storage.**
4. Simple to use: `ipfs-dataset` provides a seamless interface for PyTorch, allowing users to access IPFS datasets with minimal effort.
5. Extendable: `ipfs-dataset` can be extended to support custom dataset formats and processing logic.

## Installation

To install `ipfs-dataset`, you can use `pip`:

```bash
pip install ipfs-dataset
```

## Usage

### Getting started: export and import your Pytorch dataset

If you already have a Pytorch dataset, you can export it to IPFS and then import it using `ipfs-dataset` easily. The following example shows how to export a Pytorch dataset to IPFS and then import it using `ipfs-dataset`.

```python
from ipfs_dataset.exporter import IPFSDatasetExporter
from torchvision.datasets import CIFAR10


ipfs_api_address = "/ip4/127.0.0.1/tcp/5001"
dataset_from_pytorch = CIFAR10(root="data", train=True, download=True)

# export the dataset to IPFS
cid = IPFSDatasetExporter(ipfs_api_address).export_to_ipfs(dataset_from_pytorch)

# import the dataset from IPFS
dataset_from_ipfs = IPFSDatasetExporter(ipfs_api_address).import_from_ipfs(f"ipfs://{cid}")
```

The dataset will be sharded, packaged and uploaded to IPFS automatically to allow the best IO performance.

### Upload your dataset

#### Upload your dataset to IPFS as a `tar` file, a `zip` file, or a plain file

Simply manage your dataset as you would normally do. When the dataset is ready, you can compress it into a `tar` or `zip` file. Then, you can upload the compressed file to IPFS using the `ipfs` command-line tool or any other IPFS client.

```bash
> tar -cvf dataset.tar dataset/
> ipfs add dataset.tar
```

If you are dealing with a single file containing NLP data or any other type of data, you can upload it to IPFS directly as a plain file.

```bash
> ipfs add dataset.txt
```

#### Upload your dataset to IPFS as a folder

If you have a folder containing your dataset (like the `ImageFolder` in Pytorch), and you don't want to compress it into a `tar` or `zip` file, you can upload the entire folder to IPFS using the `ipfs` command-line tool or any other IPFS client.

Before uploading, create a `metadata.txt` file in the root of your dataset directory containing all relative filepaths to the data files. This file will be used by `ipfs-dataset` to read the data from IPFS. For example:

```bash
> tree -d .
|-- truck_lorry_s_000528.png
|-- dog_mutt_s_000396.png
|-- cat_felis_domesticus_s_000016.png
|-- another_dir
|   |-- automobile_convertible_s_001372.png
|-- metadata.txt

> cat metadata.txt
truck_lorry_s_000528.png
dog_mutt_s_000396.png
cat_felis_domesticus_s_000016.png
another_dir/automobile_convertible_s_001372.png
```

#### Upload your dataset with any format

If your dataset has a custom format, you can upload it to IPFS as a plain file. Then, you can use the `ipfs-dataset` library to read the byte data of the dataset from IPFS and process it as you need.

```bash
> ipfs add dataset.mat # for example, a Matlab file
```

### Define your IPFS object for retrieval

You can define your dataset for retrieval by creating a list of `IPFSObject` objects. Each `IPFSObject` object represents a file or a directory in IPFS. You need to provide a URL to access the object and the type of the object.

The URL of the object should be one of the following:

1. a CID, e.g., `ipfs://QmdPw1vze5nwVM1fsCjygXCs8WyzDsf3hn4uq3MpMLvw9w`
2. a gateway URL, e.g., `https://gateway.pinata.cloud/ipfs/QmdPw1vze5nwVM1fsCjygXCs8WyzDsf3hn4uq3MpMLvw9w`


```python
from ipfs_dataset.types import RemoteFileType, IPFSObject

# the following is a public Cifar10 training dataset stored in IPFS, compressed as a zip file
train_ipfs_objs = [
    IPFSObject("ipfs://QmdPw1vze5nwVM1fsCjygXCs8WyzDsf3hn4uq3MpMLvw9w", RemoteFileType.ZIP)
]
# the following is a public Cifar10 testing dataset stored in IPFS, compressed as a zip file
test_ipfs_objs = [
    IPFSObject("https://gateway.pinata.cloud/ipfs/QmUXbPZvz4NuM3StmxLJdSjoGWTJLxF2iTe3Afd4uLhPa1", RemoteFileType.ZIP)
]
```

You can also add multiple `IPFSObject` objects to the list if your dataset is stored in multiple files or directories.

```python
from ipfs_dataset.types import RemoteFileType, IPFSObject

# the following is a public Cifar10 dataset stored in IPFS, compressed as a zip file
ipfs_objs = [
    IPFSObject("ipfs://QmdPw1vze5nwVM1fsCjygXCs8WyzDsf3hn4uq3MpMLvw9ww", RemoteFileType.ZIP),
    IPFSObject("https://gateway.pinata.cloud/ipfs/QmUXbPZvz4NuM3StmxLJdSjoGWTJLxF2iTe3Afd4uLhPa1", RemoteFileType.ZIP)
]
```

You can also provide a folder as a `IPFSObject` object if your dataset is stored in a folder in IPFS.

```python
from ipfs_dataset.types import RemoteFileType, IPFSObject

ipfs_objs = [
    IPFSObject("ipfs://QmdeFFimPStrc4JZ1Lk1AiDCdRF14DSwvwutY3Pk3ivLru", RemoteFileType.FOLDER),
]
```

### Load the dataset using the IPFS object
#### Regular Pytorch dataset

A regular Pytorch dataset can be shuffled by Pytorch dataloader and can be used with PyTorch distributed sampler. It supports any pre-processing and augmentation.

```python
from ipfs_dataset.types import RemoteFileType, IPFSObject
from ipfs_dataset.ipfs_dataset import IPFSDataset
from PIL import Image
import io

from torchvision import transforms


class IPFSCifar10Dataset(IPFSDataset):
    def __init__(self, ipfs_objs, ipfs_api_address, transform=None):
        super().__init__(ipfs_objs, ipfs_api_address)
        self.transform = transform
        
    def __getitem__(self, idx):
        file_name, file_obj = super(IPFSDataset, self).__getitem__(idx)
        # Convert bytes object to image
        img = Image.open(io.BytesIO(file_obj)).convert('RGB')
        
        # Apply preprocessing functions on data
        if self.transform is not None:
            img = self.transform(img)
        return img, file_name


# # If you provide gateway url, then you can set ipfs_api_address to None.
# # However, it is considered less decentralized.
# ipfs_api_address = None
ipfs_api_address = "/ip4/127.0.0.1/tcp/5001"
ipfs_objs = [
    IPFSObject("ipfs://QmdPw1vze5nwVM1fsCjygXCs8WyzDsf3hn4uq3MpMLvw9w", RemoteFileType.ZIP),
    IPFSObject("ipfs://QmUXbPZvz4NuM3StmxLJdSjoGWTJLxF2iTe3Afd4uLhPa1", RemoteFileType.ZIP)
]

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

ipfs_cifar10_dataset = IPFSCifar10Dataset(ipfs_objs, ipfs_api_address, transform=transform)
```

#### Iterable Pytorch dataset

An iterable Pytorch dataset means IPFS objects are accessed in a sequential manner. Such form of datasets is particularly useful when data come from a stream. For the specific case of zip/tar archival files, each file contained in the archival is returned during each iteration in a streaming fashion. For all other file formats, binary blob for the whole shard is returned and users need to implement the appropriate parsing logic.

> For datasets consisting of a large number of smaller objects, accessing each object individually can be inefficient. For such datasets, it is recommended to create shards of the training data for better performance.

```python
from ipfs_dataset.types import RemoteFileType, IPFSObject
from ipfs_dataset.ipfs_dataset import IPFSIterableDataset
from PIL import Image
import io

from torchvision import transforms

class IPFSCifar10Dataset(IPFSIterableDataset):
    def __init__(self, ipfs_objs_list, ipfs_api_address, shuffle_urls=False, transform=None):
        self.ipfs_dataset = IPFSIterableDataset(ipfs_objs_list, ipfs_api_address, shuffle_urls)
        self.transform = transform

    def data_generator(self):
        try:
            while True:
                fname, fobj = next(self.ipfs_dataset_iter)
                image_np = Image.open(io.BytesIO(fobj)).convert('RGB')

                # Apply torch visioin transforms if provided
                if self.transform is not None:
                    image_np = self.transform(image_np)
                yield image_np, fname
        except StopIteration:
            return

    def __iter__(self):
        self.ipfs_dataset_iter = iter(self.ipfs_dataset)
        return self.data_generator()


# # If you provide gateway url, then you can set ipfs_api_address to None.
# # However, it is considered less decentralized.
# ipfs_api_address = None
ipfs_api_address = "/ip4/127.0.0.1/tcp/5001"
ipfs_objs = [
    IPFSObject("ipfs://QmdPw1vze5nwVM1fsCjygXCs8WyzDsf3hn4uq3MpMLvw9w", RemoteFileType.ZIP),
    IPFSObject("ipfs://QmUXbPZvz4NuM3StmxLJdSjoGWTJLxF2iTe3Afd4uLhPa1", RemoteFileType.ZIP)
]

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

ipfs_cifar10_dataset = IPFSCifar10Dataset(ipfs_objs, ipfs_api_address, transform=transform)
```

### Shuffle the dataset

When you shuffle the above two kinds of IPFS dataset, it will only shuffle the provided IPFS objects, not the data inside the IPFS objects. If you want to shuffle the data inside the IPFS objects, you can use the `ShuffleDataset`.

```python
from ipfs_dataset.ipfs_dataset import ShuffleDataset


...
shuffle_ipfs_cifar10_dataset = ShuffleDataset(ipfs_cifar10_dataset, buffer_size=1000)
...
```

### Iterate over the dataset using Pytorch dataloader

Load your IPFS dataset as you would normally do with Pytorch datasets.

```python
from torch.utils.data import DataLoader

...
dataloader = DataLoader(ipfs_cifar10_dataset, batch_size=32)
...
```

## Examples

In `examples` directory, you can find examples of how to use `ipfs-dataset` with Pytorch to train a LeNet model on the Cifar10 dataset stored in IPFS (using either IPFS gateway or IPFS API).

## License

```
MIT License

Copyright (c) 2024 Soptq

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```