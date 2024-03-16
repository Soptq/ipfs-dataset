import math

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, Dataset
import requests

import random
from itertools import chain

from .utils import tardata, zipdata, ResponseStream
from .types import RemoteFileType, IPFSObject

import ipfshttpclient


class IPFSBaseClass(object):
    """A base class for defining urls_list for IPFSDataset and IPFSIterableDataset
    """

    def __init__(self, ipfs_objs_list, ipfs_api_address):
        ipfs_objs = [ipfs_objs_list] if isinstance(ipfs_objs_list, IPFSObject) else ipfs_objs_list
        if ipfs_api_address is not None:
            self.ipfs_client = ipfshttpclient.connect(ipfs_api_address)
        else:
            self.ipfs_client = None
        self._ipfs_objs_list = self.create_ipfs_objs_list(ipfs_objs)

    def create_ipfs_objs_list(self, ipfs_objs):
        ipfs_objs_list = list()
        for ipfs_obj in ipfs_objs:
            if ipfs_obj.filetype == RemoteFileType.FOLDER:
                ipfs_objects = self.list_files(ipfs_obj.url)
                assert len(ipfs_objects) != 0, \
                    f"The directory {ipfs_obj.url} does not contain any objects."
                ipfs_objs_list.extend(ipfs_objects)
            elif ipfs_objs_list:
                ipfs_objs_list.append(ipfs_obj)
            else:
                ipfs_objs_list = [ipfs_obj]
        return ipfs_objs_list

    @property
    def ipfs_objs_list(self):
        return self._ipfs_objs_list

    def list_files(self, url):
        if url.startswith("ipfs://"):
            if self.ipfs_client is None:
                raise ValueError("IPFS client is not initialized")

            cid = url[7:]
            linked_ipfs_objs = self.ipfs_client.ls(cid).as_json()
            for link in linked_ipfs_objs['Objects'][0]['Links']:
                if link['Name'] == "metadata.txt":
                    retrieval = self.ipfs_client.cat(link["Hash"]).decode("utf-8")

                    files = list()
                    for filepath in retrieval.split("\n"):
                        filepath = filepath.strip()
                        if filepath == "":
                            continue
                        if filepath[-3:] == "tar":
                            files.append(IPFSObject(f"ipfs://{cid}/{filepath}", RemoteFileType.TAR))
                        elif filepath[-3:] == "zip":
                            files.append(IPFSObject(f"ipfs://{cid}/{filepath}", RemoteFileType.ZIP))
                        else:
                            files.append(IPFSObject(f"ipfs://{cid}/{filepath}", RemoteFileType.PLAIN))
                    return files
        elif url.startswith("https://") or url.startswith("http://"):
            if url.endswith("/"):
                url = url[:-1]
            resp = requests.get(f"{url}/metadata.txt")
            if resp.status_code != 200:
                raise FileExistsError(f"{url}/metadata.txt does not exist")
            retrieval = resp.text

            files = list()
            for filepath in retrieval.split("\n"):
                filepath = filepath.strip()
                if filepath == "":
                    continue
                if filepath[-3:] == "tar":
                    files.append(IPFSObject(f"{url}/{filepath}", RemoteFileType.TAR))
                elif filepath[-3:] == "zip":
                    files.append(IPFSObject(f"{url}/{filepath}", RemoteFileType.ZIP))
                else:
                    files.append(IPFSObject(f"{url}/{filepath}", RemoteFileType.PLAIN))
            return files
        else:
            raise ValueError(f"Invalid url: {url}")

    def file_read(self, url) -> ResponseStream:
        if url.startswith("ipfs://"):
            if self.ipfs_client is None:
                raise ValueError("IPFS client is not initialized")

            cid = url[7:]
            return ResponseStream(self.ipfs_client.cat(cid, stream=True, timeout=math.inf))
        elif url.startswith("https://") or url.startswith("http://"):
            resp = requests.get(url, stream=True)
            if resp.status_code != 200:
                raise FileExistsError(f"{url} does not exist")
            return ResponseStream(resp.iter_content(chunk_size=4096))
        else:
            raise ValueError(f"Invalid url: {url}")


class IPFSDataset(IPFSBaseClass, Dataset):
    """A mapped-style dataset for objects in IPFS.
    """
    def __init__(self, ipfs_objs_list, ipfs_api_address=None):
        IPFSBaseClass.__init__(self, ipfs_objs_list, ipfs_api_address)
        # Initialize the handler in the worker since we want each worker to have
        # it's own handler
        self.handler = None

    def __len__(self):
        return len(self.ipfs_objs_list)

    def __getitem__(self, idx):
        ipfs_obj = self.ipfs_objs_list[idx]
        fileobj = self.file_read(ipfs_obj.url).read()
        return ipfs_obj.url, fileobj


class IPFSIterableDataset(IPFSBaseClass, IterableDataset):
    """Iterate over IPFS dataset.
    It handles some bookkeeping related to DataLoader.
    """

    def __init__(self, ipfs_objs_list, ipfs_api_address=None, shuffle=False):
        self.epoch = 0
        self.shuffle = shuffle
        self.dist = dist.is_initialized() if dist.is_available() else False
        if self.dist:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        IPFSBaseClass.__init__(self, ipfs_objs_list, ipfs_api_address)

    @property
    def shuffled_list(self):
        if self.shuffle:
            random.seed(self.epoch)
            return random.sample(self.ipfs_objs_list, len(self.ipfs_objs_list))
        else:
            return self.ipfs_objs_list

    def download_data(self, ipfs_obj: IPFSObject):
        if ipfs_obj.filetype == RemoteFileType.TAR:
            tarfile = tardata(self.file_read(ipfs_obj.url))
            for fname, content in tarfile:
                yield fname, content
        elif ipfs_obj.filetype == RemoteFileType.ZIP:
            zipfile = zipdata(self.file_read(ipfs_obj.url))
            for fname, content in zipfile:
                yield fname, content
        elif ipfs_obj.filetype == RemoteFileType.PLAIN:
            yield ipfs_obj.url, self.file_read(ipfs_obj.url).read()
        elif ipfs_obj.filetype == RemoteFileType.PLAIN_STREAM:
            yield ipfs_obj.url, self.file_read(ipfs_obj.url)
        else:
            raise ValueError("Invalid remote_filetype")

    def get_stream(self, ipfs_objs_list):
        return chain.from_iterable(map(self.download_data, ipfs_objs_list))

    def worker_dist(self, objs):
        if self.dist:
            total_size = len(objs)
            objs = objs[self.rank:total_size:self.world_size]

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            wid = worker_info.id
            num_workers = worker_info.num_workers
            length = len(objs)
            return objs[wid:length:num_workers]
        else:
            return objs

    def __iter__(self):
        objs = self.worker_dist(self.shuffled_list)
        return self.get_stream(objs)

    def __len__(self):
        return len(self.ipfs_objs_list)

    def set_epoch(self, epoch):
        self.epoch = epoch


class ShuffleDataset(IterableDataset):
    def __init__(self, dataset, buffer_size):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        shufbuf = []
        try:
            dataset_iter = iter(self.dataset)
            for _ in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except StopIteration:
            self.buffer_size = len(shufbuf)

        try:
            while True:
                try:
                    if self.buffer_size == 0:
                        break
                    evict_idx = random.randint(0, self.buffer_size - 1)
                    yield shufbuf.pop(evict_idx)
                    item = next(dataset_iter)
                    shufbuf.append(item)
                except StopIteration:
                    break
            while len(shufbuf) > 0:
                evict_idx = random.randint(0, len(shufbuf) - 1)
                yield shufbuf.pop(evict_idx)
        except GeneratorExit: # pragma: no cover
            pass
