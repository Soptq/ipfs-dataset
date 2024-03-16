import io
import math

import requests
from hashlib import md5
import tarfile

import torch
from torch.utils.data import Dataset

import ipfshttpclient

from .ipfs_dataset import IPFSIterableDataset
from .types import IPFSObject, RemoteFileType


class ImportedIPFSDataset(IPFSIterableDataset):
    def __init__(self, ipfs_objs_list, ipfs_api_address):
        self.ipfs_dataset = IPFSIterableDataset(ipfs_objs_list, ipfs_api_address, False)

    def data_generator(self):
        try:
            while True:
                fname, fobj = next(self.ipfs_dataset_iter)
                yield torch.load(io.BytesIO(fobj))
        except StopIteration:
            return

    def __iter__(self):
        self.ipfs_dataset_iter = iter(self.ipfs_dataset)
        return self.data_generator()


class IPFSDatasetExporter(object):
    def __init__(self, ipfs_api_address, shard_bytes=1024 * 1024 * 256):
        self.ipfs_api_address = ipfs_api_address
        self.ipfs_client = ipfshttpclient.connect(ipfs_api_address)
        self.shard_bytes = shard_bytes
        self.shards = []

    def export_to_ipfs(self, dataset: Dataset) -> str:
        metadata = {
            "identifier": "ipfs-dataset",
            "version": "1.0",
            "list": []
        }
        shard = io.BytesIO()
        tar = tarfile.open(fileobj=shard, mode="w")
        size_tar = 0
        for data in dataset:
            buffer = io.BytesIO()
            torch.save(data, buffer)

            buffer.seek(0)
            md5sum = md5(buffer.getbuffer())
            size = buffer.getbuffer().nbytes
            size_tar += size

            buffer.seek(0)
            tarinfo = tarfile.TarInfo(name=md5sum.hexdigest())
            tarinfo.size = size
            tar.addfile(tarinfo, buffer)

            if size_tar >= self.shard_bytes:
                tar.close()
                cid = self.ipfs_client.add_bytes(shard.read(), opts={"pin": True})
                metadata["list"].append(cid)
                shard = io.BytesIO()
                tar = tarfile.open(fileobj=shard, mode="w")
                size_tar = 0

        if size_tar > 0:
            tar.close()
            cid = self.ipfs_client.add_bytes(shard.read(), opts={"pin": True})
            metadata["list"].append(cid)

        cid = self.ipfs_client.add_json(metadata, opts={"pin": True})

        return f"ipfs://{cid}"

    def import_from_ipfs(self, url: str) -> Dataset:
        if url.startswith("ipfs://"):
            cid = url[7:]
            metadata = self.ipfs_client.get_json(cid)
        elif url.startswith("https://") or url.startswith("http://"):
            resp = requests.get(url)
            if resp.status_code != 200:
                raise FileExistsError(f"{url} does not exist")
            metadata = resp.json()
        else:
            raise ValueError(f"Invalid url: {url}")

        if "identifier" not in metadata or metadata["identifier"] != "ipfs-dataset":
            raise ValueError(f"Invalid metadata: {metadata}")

        metadata_version = metadata["version"]
        if metadata_version == "1.0":
            ipfs_objs_list = [IPFSObject(f"ipfs://{cid}", RemoteFileType.TAR) for cid in metadata["list"]]
            return ImportedIPFSDataset(ipfs_objs_list, self.ipfs_api_address)
        else:
            raise ValueError(f"Invalid metadata version: {metadata_version}")
