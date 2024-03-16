from ipfs_dataset.exporter import IPFSDatasetExporter
from torchvision.datasets import CIFAR10


ipfs_api_address = "/ip4/127.0.0.1/tcp/5001"
dataset = CIFAR10(root="data", train=True, download=True)

cid = IPFSDatasetExporter(ipfs_api_address).export_to_ipfs(dataset)

# Should be ipfs://QmVwe3symQQWH4LdNg3WAsMHQLdLjLVs8nMTDVdiiiMqCf
print(f"Dataset exported to IPFS with URL: {cid}")
