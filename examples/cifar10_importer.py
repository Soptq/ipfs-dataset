from ipfs_dataset.exporter import IPFSDatasetExporter


ipfs_api_address = "/ip4/127.0.0.1/tcp/5001"

url = "ipfs://QmVwe3symQQWH4LdNg3WAsMHQLdLjLVs8nMTDVdiiiMqCf"

dataset = IPFSDatasetExporter(ipfs_api_address).import_from_ipfs(url)

for data in dataset:
    print(data)
    break
