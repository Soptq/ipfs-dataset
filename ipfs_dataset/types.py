from enum import Enum


class RemoteFileType(Enum):
    PLAIN = 1
    FOLDER = 2
    TAR = 3
    ZIP = 4
    PLAIN_STREAM = 5


class IPFSObject(object):
    def __init__(self, url, filetype: RemoteFileType):
        self.url = url
        self.filetype = filetype
