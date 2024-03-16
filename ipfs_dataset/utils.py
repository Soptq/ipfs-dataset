import io
import tarfile
import zipfile
import re
from io import BytesIO, SEEK_SET, SEEK_CUR, SEEK_END


meta_prefix = "__"
meta_suffix = "__"


def reraise_exception(exn):  # pragma: no cover
    """Called in an exception handler to re-raise the exception."""
    raise exn


def tardata(stream, skip_meta=r"__[^/]*__($|/)", handler=reraise_exception):
    """Iterator yielding filename, content pairs for the given tar stream.
    """
    # eliminated from test coverage since checking requires invalid tarfile
    try:
        stream = tarfile.open(fileobj=stream, mode="r|*")
        for tarinfo in stream:
            try:
                if not tarinfo.isreg():  # pragma: no cover
                    continue
                fname = tarinfo.name
                if fname is None:  # pragma: no cover
                    continue
                if ("/" not in fname and fname.startswith(meta_prefix)
                        and fname.endswith(meta_suffix)):  # pragma: no cover
                    # skipping metadata for now
                    continue
                if skip_meta is not None and re.match(skip_meta, fname):  # pragma: no cover
                    continue
                data = stream.extractfile(tarinfo).read()
                yield fname, data
            except Exception as exn:  # pragma: no cover
                if handler(exn):
                    continue
                else:
                    break
        del stream
    except Exception as exn:  # pragma: no cover
        handler(exn)


def zipdata(stream, skip_meta=r"__[^/]*__($|/)", handler=reraise_exception):
    """Iterator yielding filename, content pairs for the given zip stream.
    """
    # eliminated from test coverage since checking requires invalid zipfile
    try:
        with zipfile.ZipFile(io.BytesIO(stream.read()), 'r') as zfile:
            try:
                for file_ in zfile.namelist():
                    if file_ is None:
                        continue
                    if ("/" not in file_ and file_.startswith(meta_prefix)
                            and file_.endswith(meta_suffix)):  # pragma: no cover
                        # skipping metadata for now
                        continue
                    if skip_meta is not None and re.match(skip_meta, file_):  # pragma: no cover
                        continue
                    data = zfile.read(file_)
                    yield file_, data
            except Exception as exn:  # pragma: no cover
                print("Error:", exn)
    except Exception as exn:  # pragma: no cover
        print("Error:", exn)

    try:
        with zipfile.ZipFile(stream, 'r') as zfile:
            try:
                for file_ in zfile.namelist():
                    if file_ is None:
                        continue
                    if ("/" not in file_ and file_.startswith(meta_prefix)
                            and file_.endswith(meta_suffix)):  # pragma: no cover
                        # skipping metadata for now
                        continue
                    if skip_meta is not None and re.match(skip_meta, file_):  # pragma: no cover
                        continue
                    data = zfile.read(file_)
                    yield file_, data
            except Exception as exn:  # pragma: no cover
                print("Error:", exn)
    except Exception as exn:  # pragma: no cover
        print("Error:", exn)


class ResponseStream(object):
    def __init__(self, request_iterator):
        self._bytes = BytesIO()
        self._iterator = request_iterator

    def _load_all(self):
        self._bytes.seek(0, SEEK_END)
        for chunk in self._iterator:
            self._bytes.write(chunk)

    def _load_until(self, goal_position):
        current_position = self._bytes.seek(0, SEEK_END)
        while current_position < goal_position:
            try:
                current_position += self._bytes.write(next(self._iterator))
            except StopIteration:
                break

    def tell(self):
        return self._bytes.tell()

    def read(self, size=None):
        left_off_at = self._bytes.tell()
        if size is None:
            self._load_all()
        else:
            goal_position = left_off_at + size
            self._load_until(goal_position)

        self._bytes.seek(left_off_at)
        return self._bytes.read(size)

    def seek(self, offset, whence=SEEK_SET):
        if whence == SEEK_SET:
            if offset > self._bytes.tell():
                self._load_until(offset)
        elif whence == SEEK_CUR:
            if offset > 0:
                self._load_until(self._bytes.tell() + offset)
        elif whence == SEEK_END:
            self._load_all()

        self._bytes.seek(offset, whence)

    def seekable(self):
        return True
