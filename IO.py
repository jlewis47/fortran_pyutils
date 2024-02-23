import numpy as np
import sys


def read_record(stream, tgt_size, dtype, debug=False):
    nread = 0

    if not isinstance(dtype, list):
        bytesize = np.dtype(dtype).itemsize
        total_bytesize = tgt_size * bytesize
    else:
        bytesize = [np.dtype(dt).itemsize for dt in dtype]
        total_bytesize = np.sum(bytesize)

    if tgt_size > 1:  # if larger than one element, read in chunks

        outs = np.empty(total_bytesize, dtype="S1")

        if debug:
            print(total_bytesize, bytesize, tgt_size)
            while nread < total_bytesize:
                nrec = abs(np.fromfile(stream, dtype=np.int32, count=1)[0])
                print(nread, nrec, len(outs))
                outs[nread : nread + nrec] = np.frombuffer(
                    stream.read(nrec), dtype="S1"
                )
                nread += nrec
                nrec = abs(np.fromfile(stream, dtype=np.int32, count=1)[0])

        else:  # if not debug avoid debug checks
            while nread < total_bytesize:
                nrec = abs(np.fromfile(stream, dtype=np.int32, count=1)[0])
                outs[nread : nread + nrec] = np.frombuffer(
                    stream.read(nrec), dtype="S1"
                )
                nread += nrec
                nrec = abs(np.fromfile(stream, dtype=np.int32, count=1)[0])

        return np.frombuffer(outs, dtype=dtype)

    else:  # if one element then we know its i4 buffer, target dtype, i4 buffer
        # middle element could be encoded on more or less than 4 bytes
        # doing this gives performance increase e.g. reading file headers, non negligeable
        # if there are many files
        buff = np.frombuffer(stream.read(4 + bytesize + 4), dtype="S1")[4:8]
        return np.frombuffer(buff, dtype=dtype)[0]


def write_record(stream, elements, dtypes=None):
    """
    elements assumed to be a numpy array unless dtypes is specified
    if dtypes specified (should be an iterable type of dtype names),
    assume elements is an iterable of equivalent length
    """

    size = len(elements)
    byte_order = sys.byteorder

    if dtypes != None:
        assert len(elements) == len(dtypes), "elements and dtypes must be same length"

    stream.write(int(4 * size).to_bytes(4, byte_order))
    if dtypes is None:
        stream.write(elements.tobytes(order="F"))
    else:
        for element, dtype in zip(elements, dtypes):
            stream.write(element.astype(dtype).tobytes(order="F"))
    stream.write(int(4 * size).to_bytes(4, byte_order))
