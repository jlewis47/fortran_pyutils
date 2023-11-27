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
    

    outs = np.zeros(total_bytesize, dtype="S1")

    if debug:print(total_bytesize, bytesize, tgt_size)

    while nread < total_bytesize:
        nrec = abs(np.fromfile(stream, dtype=np.int32, count=1)[0])
        if debug:print(nrec)
        outs[nread : nread + nrec] = np.frombuffer(stream.read(nrec), dtype="S1")
        nread += nrec
        nrec = abs(np.fromfile(stream, dtype=np.int32, count=1)[0])
        # print(nrec)

    return np.frombuffer(outs, dtype=dtype)


def write_record(stream, elements, dtypes=None):
    """
    elements assumed to be a numpy array unless dtypes is specified
    if dtypes specified (should be an iterable type of dtype names), 
    assume elements is an iterable of equivalent length
    """

    size = len(elements)     
    byte_order = sys.byteorder

    if dtypes != None:
        assert len(elements)==len(dtypes), "elements and dtypes must be same length"
    

    stream.write(int(4*size).to_bytes(4, byte_order))
    if dtypes is None:
        stream.write(elements.tobytes(order='F'))
    else:
        for element, dtype in zip(elements, dtypes):
            stream.write(element.astype(dtype).tobytes(order='F'))
    stream.write(int(4*size).to_bytes(4, byte_order))
