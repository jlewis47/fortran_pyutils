import numpy as np
import sys


def read_record(stream, tgt_size: int, dtype, debug=False, args=None):
    """
    stream should be a file object
    tgt_size is the number of dtype formated elements expected in the record
    dtype can be a list of types if one record contains several values
    e.g. when ramses writes lines consisting of several values in a loop
    dtype should be interpretable by numpy.dtype
    """

    nread = 0

    if not isinstance(dtype, list):
        bytesize = np.dtype(dtype).itemsize
        total_bytesize = tgt_size * bytesize
    else:
        bytesize = [np.dtype(dt).itemsize for dt in dtype]
        total_bytesize = np.sum(bytesize)

    if args == None:
        args = slice(0, tgt_size)

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

        return np.frombuffer(outs, dtype=dtype)[args]

    else:  # if one element then we know its i4 buffer, target dtype, i4 buffer
        # middle element could be encoded on more or less than 4 bytes
        # doing this gives performance increase e.g. reading file headers, non negligeable
        # if there are many files
        buff = np.frombuffer(stream.read(4 + bytesize + 4), dtype="S1")[
            4 : 4 + bytesize
        ]
        return np.frombuffer(buff, dtype=dtype)[0]


def skip_record(
    stream, tgt_size, dtype
):  # sometimes we want to go past next record using seek(,1)
    # faster than reading and discarding
    """
    stream should be a file object
    tgt_size is the number of dtype formated elements expected in the record
    dtype can be a list of types if one record contains several values
    e.g. when ramses writes lines consisting of several values in a loop
    dtype should be interpretable by numpy.dtype
    """

    nread = 0

    while nread < tgt_size:
        nrec = abs(np.fromfile(stream, dtype=np.int32, count=1)[0])
        stream.seek(nrec, 1)
        nread += nrec
        nrec = abs(np.fromfile(stream, dtype=np.int32, count=1)[0])


def read_tgt_fields(
    data: dict,
    tgt_fields: list,
    fields: list,
    src,
    nrecord: int,
    args=None,
    debug=False,
):
    """
    use on open f90 unformatted file after loading heading information
    file contains the named fields "fields" a list of tuples of (name,2nd dimension,dtype), user requests "tgt_fields"
    every field is expected to contain nrecord*ndim elements
    args is a list of indexes of values within 0:nrecord to store for every field

    designed for f90 unformatted binary files where data are organized by fields e.g.
    particle ids
    masses
    positions
    etc
    """

    field_names = [tgt[0] for tgt in fields]
    field_dims = [tgt[1] for tgt in fields]
    field_types = [tgt[2] for tgt in fields]

    if type(tgt_fields[0]) == tuple:
        tgt_field_names = [tgt[0] for tgt in tgt_fields]
        if len(tgt_fields[0]) == 2:
            field_types_read = [tgt[1] for tgt in tgt_field_names]
        elif len(tgt_fields[0]) == 3:
            field_types_read = [tgt[2] for tgt in tgt_fields]
    else:
        tgt_field_names = [tgt for tgt in tgt_fields]
        field_types_read = field_types

    # print(field_names, field_dims, field_types, tgt_field_names, field_types_read)

    max_field_idx = (
        max([field_names.index(tgt) for tgt in field_names if tgt in tgt_field_names])
        + 1
    )

    if args is None:  # if no argument given then we take everything
        args = slice(0, nrecord)
        nstore = nrecord
    else:
        nstore = len(args)

    for idx in range(max_field_idx):
        cur_dim = field_dims[idx]
        if debug:
            print(field_names[idx], cur_dim, field_types[idx])
        if field_names[idx] in tgt_field_names:
            if cur_dim == 1:
                shape = nstore
            else:
                shape = (nstore, cur_dim)

            read_dat = np.empty(shape, dtype=field_types[idx])

            if cur_dim == 1:
                read_dat[:] = read_record(
                    src, nrecord, field_types_read[idx], args=args, debug=debug
                )
            else:
                for idim in range(cur_dim):
                    read_dat[:, idim] = read_record(
                        src, nrecord, field_types_read[idx], args=args, debug=debug
                    )

            data[field_names[idx]] = read_dat[:]

        else:
            for idim in range(cur_dim):
                # read_record(src, nrecord, field_types_read[idx])
                skip_record(src, nrecord, field_types_read[idx])


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
