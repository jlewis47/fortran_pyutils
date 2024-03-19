from tracemalloc import start
import numpy as np
from gremlin.read_sim_params import ramses_sim
import os


def btest(i, pos):
    return i & (1 << pos)


def hilbert3d(x, y, z, bit_length, npoint):

    if bit_length > np.int64().nbytes * 8:
        print("Maximum bit length=", np.int64.bit_length)
        print("stop in hilbert3d")
        return

    state_diagram = np.reshape(
        [
            [1, 2, 3, 2, 4, 5, 3, 5],
            [0, 1, 3, 2, 7, 6, 4, 5],
            [2, 6, 0, 7, 8, 8, 0, 7],
            [0, 7, 1, 6, 3, 4, 2, 5],
            [0, 9, 10, 9, 1, 1, 11, 11],
            [0, 3, 7, 4, 1, 2, 6, 5],
            [6, 0, 6, 11, 9, 0, 9, 8],
            [2, 3, 1, 0, 5, 4, 6, 7],
            [11, 11, 0, 7, 5, 9, 0, 7],
            [4, 3, 5, 2, 7, 0, 6, 1],
            [4, 4, 8, 8, 0, 6, 10, 6],
            [6, 5, 1, 2, 7, 4, 0, 3],
            [5, 7, 5, 3, 1, 1, 11, 11],
            [4, 7, 3, 0, 5, 6, 2, 1],
            [6, 1, 6, 10, 9, 4, 9, 10],
            [6, 7, 5, 4, 1, 0, 2, 3],
            [10, 3, 1, 1, 10, 3, 5, 9],
            [2, 5, 3, 4, 1, 6, 0, 7],
            [4, 4, 8, 8, 2, 7, 2, 3],
            [2, 1, 5, 6, 3, 0, 4, 7],
            [7, 2, 11, 2, 7, 5, 8, 5],
            [4, 5, 7, 6, 3, 2, 0, 1],
            [10, 3, 2, 6, 10, 3, 4, 4],
            [6, 1, 7, 0, 5, 2, 4, 3],
        ],
        (12, 2, 8),
    )

    # print(bit_length)
    # print(npoint)
    # print(x)

    order = np.zeros(npoint, dtype=np.float64)

    x_bit_mask = np.zeros(bit_length, dtype=bool)
    y_bit_mask = np.zeros(bit_length, dtype=bool)
    z_bit_mask = np.zeros(bit_length, dtype=bool)

    # print(len(x_bit_mask))

    for ip in range(npoint):
        # convert to binary
        for i in range(bit_length):
            x_bit_mask[i] = btest(x, i)
            y_bit_mask[i] = btest(y, i)
            z_bit_mask[i] = btest(z, i)
        # y_bit_mask = np.unpackbits(np.array([y], dtype=np.uint64).view(np.uint8))
        # z_bit_mask = np.unpackbits(np.array([z], dtype=np.uint64).view(np.uint8))

        # print(x_bit_mask)

        # interleave bits
        i_bit_mask = np.zeros(3 * bit_length, dtype=bool)
        # i_bit_mask[2::3] = x_bit_mask[:bit_length]
        # i_bit_mask[1::3] = y_bit_mask[:bit_length]
        # i_bit_mask[::3] = z_bit_mask[:bit_length]
        for i in range(bit_length):
            i_bit_mask[3 * i + 2] = x_bit_mask[i]
            i_bit_mask[3 * i + 1] = y_bit_mask[i]
            i_bit_mask[3 * i] = z_bit_mask[i]

        # print(i_bit_mask)

        # build Hilbert ordering using state diagram
        cstate = 0
        for i in range(bit_length - 1, -1, -1):
            b2 = 1 if i_bit_mask[3 * i + 2] else 0
            b1 = 1 if i_bit_mask[3 * i + 1] else 0
            b0 = 1 if i_bit_mask[3 * i] else 0
            sdigit = b2 * 4 + b1 * 2 + b0
            nstate = state_diagram[cstate, 0, sdigit]
            hdigit = state_diagram[cstate, 1, sdigit]
            # print(sdigit, 0, cstate, state_diagram[cstate, 0, sdigit])
            # print(sdigit, 1, cstate, state_diagram[cstate, 1, sdigit])
            i_bit_mask[3 * i + 2] = btest(hdigit, 2)
            i_bit_mask[3 * i + 1] = btest(hdigit, 1)
            i_bit_mask[3 * i] = btest(hdigit, 0)
            cstate = nstate

        # print(i_bit_mask)

        # save Hilbert key as double precision real
        order[ip] = np.sum(i_bit_mask * np.power(2, np.arange(3 * bit_length)))

    return order


def read_info_domains(path, snap, outputs=True, SIXDIGITS=False, debug=False):

    if not SIXDIGITS:
        out_str = f"_{snap:05d}"
    else:
        out_str = f"_{snap:06d}"

    info_fname = f"info" + out_str + ".txt"

    if outputs:
        info_path = os.path.join(path, "output" + out_str, info_fname)
    else:
        info_path = os.path.join(path, info_fname)

    if debug:
        print(f"Reading info file: {info_path}")

    with open(info_path, "r") as f:
        lines = f.readlines()

    ncpu = int(lines[0].split("=")[1].replace("\n", ""))

    bound_key = np.zeros(ncpu)

    start_read = False
    l0 = 0
    for il, l in enumerate(lines):
        if not start_read:
            start_read = "DOMAIN" in l
            if start_read:
                l0 = il
                break
        # else:

        # bound_key[il - l0] = np.double(l.split()[1])

    for impi in range(1, ncpu):

        l = lines[l0 + impi]
        # print(impi, l.split())
        bound_key[impi - 1] = np.double(l.split()[1])
        bound_key[impi] = np.double(l.split()[2])

    return bound_key


def get_files(sim: ramses_sim, snap, mins, maxs, debug=False):

    ncpu = sim.ncpu
    lvlmax = sim.levelmax

    # read domain output from snap's info file
    bound_key = read_info_domains(sim.info_path, snap, sim.info_outputs, debug=debug)

    dmax = np.max([maxs[0] - mins[0], maxs[1] - mins[1], maxs[2] - mins[2]])

    for ilvl in range(lvlmax):
        dx = 0.5**ilvl
        if dx < dmax:
            break
    lmin = ilvl
    bit_length = lmin - 1
    maxdom = 2**bit_length

    imin = jmin = kmin = imax = jmax = kmax = 0
    if bit_length > 0:
        imin = int(mins[0] * maxdom)
        imax = imin + 1
        jmin = int(mins[1] * maxdom)
        jmax = jmin + 1
        kmin = int(mins[2] * maxdom)
        kmax = kmin + 1

    dkey = (2 ** (lvlmax + 1) / maxdom) ** sim.ndim
    ndom = 1

    if bit_length > 0:
        ndom = 8

    bounding_min = np.zeros(ndom)
    bounding_max = np.zeros(ndom)

    idom = [imin, imax, imin, imax, imin, imax, imin, imax]
    jdom = [jmin, jmin, jmax, jmax, jmin, jmin, jmax, jmax]
    kdom = [kmin, kmin, kmin, kmin, kmax, kmax, kmax, kmax]

    for iterdom in range(ndom):

        if bit_length > 0:
            # print(idom[iterdom], jdom[iterdom], kdom[iterdom], bit_length, 1)
            order_min = hilbert3d(
                idom[iterdom], jdom[iterdom], kdom[iterdom], bit_length, 1
            )
        else:
            order_min = 0

        bounding_min[iterdom] = order_min * dkey
        bounding_max[iterdom] = (order_min + 1.0) * dkey

        # print(iterdom, order_min, bounding_min[iterdom], bounding_max[iterdom])

    cpu_min = np.zeros(ndom, dtype=int)
    cpu_max = np.zeros(ndom, dtype=int)

    for impi in range(1, ncpu):
        # print(impi, bound_key[impi - 1], bound_key[impi])
        for iterdom in range(ndom):
            if (
                bound_key[impi - 1] <= bounding_min[iterdom]
                and bound_key[impi] > bounding_min[iterdom]
            ):
                cpu_min[iterdom] = impi
            if (
                bound_key[impi - 1] < bounding_max[iterdom]
                and bound_key[impi] >= bounding_max[iterdom]
            ):
                cpu_max[iterdom] = impi

    cpus_to_read = []

    # for iterdom in range(ndom):
    #     print(cpu_min[iterdom], cpu_max[iterdom])
    #     cpus_to_read = np.union1d(
    #         cpus_to_read, np.arange(cpu_min[iterdom], cpu_max[iterdom] + 1)
    #     )

    for iterdom in range(ndom):
        for j in range(cpu_min[iterdom], cpu_max[iterdom] + 1):
            if not j in cpus_to_read:
                cpus_to_read.append(j)

    if debug:
        print(f"{len(cpus_to_read):d} cpus to read")

    return cpus_to_read
