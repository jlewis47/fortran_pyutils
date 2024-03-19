Set of routines for reading and writing unformatted f90 binary records and files

read_record, write_record, skip_record have straigtforward use cases


read_tgt_fields is a composite function designed to chain several read_record and skip_record as required to read a with data columns written sequentially (e.g. particle ids, masses, 
positions, velocities, ...) as if often the case with ramses.
One must first read in the header of the file, then the function can be called to read the all the columns of a given size (typically given by the header). Each column is specified thanks
to a list of (name:str, number of dimensions:int, dtype:numpy.dtype interpretable dtype). User may specifiy which file columns to read in a list of 
(name:str, dtype:numpy.dtype interpretable dtype) tuples. This can be exploited to read e.g. f8 data but store it in f4 numpy arrays.
