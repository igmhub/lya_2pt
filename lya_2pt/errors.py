"""This file defines the error classes used by lya_2pt"""


class CosmologyError(Exception):
    """
        Exceptions occurred in class Cosmology
    """


class ReaderException(Exception):
    """
        Exceptions occurred in classes ForestHealpixReader and Tracer2Reader
    """


class MPIError(Exception):
    """
        Exceptions related to MPI parallelization
    """


class ParserError(Exception):
    """
        Exceptions occurred in parsing
    """


class FindBinsError(Exception):
    """
        Exceptions occurred in parsing
    """

