"""Tests for output-directory initialization."""

from configparser import ConfigParser

import pytest

import lya_2pt.output as output_module
from lya_2pt.output import Output


class _MPICommunicator:
    def __init__(self, rank):
        self.rank = rank
        self.barrier_calls = 0

    def Get_rank(self):
        return self.rank

    def Barrier(self):
        self.barrier_calls += 1


@pytest.mark.parametrize("rank", [0, 1])
def test_mpi_output_directory_setup_runs_on_rank_zero(tmp_path, monkeypatch, rank):
    output_directory = tmp_path / "output"
    config = ConfigParser()
    config["output"] = {"output-dir": str(output_directory)}
    communicator = _MPICommunicator(rank)
    checked_directories = []
    monkeypatch.setattr(output_module, "check_dir", checked_directories.append)

    Output(config["output"], mpi_comm=communicator)

    if rank == 0:
        assert checked_directories == [output_directory, output_directory / "healpix_files_lyaxlya"]
    else:
        assert checked_directories == []
    assert communicator.barrier_calls == 1
