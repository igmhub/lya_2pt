"""Smoke tests for the installed command-line entry points."""

import sys

import pytest

from lya_2pt.scripts import run, run_cf, run_dmat, run_export, run_mpi


@pytest.mark.parametrize(
    ("program", "entry_point"),
    [
        ("lya-2pt", run.main),
        ("lya-2pt-cf", run_cf.main),
        ("lya-2pt-dmat", run_dmat.main),
        ("lya-2pt-export", run_export.main),
        ("lya-2pt-mpi", run_mpi.main),
    ],
)
def test_cli_help(program, entry_point, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [program, "--help"])

    with pytest.raises(SystemExit) as error:
        entry_point()

    assert error.value.code == 0
    assert "usage:" in capsys.readouterr().out
