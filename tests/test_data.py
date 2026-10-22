import json

import pytest

from cir.data import alpaca_record, write_dataset
from cir.mixtures import mixture_counts


def test_response_mapping_preserves_unicode_and_input(tmp_path):
    row = {"instruction": "Explain café", "response": "A café", "input": "context"}
    write_dataset([row], tmp_path, "C++_data_12K", {"revision": "example-commit"})
    info = json.loads((tmp_path / "dataset_info.json").read_text())
    records = json.loads((tmp_path / info["C++_data_12K"]["file_name"]).read_text())
    assert records == [{"instruction": "Explain café", "output": "A café", "input": "context"}]
    metadata = json.loads((tmp_path / "C++_data_12K.provenance.json").read_text())
    assert metadata["revision"] == "example-commit"
    assert metadata["num_samples"] == 1
    with pytest.raises(FileExistsError):
        write_dataset([row], tmp_path, "C++_data_12K", {})


@pytest.mark.parametrize("response", [None, "", "  ", 12])
def test_invalid_responses_are_rejected(response):
    with pytest.raises(ValueError):
        alpaca_record({"instruction": "Explain", "response": response})


def test_rejects_path_traversal_and_empty_data(tmp_path):
    with pytest.raises(ValueError):
        write_dataset([], tmp_path, "../outside", {})
    with pytest.raises(ValueError):
        write_dataset([], tmp_path, "empty", {})
    assert not (tmp_path / "dataset_info.json").exists()


def test_mixture_ratio_means_natural_language_fraction():
    assert mixture_counts(120000, 2.5) == (117000, 3000)
    assert mixture_counts(120000, 97.5) == (3000, 117000)
    with pytest.raises(ValueError):
        mixture_counts(120000, 101)
