from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from toothfairy2_feature_extraction.io_utils import (
    case_id_for,
    resolve_inputs,
    save_segmentation,
    shard_inputs,
)


def test_case_id_for_dataset_layout():
    assert case_id_for(Path("/dataset/A003/cbct/volume.nii.gz")) == "A003"
    assert case_id_for(Path("scan.nii.gz")) == "scan"


def test_shards_partition_without_overlap():
    inputs = [Path(f"case-{index}.nii.gz") for index in range(8)]
    shards = [shard_inputs(inputs, 3, index) for index in range(3)]
    assert shards == [inputs[0::3], inputs[1::3], inputs[2::3]]
    assert sorted(item for shard in shards for item in shard) == sorted(inputs)
    with pytest.raises(ValueError):
        shard_inputs(inputs, 0, 0)
    with pytest.raises(ValueError):
        shard_inputs(inputs, 2, 2)


def test_dataset_root_discovery_and_duplicate_detection(tmp_path):
    expected = []
    for case_id in ("A003", "A001"):
        volume = tmp_path / case_id / "cbct" / "volume.nii.gz"
        volume.parent.mkdir(parents=True)
        volume.touch()
        expected.append(volume.resolve())
    assert resolve_inputs([str(tmp_path)]) == sorted(
        expected, key=lambda path: path.parent.parent.name
    )

    duplicate = tmp_path / "other" / "A001.nii.gz"
    duplicate.parent.mkdir()
    duplicate.touch()
    with pytest.raises(ValueError, match="Duplicate case IDs"):
        resolve_inputs([str(tmp_path), str(duplicate)])


def test_segmentation_preserves_nifti_geometry(tmp_path):
    affine = np.array(
        [[-0.2, 0, 0, 10], [0, -0.3, 0, 20], [0, 0, 0.4, 30], [0, 0, 0, 1]], dtype=float
    )
    reference = nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.float32), affine)
    reference.set_qform(affine, 2)
    reference.set_sform(affine, 1)
    input_path = tmp_path / "input.nii.gz"
    nib.save(reference, input_path)
    segmentation_zyx = np.zeros((6, 5, 4), dtype=np.uint8)
    segmentation_zyx[3, 2, 1] = 36
    output_path = tmp_path / "segmentation.nii.gz"
    save_segmentation(output_path, segmentation_zyx, input_path)
    output = nib.load(output_path)
    assert output.shape == reference.shape
    np.testing.assert_allclose(output.affine, affine)
    assert int(output.header["qform_code"]) == 2
    assert int(output.header["sform_code"]) == 1
    assert output.get_data_dtype() == np.dtype(np.uint8)
    assert np.asarray(output.dataobj)[1, 2, 3] == 36
