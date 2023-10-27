import pytest

from mdai_utils.upload_dataset import upload_dataset


@pytest.mark.upload_only(
    reason="Only need to upload once. run pytest tests with --upload-only to run it."
)
def test_upload_dataset(mdai_setup):
    parameters = mdai_setup["parameters"]
    mdai_dataset_id = parameters.get("mdai_dataset_id")
    fixtures_dir = mdai_setup["fixtures_dir"]
    dicom_dir = fixtures_dir / "humanct_0002_1000_1004"
    assert dicom_dir.exists()
    completed_process = upload_dataset(mdai_dataset_id, dicom_dir)
    process_message = completed_process.stdout.strip()
    print(process_message)
    # Check the status of subprocess
    assert completed_process.returncode == 0
