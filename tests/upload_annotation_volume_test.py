import json

from mdai_utils.dicoms_to_volume import metadata_dict_to_sop_instance_uids
from mdai_utils.upload_annotation_volume import upload_image_annotation_volume


def test_upload_image_annotation_volume(mdai_setup):
    parameters = mdai_setup["parameters"]
    fixtures_dir = mdai_setup["fixtures_dir"]
    mdai_client = mdai_setup["mdai_client"]

    mdai_project_id = parameters["mdai_project_id"]
    mdai_dataset_id = parameters["mdai_dataset_id"]
    mdai_label_ids = parameters["mdai_label_ids"]
    label_name = parameters["labels"][0]
    mdai_label_id = mdai_label_ids[label_name]

    sop_instance_uids_file_path = (
        fixtures_dir / "humanct_0002_1000_1004_SOPInstanceUIDs.json"
    )
    if not sop_instance_uids_file_path.exists():
        raise FileNotFoundError(
            f"The file {sop_instance_uids_file_path} does not exist."
        )
    with open(sop_instance_uids_file_path) as f:
        metadata_dict = json.load(f)

    sop_instance_uids = metadata_dict_to_sop_instance_uids(metadata_dict)

    input_annotation = fixtures_dir / "humanct_0002_1000_1004_seg.nii.gz"

    failed_annotations = upload_image_annotation_volume(
        segmentation_image_path=input_annotation,
        sop_instance_uids=sop_instance_uids,
        mdai_client=mdai_client,
        mdai_project_id=mdai_project_id,
        mdai_dataset_id=mdai_dataset_id,
        mdai_label_id=mdai_label_id,
    )
    assert len(failed_annotations) == 0
