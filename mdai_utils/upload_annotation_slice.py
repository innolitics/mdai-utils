import itk
import mdai
import numpy as np


def read_data_image(image_path):
    """
    Read an image using itk, and returns a numpy data array.
    """
    image = itk.imread(image_path)
    data_np = itk.array_from_image(image)
    return data_np


def upload_data_annotation_slice(
    data_np: np.ndarray,
    sop_instance_uid: str,
    mdai_client: mdai.Client,
    mdai_project_id: str,
    mdai_dataset_id: str,
    mdai_label_id: str,
) -> list:
    """
    Uploads the input data annotation (a 2D slice) to the serverwith the specified sop_instance_uid.
    Args:
        data_np (np.ndarray): numpy array with pixel data.
        sop_instance_uid (str): SOPInstanceUID of the DICOM key-slice image. Returned from @inverse_transform
        mdai_client (mdai.Client): Client to the MD.ai API. See @get_mdai_client
        mdai_project_id (str): Project ID. Check in the MD.ai web interface.
        mdai_dataset_id (str): Dataset ID. Check in the MD.ai web interface.
        mdai_label_id (str): Label ID. Check in the MD.ai web interface.

    Returns:
        failed_annotations (list): List of failed annotations. If empty, all annotations were uploaded successfully.
    """

    annotation_dict = {
        "labelId": mdai_label_id,
        "SOPInstanceUID": sop_instance_uid,
        "data": mdai.common_utils.convert_mask_data(data_np),
    }
    failed_annotations = mdai_client.import_annotations(
        [annotation_dict], mdai_project_id, mdai_dataset_id
    )
    return failed_annotations


def upload_image_annotation_slice(
    segmentation_image_path: str,
    sop_instance_uid: str,
    mdai_client: mdai.Client,
    mdai_project_id: str,
    mdai_dataset_id: str,
    mdai_label_id: str,
) -> list:
    """
    Uploads an annotation to the server. It requires that the DICOM image is already
    uploaded to the server. See upload_dicom_image.py for that.
    The input image can be in any format supported by ITK.

    Args:
        segmentation_image_path (str): Path to the segmentation image.
        sop_instance_uid (str): SOPInstanceUID of the DICOM key-slice image. Returned from @inverse_transform
        mdai_client (mdai.Client): Client to the MD.ai API. See @get_mdai_client
        mdai_project_id (str): Project ID. Check in the MD.ai web interface.
        mdai_dataset_id (str): Dataset ID. Check in the MD.ai web interface.
        mdai_label_id (str): Label ID. Check in the MD.ai web interface.

    Returns:
        failed_annotations (list): List of failed annotations. If empty, all annotations were uploaded successfully.
    """
    data_np = read_data_image(segmentation_image_path)
    if data_np.ndim == 3:
        # The perpendicular dimension is at index 0 in the numpy array.
        data_np = data_np.squeeze(0)
    return upload_data_annotation_slice(
        data_np=data_np,
        sop_instance_uid=sop_instance_uid,
        mdai_client=mdai_client,
        mdai_project_id=mdai_project_id,
        mdai_dataset_id=mdai_dataset_id,
        mdai_label_id=mdai_label_id,
    )


def _get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_annotation",
        type=str,
        required=True,
        help="Path to the segmentation image to upload.",
    )
    parser.add_argument(
        "-l",
        "--label_name",
        type=str,
        required=True,
        help="label name corresponding to the annotation.",
    )
    parser.add_argument(
        "--sop_instance_uid",
        type=str,
        default=None,
        help="sop_instance_uid of the annotation file. Needed to match the annotation with the DICOM image in mdai.",
    )
    parser.add_argument(
        "-p",
        "--parameters",
        type=str,
        default=None,
        help="""
Path to a json file containing the parameters for md.ai variables: mdai_project_id, mdai_dataset_id, mdai_label_ids, etc.
See example in tests/test_parameters.json.
""",
    )

    return parser


def main(
    input_annotation,
    label_name,
    sop_instance_uid,
    mdai_client,
    mdai_project_id,
    mdai_dataset_id,
    mdai_label_ids,
):
    mdai_label_id = mdai_label_ids[label_name]
    failed_annotations = upload_image_annotation_slice(
        segmentation_image_path=input_annotation,
        sop_instance_uid=sop_instance_uid,
        mdai_client=mdai_client,
        mdai_project_id=mdai_project_id,
        mdai_dataset_id=mdai_dataset_id,
        mdai_label_id=mdai_label_id,
    )
    return failed_annotations


if __name__ == "__main__":
    import json

    from mdai_utils.common import get_mdai_access_token

    parser = _get_parser()
    args = parser.parse_args()
    print(args)

    with open(args.parameters, "r") as f:
        parameters = json.load(f)

    mdai_project_id = parameters["mdai_project_id"]
    mdai_dataset_id = parameters["mdai_dataset_id"]
    mdai_label_ids = parameters["mdai_label_ids"]
    mdai_domain = parameters["mdai_domain"]

    input_annotation = args.input_annotation
    label_name = args.label_name
    mdai_label_id = mdai_label_ids[label_name]
    sop_instance_uid = args.sop_instance_uid
    if sop_instance_uid is None:
        raise ValueError(
            "sop_instance_uid is required to match the annotation with the DICOM image in mdai."
        )
    token = get_mdai_access_token()
    mdai_client = mdai.Client(domain=mdai_domain, access_token=token)

    failed_annotations = main(
        input_annotation=input_annotation,
        label_name=label_name,
        sop_instance_uid=sop_instance_uid,
        mdai_client=mdai_client,
        mdai_project_id=mdai_project_id,
        mdai_dataset_id=mdai_dataset_id,
        mdai_label_ids=mdai_label_ids,
    )

    if len(failed_annotations) == 0:
        print("All annotations uploaded successfully.")
        exit(0)
    else:
        print(f"Failed annotations: {failed_annotations}")
        exit(1)
