import itk
import mdai
import numpy as np


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


def read_data_image(image_path):
    """
    Read an image using itk, and returns a numpy data array.
    """
    image = itk.imread(image_path)
    data_np = itk.array_from_image(image)
    return data_np


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

    Args:
        segmentation_image_path (str): Path to the segmentation image. With fixed metadata. See @inverse_transform
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
        mdai_client=mdai_client,
        mdai_project_id=mdai_project_id,
        mdai_dataset_id=mdai_dataset_id,
        mdai_label_id=mdai_label_id,
        sop_instance_uid=sop_instance_uid,
    )
