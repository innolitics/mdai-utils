import datetime
import difflib
import json
import os
from pathlib import Path
from typing import Union

import cv2
import itk
import mdai
import numpy as np
from bidict import bidict

from mdai_utils.common import get_mdai_access_token

DEFAULT_DATA_PATH: str = "./data"
LABELS_FOLDER_IDENTIFIER: str = "segmentations"


def supersample_vertices(vertices, original_image_shape, upscale_factor):
    """
    Upscale the vertices by a factor of upscale_factor
    Also returns a mask of zeros with the upscaled shape.
    """
    vertices = vertices * upscale_factor
    mask_upscaled = np.zeros(
        np.array(original_image_shape) * upscale_factor, dtype=np.uint8
    )
    return vertices, mask_upscaled


def downsample_mask(mask, downscale_factor):
    """
    Downsample the upscaled mask ( from @supersample_vertices ) by a factor of downscale_factor.
    upscale and downscale factors must be the same integers.
    """
    # interpolation = cv2.INTER_NEAREST # best for upsampling annotations
    interpolation = cv2.INTER_AREA  # best for downsampling annotations
    return cv2.resize(
        mask,
        (mask.shape[1] // downscale_factor, mask.shape[0] // downscale_factor),
        interpolation=interpolation,
    )


def get_empty_mask(original_image_shape=None):
    dtype_mask = np.uint8
    default_shape = [512, 512]
    slice_shape = (
        default_shape if original_image_shape is None else original_image_shape
    )
    slice_label_mask = np.zeros(slice_shape, dtype=dtype_mask)
    return slice_label_mask


def get_mask_from_vertices(vertices, upscale_factor=100, original_image_shape=None):
    """
    vertices is a np.array of vertices of the polygon: [[y1, x1], [y2, x2], ...]
    If you have a list of vertices: [[x1, y1], [x2, y2], ...], you can use:
    vertices = np.array(vertices).reshape((-1, 2))
    """
    slice_label_mask = get_empty_mask(original_image_shape)
    slice_shape = slice_label_mask.shape
    apply_supersample = False if upscale_factor is None else upscale_factor > 1
    if not apply_supersample:
        cv2.fillPoly(
            slice_label_mask, pts=[np.round(vertices).astype(np.int32)], color=[1.0]
        )
    else:
        vertices_upscaled, mask_upscaled = supersample_vertices(
            vertices, slice_shape, upscale_factor
        )
        cv2.fillPoly(
            mask_upscaled,
            pts=[np.round(vertices_upscaled).astype(np.int32)],
            color=[1.0],
        )
        slice_label_mask = downsample_mask(mask_upscaled, upscale_factor)

    return slice_label_mask


def get_mask_from_annotation(row, original_image_shape=None, upscale_factor=100):
    annotationMode = row["annotationMode"]
    if annotationMode == "freeform":
        vertices = np.array(row["data"]["vertices"]).reshape((-1, 2))
        slice_label_mask = get_mask_from_vertices(
            vertices, upscale_factor, original_image_shape
        )
    elif annotationMode == "mask":
        slice_label_mask = get_empty_mask(original_image_shape)
        if row.data["foreground"]:
            for i in row.data["foreground"]:
                slice_label_mask = cv2.fillPoly(
                    slice_label_mask, pts=[np.array(i, dtype=np.int32)], color=[1.0]
                )
        if row.data["background"]:
            for i in row.data["background"]:
                slice_label_mask = cv2.fillPoly(
                    slice_label_mask, pts=[np.array(i, dtype=np.int32)], color=[0.0]
                )

    else:
        raise ValueError(
            "Unknown annotation mode: {}. Should be 'freeform' or 'mask'".format(
                annotationMode
            )
        )

    return slice_label_mask


def get_last_json_file(
    input_path: Union[str, os.PathLike] = DEFAULT_DATA_PATH,
    match_str: str = "",
) -> os.PathLike[str]:
    query_regex = f"*{match_str}*.json" if match_str else "*.json"
    jsons_files = sorted(
        Path(input_path).glob(query_regex), key=os.path.getctime, reverse=True
    )
    if len(jsons_files) == 0:
        raise FileNotFoundError("No json files found in {}".format(input_path))
    return jsons_files[0]


def match_folder_to_json_file(
    json_file: Union[str, os.PathLike],
    input_path: Union[str, os.PathLike] = DEFAULT_DATA_PATH,
    cutoff: float = 0.6,
) -> Union[os.PathLike, None]:
    # List of the folders in the input path
    json_file = str(json_file)
    folders = [
        f for f in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, f))
    ]
    # Match the one that is closer to the name of the json file:
    json_file_name = os.path.basename(json_file.replace("_annotations", "_images"))
    folder_match = difflib.get_close_matches(
        json_file_name, folders, n=50, cutoff=cutoff
    )
    # Remove matches containing the word LABELS_FOLDER_IDENTIFIER
    folder_match = [f for f in folder_match if LABELS_FOLDER_IDENTIFIER not in f]
    return Path(input_path) / folder_match[0] if folder_match else None


def main(args):
    print(args)

    parameters = {}
    if args.parameters is not None:
        with open(args.parameters) as f:
            parameters = json.load(f)

    mdai_domain = args.mdai_domain or parameters.get("mdai_domain", "md.ai")
    mdai_token_env_variable = args.mdai_token_env_variable or parameters.get(
        "mdai_token_env_variable", "MDAI_TOKEN"
    )
    mdai_project_id = args.mdai_project_id or parameters.get("mdai_project_id", None)
    mdai_label_group_id = args.mdai_label_group_id or parameters.get(
        "mdai_label_group_id", None
    )
    mdai_dataset_id = args.mdai_dataset_id or parameters.get("mdai_dataset_id", None)

    out_folder = args.out_folder or parameters.get("out_folder", DEFAULT_DATA_PATH)
    no_download = args.no_download or parameters.get("no_download", False)
    mdai_annotations_only = not args.download_dicoms or parameters.get(
        "mdai_annotations_only", True
    )
    no_fixing_metadata = args.no_fixing_metadata or parameters.get(
        "mdai_no_fixing_metadata", False
    )
    labels = args.labels or parameters.get("labels", [])

    mdai_label_ids = bidict(parameters.get("mdai_label_ids", {}))
    if labels:
        # Get the ids of the labels you want to process
        # pop the labels that are not in the list of labels
        mdai_label_ids = bidict(
            {k: v for k, v in mdai_label_ids.items() if k in labels}
        )

        if len(mdai_label_ids) != len(labels):
            raise ValueError(
                f"Some labels {labels} not found in the mdai_label_ids dictionary {mdai_label_ids.keys()}"
            )

    if not Path(out_folder).exists():
        raise FileNotFoundError(f"The output folder does not exist: {out_folder}")

    # Checks
    if mdai_project_id is None:
        raise ValueError(
            "Please provide mdai_project_id with either the parameters file or the command line arguments."
        )
    if mdai_label_group_id is None:
        raise ValueError(
            "Please provide mdai_label_group_id with either the parameters file or the command line arguments."
        )

    if not no_download:
        mdai_token = get_mdai_access_token(env_variable=mdai_token_env_variable)
        mdai_client = mdai.Client(domain=mdai_domain, access_token=mdai_token)
        mdai_client.project(
            mdai_project_id,
            label_group_id=mdai_label_group_id,
            dataset_id=mdai_dataset_id,
            path=str(out_folder),
            annotations_only=mdai_annotations_only,
        )

    # Get the json for annotations
    last_json_file = get_last_json_file(out_folder, match_str=mdai_project_id)

    print(f"Last json file: {last_json_file}")

    # And get the folder where dicoms are, we use the match, because we are not always downloading data
    match_folder = match_folder_to_json_file(last_json_file)
    print(f"Matching data folder (dicoms): {match_folder or 'None'}")

    now = datetime.datetime.utcnow()
    now_formatted_date = now.strftime("%Y-%m-%d-%H%M%S")
    labels_parent_folder = (
        Path(out_folder)
        / f"{Path(last_json_file).stem}_{LABELS_FOLDER_IDENTIFIER}_{now_formatted_date}"
    )
    labels_parent_folder.mkdir(parents=True, exist_ok=True)
    pair_data_json_file = labels_parent_folder / "pair_data.json"
    pair_data = []

    result = mdai.common_utils.json_to_dataframe(last_json_file)

    # The following variables are all pandas
    # mdai_labels = result["labels"]
    # mdai_studies = result["studies"]
    mdai_annotations = result["annotations"]
    hash_columns = [
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
    ]
    dh = mdai_annotations.groupby(hash_columns)

    for hash_, group in dh:
        sep = "__"
        hash_id = sep.join(hash_)
        pair_data_entry = {}

        raw_slice_image = None
        if not no_fixing_metadata:
            if match_folder is None:
                raise ValueError(
                    f"Could not find a data folder matching the json file {last_json_file}."
                    "Have you downloaded the dicom data? Disable this check with --no_fixing_metadata"
                )
            # Read the key slice
            hash_path = Path(hash_id.replace(sep, "/") + ".dcm")
            raw_slice_path = Path(match_folder) / hash_path
            pair_data_entry["image"] = str(raw_slice_path.resolve())
            raw_slice_image = itk.imread(raw_slice_path)

        for _, row in group.iterrows():
            row_label_id = row["labelId"]
            # Skip if the label id is not in the list of labels we want to process
            if labels and row_label_id not in mdai_label_ids.inverse:
                continue
            # We handle mask and freeform annotations differently
            annotationMode = row["annotationMode"]
            if annotationMode not in ["mask", "freeform"]:
                continue
            width, height = row["width"], row["height"]
            shape = np.array([int(height), int(width)])  # Y, X
            slice_label_mask = get_mask_from_annotation(row, original_image_shape=shape)
            # Add a dummy dimension for the z axis
            slice_label_mask = np.expand_dims(slice_label_mask, axis=0)  # Z, Y, X
            # slice_label_mask = np.flipud(slice_label_mask)
            # Use itk to save the mask, even in nifti format
            label_image = itk.image_from_array(slice_label_mask)
            # We are going to save a 3D slice, we are interested in storing the z-position.

            label_name = (
                mdai_label_ids.inverse.get(row_label_id, False)
                or row.get("labelName", False)
                or "noname"
            )
            label_file = labels_parent_folder / f"{label_name}__{hash_id}.nii.gz"
            # Check we are not introducing two labels:
            if label_name in pair_data_entry:
                raise ValueError(
                    f"Found two labels with the same name {label_name} in the same slice {hash_id}. Please fix it in md.ai."
                )
            pair_data_entry[label_name] = str(label_file.resolve())

            if not no_fixing_metadata:
                assert raw_slice_image is not None
                label_image.SetOrigin(raw_slice_image.GetOrigin())
                label_image.SetSpacing(raw_slice_image.GetSpacing())
                label_image.SetDirection(raw_slice_image.GetDirection())

            itk.imwrite(label_image, str(label_file))

        pair_data.append(pair_data_entry)

    with open(pair_data_json_file, "w") as f:
        json.dump(pair_data, f, indent=2)

    print(f"pair_data_folder: {labels_parent_folder}")


def get_download_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download all annotations for the specified label group from MD.ai. Requires MDAI_TOKEN environment variable to be set."
    )
    parser.add_argument("-o", "--out_folder", type=str, default=None)

    parser.add_argument(
        "--mdai_domain",
        type=str,
        default=None,
        help="MD.ai domain. Default: md.ai",
    )
    parser.add_argument(
        "--mdai_token_env_variable",
        type=str,
        default=None,
        help="Environment variable with the MD.ai token. Default: MDAI_TOKEN",
    )
    parser.add_argument(
        "--mdai_project_id",
        type=str,
        default=None,
        help="Project ID. Check in the MD.ai web interface.",
    )
    parser.add_argument(
        "--mdai_label_group_id",
        type=str,
        default=None,
        help="Label group ID For example: 'G_P914xZ'.",
    )
    parser.add_argument(
        "--mdai_dataset_id",
        type=str,
        default=None,
        help="Dataset ID in the project. For example: 'D_4Ve4b3'.",
    )
    parser.add_argument(
        "--no_download",
        action="store_true",
        help="If set, the script will not download the images, but still work on the last json file downloaded.",
    )
    parser.add_argument(
        "--download_dicoms",
        action="store_true",
        help="If set, this will download all the dicoms AND the annotations. Do it once, or when data has changed. It can also be set in parameters with mdai_annotations_only=false ",
    )
    parser.add_argument(
        "--no_fixing_metadata",
        action="store_true",
        help="If set, the script will download segmentations, but the metadata will not be matched to their slice. Use it in case you will fix the metadata by other means",
    )

    parser.add_argument(
        "-l",
        "--labels",
        action="append",
        type=str,
        default=[],
        help="Labels to process. If none is set, all labels found will be processed.",
    )

    parser.add_argument(
        "--parameters",
        type=str,
        default=None,
        help="""
Path to a json file containing the parameters for md.ai variables: mdai_project_id, mdai_dataset_id, mdai_mask_label_ids_by_label_group.
See example in mdai_common/mdai_parameters_example.json.
""",
    )

    return parser


if __name__ == "__main__":
    parser = get_download_parser()
    args = parser.parse_args()
    main(args)
