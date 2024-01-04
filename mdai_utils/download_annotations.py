import datetime
import difflib
import json
import logging
import os
from pathlib import Path
from typing import Union

import cv2
import itk
import mdai
import numpy as np
from bidict import bidict

from mdai_utils.common import get_mdai_access_token
from mdai_utils.log_utils import set_dual_logger
from mdai_utils.merge_downloaded_parts import merge_downloaded_parts

DEFAULT_DATA_PATH: str = "./data"
LABELS_FOLDER_IDENTIFIER: str = "segmentations"

logger = logging.getLogger(__name__)


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
    if len(vertices) == 0:
        return slice_label_mask
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
    rowdata = row.get("data", None)
    if rowdata is None:
        return get_empty_mask(original_image_shape)
    if annotationMode == "freeform":
        rowvertices = rowdata.get("vertices", [])
        vertices = np.array(rowvertices).reshape((-1, 2))
        slice_label_mask = get_mask_from_vertices(
            vertices, upscale_factor, original_image_shape
        )
    elif annotationMode == "mask":
        slice_label_mask = get_empty_mask(original_image_shape)
        if rowdata.get("foreground", None) is None:
            return slice_label_mask
        if rowdata.get("background", None) is None:
            return slice_label_mask
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
    folder_to_search: Union[str, os.PathLike],
    hint_str: str = "",
    cutoff: float = 0.6,
) -> Union[os.PathLike, None]:
    # List of the folders in the input path
    json_file = str(json_file)
    folders = [
        f
        for f in os.listdir(folder_to_search)
        if os.path.isdir(os.path.join(folder_to_search, f))
    ]
    # Match the one that is closer to the name of the json file:
    json_file_name = os.path.basename(json_file.replace("_annotations", "_images"))
    folder_match = difflib.get_close_matches(
        json_file_name, folders, n=50, cutoff=cutoff
    )
    # Remove matches containing the word LABELS_FOLDER_IDENTIFIER
    folder_match = [f for f in folder_match if LABELS_FOLDER_IDENTIFIER not in f]
    if hint_str:
        folder_match = [f for f in folder_match if hint_str in f]
    return Path(folder_to_search) / folder_match[0] if folder_match else None


def get_dicom_names_ordered_and_metadata(dicom_dir):
    """
    Read the dicom files in a folder.
    Return a list of tuples (filename, sop_instance_uid) ordered by z position.
    """
    io = itk.GDCMSeriesFileNames.New()  # pyright: ignore[reportGeneralTypeIssues]
    io.AddSeriesRestriction("0008|0021")
    io.SetUseSeriesDetails(True)
    io.SetDirectory(str(dicom_dir))
    io.Update()

    series_uids = io.GetSeriesUIDs()

    selected_index = 0
    if len(series_uids) == 0:
        raise ValueError(f"Found no series in folder: {dicom_dir}")
    elif len(series_uids) > 1:
        error_msg = f"""Found more than one series in the same folder: {dicom_dir}.
        Probably BUG in md.ai, which is merging different series_ids into one."""
        num_files = []
        sops_per_series = []
        for index, s in enumerate(series_uids):
            files = io.GetFileNames(s)
            sops = [Path(f).stem for f in files]
            sops_per_series.append(sops)
            len_files = len(files)
            num_files.append(len_files)
            error_msg += f"\nSeries {index}: {s} with {len_files} files:\n{sops}"
        # Select the first index, if it contains less than 20 files, select the
        # one with more files.
        if num_files[0] < 20:
            selected_index = np.argmax(num_files)
        # Handle the case where there are two series with more than 20 files each
        if all([n >= 20 for n in num_files]):
            # Read the series without the SeriesRestriction
            io = (
                itk.GDCMSeriesFileNames.New()  # pyright: ignore[reportGeneralTypeIssues]
            )
            io.SetUseSeriesDetails(False)
            io.SetDirectory(str(dicom_dir))
            io.Update()
            series_uids = io.GetSeriesUIDs()
            error_msg += f"\nFound more than one series with more than 20 files each {num_files}. Reading the series without date distinction, merging into one series."
            selected_index = 0
            files = io.GetFileNames(series_uids[0])
            num_files = [len(files)]
            sops_per_series = [Path(f).stem for f in files]

        error_msg += (
            f"\nSelected series {selected_index} with {num_files[selected_index]} files"
        )

        error_msg += (
            "\nPlease fix it in md.ai to avoid this warning and clean the dataset."
        )
        logger.error(error_msg)

    series_id = series_uids[selected_index]
    # Ordered by z position
    dicom_names_ordered = io.GetFileNames(series_id)
    # Get also SOPInstanceUID for each file

    # Create an instance of GDCMImageIO
    gdcm_io = itk.GDCMImageIO.New()  # pyright: ignore[reportGeneralTypeIssues]
    gdcm_io.LoadPrivateTagsOn()

    # Get SOPInstanceUID for each file
    dicom_names_with_uid_ordered = []
    metadict_volume = {}
    for i, dicom_name in enumerate(dicom_names_ordered):
        gdcm_io.SetFileName(dicom_name)
        gdcm_io.ReadImageInformation()
        if i == 0:
            reader = (
                itk.ImageFileReader.New(  # pyright: ignore[reportGeneralTypeIssues]
                    FileName=dicom_names_ordered[0], ImageIO=gdcm_io
                )
            )
            reader.Update()
            metadict = reader.GetMetaDataDictionary()
            tagkeys = metadict.GetKeys()
            for tagkey in tagkeys:
                if (
                    tagkey == "ITK_original_direction"
                    or tagkey == "ITK_original_spacing"
                ):
                    continue
                tagvalue = metadict[tagkey]
                metadict_volume[tagkey] = tagvalue

        _, sop_instance_uid = gdcm_io.GetValueFromTag(
            "0008|0018", ""
        )  # Tag for SOPInstanceUID
        dicom_names_with_uid_ordered.append((dicom_name, sop_instance_uid))

    # Add to the metadict_volume a new key: "sop_instances_uids_ordered_by_index"
    sop_instances_uids_ordered_by_index = [
        uid for _, uid in dicom_names_with_uid_ordered
    ]
    metadict_volume[
        "sop_instances_uids_ordered_by_index"
    ] = sop_instances_uids_ordered_by_index

    return dicom_names_with_uid_ordered, metadict_volume


def get_global_annotations(mdai_annotations):
    """
    Get the series annotations from the mdai_annotations dataframe.
    Transform into a dictionary with key: study__series
    and values: list of labels in that series
    """
    # Get the series metadata
    with_study_scope = mdai_annotations["scope"] == "STUDY"
    study_annotations = mdai_annotations.loc[with_study_scope]

    with_series_scope = mdai_annotations["scope"] == "SERIES"
    series_annotations = mdai_annotations.loc[with_series_scope]
    # Transform into a dictionary with key: study__global

    global_annotations_dict = {}
    for _, row in study_annotations.iterrows():
        study_id = row["StudyInstanceUID"]
        global_annotations_dict.setdefault(study_id, {})
        global_annotations_dict[study_id].setdefault("study_labels", [])
        global_annotations_dict[study_id]["study_labels"].append(row["labelName"])

    for _, row in series_annotations.iterrows():
        study_id = row["StudyInstanceUID"]
        series_id = row["SeriesInstanceUID"]
        global_annotations_dict.setdefault(study_id, {})
        global_annotations_dict[study_id].setdefault(series_id, [])
        global_annotations_dict[study_id][series_id].append(row["labelName"])

    global_annotations_dict["mdai_label_group_ids"] = list(
        mdai_annotations["labelGroupId"].unique()
    )
    return global_annotations_dict, study_annotations, series_annotations


def populate_global_labels_dict(global_annotations_dict, study_id, series_id):
    global_labels_dict = {}
    if not global_annotations_dict:
        return global_labels_dict

    if "mdai_label_group_ids" in global_annotations_dict:
        global_labels_dict["mdai_label_group_ids"] = global_annotations_dict[
            "mdai_label_group_ids"
        ]
    if study_id in global_annotations_dict:
        # Check for study labels first:
        if "study_labels" in global_annotations_dict[study_id]:
            for label in global_annotations_dict[study_id]["study_labels"]:
                global_labels_dict.setdefault(label, 0)
                global_labels_dict[label] += 1
        # Check for series labels:
        if series_id in global_annotations_dict[study_id]:
            for label in global_annotations_dict[study_id][series_id]:
                global_labels_dict.setdefault(label, 0)
                global_labels_dict[label] += 1

    return global_labels_dict


def merge_slices_into3D(
    pair_data_json_file,
    labels,
    volumes_path,
    process_grayscale=False,
    global_annotations_dict={},
):
    """
    PRECONDITION: pair_data_json_file contains a list of slices of paired data:
        image (dicom) and label (mask in nifti format)
    We want to read all dicoms as a 3D volume, and merge all nifti masks into a single 3D mask,
    with the same shape as the dicom volume.
    global_annotation is used to write all SERIES and EXAM metadata from mdai into the volume metadata. See @get_global_annotations
    """

    with open(pair_data_json_file) as f:
        pair_data = json.load(f)

    if len(pair_data) == 0:
        raise ValueError(f"pair_data_json_file is empty: {pair_data_json_file}")

    # We want to group all the slices sharing same StudyInstanceUID and SeriesInstanceUID
    pair_data_grouped = {}
    for slice in pair_data:
        hash_id = slice["study__series__sop_ids"]
        study_id, series_id, sop_id = hash_id.split("__")
        if study_id not in pair_data_grouped:
            pair_data_grouped[study_id] = {}
        if series_id not in pair_data_grouped[study_id]:
            pair_data_grouped[study_id][series_id] = []
        slice_with_sop_id = slice.copy()
        slice_with_sop_id["sop_id"] = sop_id
        pair_data_grouped[study_id][series_id].append(slice_with_sop_id)

    # About the ordering...
    # We are going to read the original dicom files with GDCM via itk
    for study_id, series_dict in pair_data_grouped.items():
        for series_id, slices in series_dict.items():
            # Read the dicom files
            dicom_dir = Path(slices[0]["image"]).parent
            (
                dicom_names_with_uid_ordered,
                metadict_volume,
            ) = get_dicom_names_ordered_and_metadata(dicom_dir)

            global_labels_dict = populate_global_labels_dict(
                global_annotations_dict=global_annotations_dict,
                study_id=study_id,
                series_id=series_id,
            )

            # Read the dicom files with itk SeriesReader
            filenames_ordered = [name for name, _ in dicom_names_with_uid_ordered]
            reader = (
                itk.ImageSeriesReader.New(  # pyright: ignore[reportGeneralTypeIssues]
                    FileNames=filenames_ordered, ForceOrthogonalDirection=False
                )
            )
            reader.Update()
            dicom_volume = reader.GetOutput()
            output_case_parent_folder = Path(volumes_path) / study_id / series_id
            output_case_parent_folder.mkdir(parents=True, exist_ok=True)
            if process_grayscale:
                output_dicom_volume_path = output_case_parent_folder / "image.nrrd"
                # Append metadata to the dicom_volume
                itk_metadict = dicom_volume.GetMetaDataDictionary()
                for tagkey, tagvalue in metadict_volume.items():
                    try:
                        itk_metadict.Set(
                            str(tagkey),
                            itk.MetaDataObject[  # pyright: ignore[reportGeneralTypeIssues]
                                str
                            ].New(
                                MetaDataObjectValue=str(tagvalue)
                            ),
                        )
                    except TypeError as e:
                        logger.error(
                            f"Could not set/encode metadata with key {tagkey}. Skipping this key.\n{e}"
                        )
                # Write the volume
                itk.imwrite(dicom_volume, output_dicom_volume_path)
                # Save the metadata in json
                metadict_volume_output_path = (
                    output_case_parent_folder / "volume_metadata.json"
                )
                with open(metadict_volume_output_path, "w") as f:
                    json.dump(metadict_volume, f, indent=2)

            # Ok, now we have the dicom volume and the mapping for ordered SOPInstanceUID
            # We now need to create a 3D mask with the same shape as the dicom volume
            # And populate it with the masks from the slices, after ordering them using the SOPInstanceUID order from the dicom_named_ordered

            ordered_sop_ids = [sop_id for _, sop_id in dicom_names_with_uid_ordered]

            # Create the 3D mask
            for label in labels:
                label_volume_np = np.zeros(dicom_volume.shape, dtype=np.uint8)
                for slice in slices:
                    # Read the mask
                    mask_path = slice.get(label)
                    if mask_path is None:
                        raise ValueError(
                            f"Could not find label {label} in slice {slice}"
                        )
                    label_image = itk.imread(mask_path)
                    # Assert image is 3D.
                    if label_image.GetImageDimension() != 3:
                        raise ValueError(
                            f"Expected that the mask is a 3D image, with Z=1 slice, found {label_image.GetImageDimension()}"
                        )
                    # Get the SOPInstanceUID
                    hash_id = slice["study__series__sop_ids"]
                    _, _, sop_id = hash_id.split("__")
                    # Get the index of the SOPInstanceUID in the ordered list
                    sop_index = ordered_sop_ids.index(sop_id)
                    label_image_np = itk.array_from_image(
                        label_image
                    )  # The label is Z=1, Y, X
                    # Populate the 3D mask slice
                    label_volume_np[sop_index] = label_image_np.squeeze()

                # Convert to itk image
                label_volume = itk.image_from_array(label_volume_np)
                label_volume.SetOrigin(dicom_volume.GetOrigin())
                label_volume.SetSpacing(dicom_volume.GetSpacing())
                label_volume.SetDirection(dicom_volume.GetDirection())

                # Add global annotations to the metadata
                itk_label_metadict = label_volume.GetMetaDataDictionary()
                for tagkey, tagvalue in global_labels_dict.items():
                    itk_label_metadict.Set(
                        str(tagkey),
                        itk.MetaDataObject[  # pyright: ignore[reportGeneralTypeIssues]
                            str
                        ].New(MetaDataObjectValue=str(tagvalue)),
                    )

                # Check label and image have the same shape
                label_volume_size = np.array(label_volume.shape)
                image_volume_size = np.array(dicom_volume.shape)
                if not np.all(label_volume_size == image_volume_size):
                    raise ValueError(
                        f"Label and image have different shapes: {label_volume_size} != {image_volume_size}"
                    )

                # Write the label volume
                output_mask_path = output_case_parent_folder / f"{label}.nrrd"
                itk.imwrite(label_volume, output_mask_path)

            # Also write a json with metadata (shared with all labels)
            if labels and global_labels_dict:
                label_group_id = global_labels_dict["mdai_label_group_ids"][0]
                metadict_label_volume_output_path = (
                    output_case_parent_folder / f"global_labels_{label_group_id}.json"
                )
                with open(metadict_label_volume_output_path, "w") as f:
                    json.dump(global_labels_dict, f, indent=2)


def main_create_volumes(
    labels_parent_folder,
    volumes_path,
    mdai_annotations,
    labels,
    create_volumes,
    match_folder,
):
    pair_data_json_file = Path(labels_parent_folder) / "pair_data.json"
    volumes_path = Path(volumes_path)
    if not volumes_path.exists():
        logging.info("Creating volumes_path: {}".format(volumes_path))
        volumes_path.mkdir(parents=True, exist_ok=True)
    process_grayscale = create_volumes in ["all", "grayscale"]
    if process_grayscale and match_folder is None:
        # Skip grayscale if we don't have the dicoms
        process_grayscale = False

    global_annotations_dict, _, _ = get_global_annotations(mdai_annotations)
    merge_slices_into3D(
        pair_data_json_file=pair_data_json_file,
        labels=labels,
        volumes_path=volumes_path,
        process_grayscale=process_grayscale,
        global_annotations_dict=global_annotations_dict,
    )


def main(args):
    logger.info(args)

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

    download_dicoms = args.download_dicoms or not parameters.get(
        "mdai_annotations_only", True
    )
    mdai_annotations_only = not download_dicoms
    no_fixing_metadata = args.no_fixing_metadata or parameters.get(
        "mdai_no_fixing_metadata", False
    )
    series_only_annotations = args.series_only_annotations or parameters.get(
        "series_only_annotations", False
    )
    labels = args.labels or parameters.get("labels", [])
    create_volumes = args.create_volumes or parameters.get("create_volumes", None)
    volumes_path = args.volumes_path or parameters.get("volumes_path", None)
    # Check create_volumes is valid:
    valid_create_volumes = ["all", "grayscale", "mask", "none", None]
    if create_volumes not in valid_create_volumes:
        raise ValueError(
            f"Invalid value for --create_volumes: {create_volumes}. Valid values are: {valid_create_volumes}"
        )

    if create_volumes is not None and create_volumes != "none" and volumes_path is None:
        raise ValueError(
            f"You must provide --volumes_path if --create_volumes is set to {create_volumes}"
        )

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
    only_merge_3d_slices = args.only_merge_3d_slices or parameters.get(
        "only_merge_3d_slices", False
    )
    already_existing_segmentation_folder = (
        args.existing_segmentation_folder
        or parameters.get("existing_segmentation_folder", None)
    )
    if only_merge_3d_slices:
        no_download = True

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
        if not mdai_annotations_only:
            logger.info("Downloaded dicoms and annotations. Merging all parts...")
            _part_common = f"project_{mdai_project_id}_images_dataset_{mdai_dataset_id}"
            part_folder = list(Path(out_folder).glob(f"*{_part_common}*part_*of*"))
            if len(part_folder) == 0:
                logger.info(
                    f"Could not find any part folder in {out_folder} matching {_part_common}. Maybe the dataset is small enough."
                )
            else:
                merge_downloaded_parts(
                    part_folder=part_folder[0],
                    remove_zip_part_file=False,
                )
                logger.info("Done merging all parts. zip files were not removed.")

    # Get the json for annotations
    last_json_file = get_last_json_file(out_folder, match_str=mdai_dataset_id)

    logger.info(f"Last json file: {last_json_file}")

    # And get the folder where dicoms are, we use the match, because we are not always downloading data
    match_folder = match_folder_to_json_file(
        last_json_file, folder_to_search=out_folder, hint_str=mdai_dataset_id
    )
    logger.info(f"Matching data folder (dicoms): {match_folder or 'None'}")

    now = datetime.datetime.utcnow()
    now_formatted_date = now.strftime("%Y-%m-%d-%H%M%S")
    labels_parent_folder = (
        Path(out_folder)
        / f"{Path(last_json_file).stem}_{LABELS_FOLDER_IDENTIFIER}_{now_formatted_date}"
    )
    if already_existing_segmentation_folder:
        labels_parent_folder = Path(already_existing_segmentation_folder)
        logging.info(
            f"Using already existing segmentation folder: {labels_parent_folder}"
        )
        # Check that the folder exists
        if not labels_parent_folder.exists():
            raise FileNotFoundError(f"Could not find the folder {labels_parent_folder}")
    else:
        labels_parent_folder.mkdir(parents=True, exist_ok=True)

    pair_data_json_file = labels_parent_folder / "pair_data.json"
    pair_data = []

    result = mdai.common_utils.json_to_dataframe(last_json_file)

    # The following variables are all pandas
    # mdai_labels = result["labels"]
    # mdai_studies = result["studies"]
    mdai_annotations = result["annotations"]
    try:
        hash_columns = [
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "SOPInstanceUID",
        ]
        dh = mdai_annotations.groupby(hash_columns)
    except KeyError:
        logger.warning("Annotations do not have SOPInstanceUID. Using only two columns")
        hash_columns = [
            "StudyInstanceUID",
            "SeriesInstanceUID",
        ]
        dh = mdai_annotations.groupby(hash_columns)

    if series_only_annotations and volumes_path is not None:
        logger.info(
            f"series_only_annotations is set to True. Creating global_labels_$label_group_id.json files in {volumes_path} for each serie and study already existing from other group labels."
        )
        global_annotations_dict, _, _ = get_global_annotations(mdai_annotations)
        for hash_, group in dh:
            if len(hash_) != 2:
                raise ValueError(
                    f"Found more than two columns in the hash: {hash_}. Expected {hash_columns} without SOPInstanceUID"
                )
            study_id, series_id = hash_
            global_labels_dict = populate_global_labels_dict(
                global_annotations_dict=global_annotations_dict,
                study_id=str(study_id),
                series_id=str(series_id),
            )

            # Write
            output_case_parent_folder = Path(volumes_path) / study_id / series_id
            if not output_case_parent_folder.exists():
                logger.debug(f"Skipping folder: {output_case_parent_folder}")
                continue

            label_group_id = global_labels_dict["mdai_label_group_ids"][0]
            metadict_label_volume_output_path = (
                output_case_parent_folder / f"global_labels_{label_group_id}.json"
            )
            with open(metadict_label_volume_output_path, "w") as f:
                json.dump(global_labels_dict, f, indent=2)

        logger.info("Done writing global_labels_$label_group_id.json files.")
        return

    if only_merge_3d_slices:
        main_create_volumes(
            labels_parent_folder=labels_parent_folder,
            volumes_path=volumes_path,
            mdai_annotations=mdai_annotations,
            labels=labels,
            create_volumes=create_volumes,
            match_folder=match_folder,
        )
        return

    for hash_, group in dh:
        sep = "__"
        hash_id = sep.join(hash_)
        pair_data_entry = {}
        pair_data_entry["study__series__sop_ids"] = hash_id

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
            try:
                raw_slice_image = itk.imread(raw_slice_path)
            except RuntimeError as e:
                logger.warning(
                    f"Could not read dicom file: {raw_slice_path}.\n{e}\nIt might be an invalid json data from mdai. Safe to ignore."
                )
                continue

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
            label_name = (
                mdai_label_ids.inverse.get(row_label_id, False)
                or row.get("labelName", False)
                or "noname"
            )
            label_file = labels_parent_folder / f"{label_name}__{hash_id}.nrrd"
            # Check we are not introducing two labels:
            if label_name in pair_data_entry:
                old_label_file = Path(pair_data_entry[label_name])
                old_label_image = itk.imread(old_label_file)
                old_label_image_np = itk.array_from_image(old_label_image)
                # Merge both np arrays: old_label_image_np and slice_label_mask
                merged_label_image_np = np.maximum(old_label_image_np, slice_label_mask)
                # This is not always an error, there could be two unconected masks in the same slice.
                logger.warning(
                    f"Found labels with the same name {label_name} in the same slice {hash_id}. Merging into one."
                )
                # overwrite the label_image with the merged one
                label_image = itk.image_from_array(merged_label_image_np)

            pair_data_entry[label_name] = str(label_file.resolve())

            if not no_fixing_metadata:
                assert raw_slice_image is not None
                label_image.SetOrigin(raw_slice_image.GetOrigin())
                label_image.SetSpacing(raw_slice_image.GetSpacing())
                label_image.SetDirection(raw_slice_image.GetDirection())

            itk.imwrite(label_image, str(label_file))

        # Check that pair_data_entry contains at least one label
        if len(pair_data_entry) < 3:
            continue
        pair_data.append(pair_data_entry)

    with open(pair_data_json_file, "w") as f:
        json.dump(pair_data, f, indent=2)

    logging.info(f"pair_data_folder: {labels_parent_folder}")

    if create_volumes is not None and create_volumes != "none":
        main_create_volumes(
            labels_parent_folder=labels_parent_folder,
            volumes_path=volumes_path,
            mdai_annotations=mdai_annotations,
            labels=labels,
            create_volumes=create_volumes,
            match_folder=match_folder,
        )


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
        "--create_volumes",
        type=str,
        default=None,
        help="""Create 3D volumes from dicoms. Set to : 'all', 'grayscale', 'masks', 'none' or None. Default: None.
        Set --volumes_path to specify the folder where to save the volumes.
        """,
    )

    parser.add_argument(
        "--volumes_path",
        type=str,
        default=None,
        help="""Path to the folder where to save the volumes.
        Required if --create_volumes is set.
        """,
    )

    parser.add_argument(
        "--only_merge_3d_slices",
        action="store_true",
        help="If set, this will only merge the 3D slices into a volume, without downloading or processing dicoms or annotations.",
    )
    parser.add_argument(
        "--existing_segmentation_folder",
        type=str,
        default=None,
        help="Select the already existing segmentation folder. Only needed when --only_merge_3d_slices is set.",
    )
    parser.add_argument(
        "--series_only_annotations",
        action="store_true",
        help="Set to true for label groups with no image data, only series and study level annotations",
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
    logger = set_dual_logger("download_annotations", logs_dir="./logs", verbose=True)

    parser = get_download_parser()
    args = parser.parse_args()
    main(args)
