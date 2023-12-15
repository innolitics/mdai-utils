import re
import shutil
from pathlib import Path


def merge_downloaded_parts(part_folder, remove_zip_part_file=False):
    """
    md.ai splits the data into multiple parts (~50GB each). This function
    merges all the parts into one directory. A simple mv does not work because
    there are duplicated folders (with different content) in the parts.

    part_folder is the path to the folder containing the parts of the data.
    For example:
    mdai_uab_project_7YNdkRbz_images_dataset_D_vVO1L2_2023-12-14-121547_part21of21

    The output folder will be parallel to the part folder, without the _partXXofYY.
    """

    # Create a new directory with the dir string
    part_folder_path = Path(part_folder)
    out_folder_path = part_folder_path.parent
    dir_string = part_folder_path.name
    # Remove the part information from the dir string
    re_pattern = re.compile("_part[0-9]+of[0-9]+")
    dir_string = re.sub(re_pattern, "", dir_string)

    # Create a new directory with the dir string
    out_folder_path = part_folder_path.parent / dir_string
    out_folder_path.mkdir(exist_ok=True, parents=True)

    # Collect all parts folder, don't capture zip files
    all_parts = out_folder_path.parent.glob(dir_string + "_part*")
    all_parts = [part for part in all_parts if not part.name.endswith(".zip")]
    all_parts = sorted(all_parts)
    if len(all_parts) == 0:
        raise ValueError(f"No part folders matching {out_folder_path}")

    for part_folder in all_parts:
        # Move all files from the part folder to the out folder.
        print(f"Moving files from {part_folder}")
        for study in part_folder.glob("*"):
            for series in study.glob("*"):
                destination = out_folder_path / study.name / series.name
                shutil.move(str(series), str(destination))
            assert (
                len(list(study.glob("*"))) == 0
            ), f"Study folder is not empty: {study}. Aborting, do it manually for this part folder."
            shutil.rmtree(str(study))
        assert (
            len(list(part_folder.glob("*"))) == 0
        ), f"Part folder is not empty: {part_folder}. Aborting, do it manually for this part folder."
        shutil.rmtree(str(part_folder))
        if remove_zip_part_file:
            zip_file = part_folder.with_suffix(".zip")
            zip_file.unlink()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--part_folder", type=str, help="The path to one (any) part folder"
    )
    parser.add_argument(
        "--remove_zip_files",
        action="store_true",
        help="Remove the zip file containing the part. Default: False",
    )
    args = parser.parse_args()
    print(args)

    merge_downloaded_parts(args.part_folder, args.remove_zip_files)
