# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#    Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import re
import tempfile
from pathlib import Path
from shutil import move
from uuid import uuid4

import requests
from .base_expert import BaseExpert
from .utils import get_monai_transforms, get_slice_filenames
from monai.bundle import create_workflow

from hd_glio.utils import blockPrint, enablePrint
blockPrint()
from nnunet.inference.predict import predict_cases
enablePrint()
from hd_glio.paths import folder_with_parameter_files
from hd_glio.setup_hd_glio import maybe_download_weights


class ExpertHDGLIO(BaseExpert):
    """Expert model for BRATS."""

    def __init__(self) -> None:
        """Initialize the VISTA-3D expert model."""
        self.model_name = "HD-GLIO"

    def segmentation_to_string(
        self,
        output_dir: Path,
        img_file: str,
        seg_file: str,
        slice_index: int,
        image_filename: str,
        label_filename: str,
        modality: str = "MRI",
        axis: int = 2,
        output_prefix="The results are <image>. The colors in this image describe\n",
    ):
        """Convert the segmentation to a string."""
        output_dir = Path(output_dir)

        transforms = get_monai_transforms(
            ["image", "label"],
            output_dir,
            modality=modality,
            slice_index=slice_index,
            axis=axis,
            image_filename=image_filename,
            label_filename=label_filename,
        )
        data = transforms({"image": img_file, "label": seg_file})
        ncr = data["colormap"].get(1, None)
        ed = data["colormap"].get(2, None)
        et = data["colormap"].get(4, None)
        output = output_prefix
        if ncr is not None and et is not None:
            output += f"{ncr} and {et}: tumor core, "
        if et is not None:
            output += f"only {et}: enhancing tumor, "
        if ncr is not None or et is not None or ed is not None:
            output += "all colors: whole tumor\n"
        output += f"{ncr} (Non-enhancing Core), {et} (Enhancing Tumor), {ed} (Edema)"
        return output

    def download_file(self, url: str, img_file: str):
        """
        Download the file from the URL.

        Args:
            url (str): The URL.
            img_file (str): The file path.
        """
        parent_dir = os.path.dirname(img_file)
        os.makedirs(parent_dir, exist_ok=True)
        with open(img_file, "wb") as f:
            response = requests.get(url)
            f.write(response.content)

    def mentioned_by(self, input: str):
        matches = re.findall(r"<(.*?)>", str(input))
        print(matches)
        if len(matches) != 1:
            return False
        return self.model_name in str(matches[0])
    
    import os

    def standardize_modality_dict(self, img_paths: list[str]) -> dict:
        """
        Automatically detect modality type from filenames and return a dict of modality → file path.

        Args:
            img_paths (list[str]): List of file paths (nii or nii.gz) containing T1, T1CE, T2, FLAIR modalities.

        Returns:
            dict: {"T1": ..., "T1CE": ..., "T2": ..., "FLAIR": ...}
        """
        modality_map = {}
        cnt = 0

        for path in img_paths:
            filename = os.path.basename(path).lower()

            if "t1ce" in filename or "t1gd" in filename:
                modality = "T1CE"
                print(f'{modality}, {filename}')
            elif "t1" in filename:
                modality = "T1"
                print(f'{modality}, {filename}')
            elif "t2" in filename:
                modality = "T2"
                print(f'{modality}, {filename}')
            elif "flair" in filename:
                modality = "FLAIR"
                print(f'{modality}, {filename}')
            else:
                modality = f"etc{cnt}"
                cnt += 1
                print(f'{modality}, {filename}')

            if modality in modality_map:
                raise ValueError(f"Duplicate modality detected: {modality} in path {path}")
            modality_map[modality] = path

        required_modalities = ["T1", "T1CE", "T2", "FLAIR"]
        for m in required_modalities:
            if m not in modality_map:
                raise ValueError(f"Missing modality: {m}")
        print(f'[DEBUG] {modality_map}')
        return modality_map

    def run(
        self, 
        img_file: list[str] | None = None,
        image_url: list[str] | None = None,
        input: str = "",
        output_dir: str = "",
        slice_index: int = 0,
        prompt: str = "",
        **kwargs
    ):
        print(f'[DEBUG] Expert Model Inference')
        if not img_file:
            print(f'[DEBUG] img_file None')
            img_dict = {}
            for url in image_url:
                basename = os.path.basename(url)
                lowername = basename.lower()
                file_path = os.path.join(output_dir, basename)
                self.download_file(url, file_path)

                if "t1ce" in lowername or "t1gd" in lowername:
                    modality = "T1CE"
                elif "t1" in lowername:
                    modality = "T1"
                elif "t2" in lowername:
                    modality = "T2"
                elif "flair" in lowername:
                    modality = "FLAIR"
                else:
                    modality = f"UNKNOWN_{len(img_dict)}"
                img_dict[modality] = file_path
        else:
            img_dict = self.standardize_modality_dict(img_file)
            print(f'[DEBUG] img_file processing\n{img_dict}')

        print(f'[DEBUG] {img_dict}')

        output_file = os.path.join(output_dir, 'hd_glio_seg.nii.gz')

        input_list = [img_dict[modality] for modality in ['T1', 'T1CE', 'T2', 'FLAIR']]

        maybe_download_weights()
        predict_cases(folder_with_parameter_files, [input_list], [output_file], (0, ), False, 1, 1, None, True,
                  None, True, checkpoint_name="model_final_checkpoint")
        
        seg_image = f"seg_{uuid4()}.jpg"
        text_output = self.segmentation_to_string(
            output_dir,
            img_file[0],
            output_file,
            slice_index,
            get_slice_filenames(img_file[0], slice_index),
            seg_image,
            modality="MRI",
            axis=2,
        )

        if "segmented" in input:
            instruction = ""  # no need to ask for instruction
        else:
            instruction = "Use this result to respond to this prompt:\n" + prompt
        return text_output, os.path.join(output_dir, seg_image), instruction
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Debug ExpertHDGLIO")
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Path to directory containing 4 modality files (T1, T1CE, T2, FLAIR)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output",
        help="Directory to save segmentation results and visualizations"
    )
    parser.add_argument(
        "--slice_index", type=int, default=80,
        help="Index of the slice to visualize"
    )
    args = parser.parse_args()

    # 입력 폴더 내 모든 NIfTI 파일 수집
    img_paths = sorted([
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ])

    # ExpertHDGLIO 인스턴스 생성
    expert = ExpertHDGLIO()

    # Segmentation 실행
    result_text, seg_image_path, instruction = expert.run(
        img_file=img_paths,
        output_dir=args.output_dir,
        slice_index=args.slice_index,
        prompt="Segment the tumor in this MRI scan.",
        input="segmented"
    )

    print("\n=== Segmentation Result ===")
    print(result_text)
    print(f"Segmentation image saved to: {seg_image_path}")
    print("===========================\n")