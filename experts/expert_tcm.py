from experts.base_expert import BaseExpert
import re
import torch
import tcm
import nibabel as nib
import numpy as np
import os
import tempfile
import urllib.request
from monai.transforms import (
    Compose, NormalizeIntensityD, ToTensorD, SpatialPadd, CenterSpatialCropD
)
from scipy.ndimage import center_of_mass

LABEL_MAP = {"Glioma": 0, "Lymphoma": 1, "Metastasis": 2}


class ExpertTCM(BaseExpert):
    """Expert model for the TorchXRayVision model."""

    def __init__(self) -> None:
        """Initialize the CXR expert model."""
        self.model_name = "TCM"
        self.pt_path = "/home/hufsaim/VLM/TCM/output/best_model.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model Loading
        self.model = tcm.model.load_model(
            layers=[2, 2, 2, 2],
            device=self.device,
        )
        self.model.load_state_dict(torch.load(self.pt_path))

    def classification_to_string(self, pred):
        """Format the classification outputs to a string."""

        inv_label_map = {v: k for k, v in LABEL_MAP.items()}
        label_name = inv_label_map.get(pred.item(), "Unknown")

        return f"The predicted tumor type is: {label_name}.\n"

    def mentioned_by(self, input: str):
        """
        Check if the CXR model is mentioned in the input string.

        Args:
            input (str): Text from the LLM, e.g. "Let me trigger <TCM>."

        Returns:
            bool: True if the CXR model is mentioned, False otherwise.
        """
        matches = re.findall(r"<(.*?)>", str(input))
        if len(matches) != 1:
            return False
        return self.model_name in str(matches[0])
    
    def get_transform(self):
        keys = ["image"]
        return Compose([
            NormalizeIntensityD(keys=keys, nonzero=True, channel_wise=True),
            SpatialPadd(keys=keys, spatial_size=(224, 224)),
            CenterSpatialCropD(keys=keys, roi_size=(224, 224)),
            ToTensorD(keys=["image"])
        ])

    def preprocess_multimodal_case(self, image_paths: list[str]):
        def find_file(include_keywords, exclude_keywords=None):
            for f in image_paths:
                name = f.upper()
                if all(k in name for k in include_keywords) and name.endswith(".NII.GZ"):
                    if exclude_keywords is None or all(k not in name for k in exclude_keywords):
                        return f
            return None

        # Find modality files
        t1_path    = find_file(["T1"], exclude_keywords=["GD", "CE"])
        t1gd_path  = find_file(["T1GD"]) or find_file(["WB"])
        t2_path    = find_file(["T2"])
        flair_path = find_file(["FLAIR"])
        label_path = find_file(["LABEL"])

        if not all([t1_path, t1gd_path, t2_path, flair_path]):
            raise FileNotFoundError(f"Missing modality in {image_paths}")
        
        if label_path:
            label_vol = nib.load(label_path).get_fdata()

            if len(np.argwhere(label_vol > 0)) == 0:
                center_z = label_vol.shape[2] // 2
            else:
                _, _, center_z = center_of_mass(label_vol)
            
            slice_idx = int(center_z)
        else:
            # Determine mid-slice from one modality (e.g., T1)
            t1_vol = nib.load(t1_path).get_fdata()
            t1_vol = np.nan_to_num(t1_vol)
            slice_idx = np.argmax(np.std(t1_vol, axis=(0, 1)))  # highest variance slice

        def load_slice(p):
            img_vol = nib.load(p).get_fdata(dtype=np.float32)
            return np.expand_dims(img_vol[:, :, slice_idx], axis=0)

        stacked = np.concatenate([
            load_slice(t1_path),
            load_slice(t1gd_path),
            load_slice(t2_path),
            load_slice(flair_path)
        ], axis=0)  # shape: (4, H, W)

        # Wrap in dict and apply transform
        data = {"image": stacked}
        transform = self.get_transform()
        data = transform(data)

        return data["image"]  # Tensor shape: (4, 224, 224)

    def run(
            self, 
            img_file: list[str] | None = None,
            image_url: str = "", 
            prompt: str = "", 
            **kwargs
        ):
        """
        Run the CXR model to classify the image.

        Args:
            image_url (str): The image URL.
            prompt: the original prompt.

        Returns:
            tuple: The classification string, file path, and the next step instruction.
        """
        # Data Preprocessing
        img = self.preprocess_multimodal_case(img_file)

        # Inference
        self.model.eval() 
        with torch.no_grad():
            img = img.unsqueeze(0).to(self.device)

            outputs = self.model(img)

            prob = torch.softmax(outputs, dim=1)
            pred = torch.argmax(prob, dim=1)

        return (
            self.classification_to_string(pred),
            None,
            "Use this result to respond to this prompt:\n" + prompt,
        )
    
if __name__ == "__main__":
    model = ExpertTCM()
    output = '<TCM()>'

    image_path = ["/home/hufsaim/VLM/VLM/data/TumorClassification/test/Glioma/1326521/1326521_T1.nii.gz",
                  "/home/hufsaim/VLM/VLM/data/TumorClassification/test/Glioma/1326521/1326521_T2.nii.gz",
                  "/home/hufsaim/VLM/VLM/data/TumorClassification/test/Glioma/1326521/1326521_T1GD.nii.gz",
                  "/home/hufsaim/VLM/VLM/data/TumorClassification/test/Glioma/1326521/1326521_FLAIR.nii.gz",
                  "/home/hufsaim/VLM/VLM/data/TumorClassification/test/Glioma/1326521/1326521_label.nii.gz"]
    prompt = 'Does the tumor shown in this image appear to be a glioma, lymphoma, or metastasis?'

    expert_output, _, inst = model.run(
        img_file=image_path,
        prompt=prompt
    )
    print(expert_output, inst)