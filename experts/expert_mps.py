from experts.base_expert import BaseExpert
import re
import torch
import mps
import nibabel as nib
import numpy as np
import os
import tempfile
import urllib.request
from monai.transforms import (
    Compose, CenterSpatialCropd, ResizeD, NormalizeIntensityD, ToTensorD
)

LABEL_MAP = {"t1": 0, "t2": 1, "t1ce": 2, "flair": 3}


class ExpertMPS(BaseExpert):
    """Expert model for the TorchXRayVision model."""

    def __init__(self) -> None:
        """Initialize the CXR expert model."""
        self.model_name = "MPS"
        self.pt_path = "/home/hufsaim/VLM/MPS/model_params/0621/checkpoint_0.0803_0.9811.pt"
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        # Model Loading
        self.model = mps.model_arch_2d.load_model(
            layers=[2, 2, 2, 2],
            device=self.device,
        )
        pt_="/home/hufsaim/VLM/MPS/model_params/checkpoint_0.0803_0.9811.pt"
        self.model.load_state_dict(torch.load(pt_))

    def classification_to_string(self, pred):
        """Format the classification outputs to a string."""

        inv_label_map = {v: k.upper() for k, v in LABEL_MAP.items()}
        label_name = inv_label_map.get(pred.item(), "Unknown")

        if label_name == "T1CE":
            readable = "contrast-enhanced (T1CE)"
        else:
            readable = label_name

        return f"The predicted MRI sequence is: {readable}.\n"

    def mentioned_by(self, input: str):
        """
        Check if the CXR model is mentioned in the input string.

        Args:
            input (str): Text from the LLM, e.g. "Let me trigger <MPS>."

        Returns:
            bool: True if the CXR model is mentioned, False otherwise.
        """
        matches = re.findall(r"<(.*?)>", str(input))
        if len(matches) != 1:
            return False
        return self.model_name in str(matches[0])
    
    def get_single_image_transform(self):
        return Compose([
            CenterSpatialCropd(keys=["image"], roi_size=(128, 128)),
            ResizeD(keys=["image"], spatial_size=(224, 224)),
            NormalizeIntensityD(keys=["image"], nonzero=False, channel_wise=True),
            ToTensorD(keys=["image"])
        ])

    def preprocess_single_image(self, image_path: str):
        # Load image
        img = nib.load(image_path).get_fdata(dtype=np.float32)
        img = np.nan_to_num(img)

        # Mid-slice (Z-axis)
        slice_idx = np.argmax(np.std(img, axis=(0, 1)))
        img2d = img[:, :, slice_idx]
        img2d = img2d[None, :, :]  # add channel dim: (1, H, W)

        # Wrap in dict and apply transform
        data = {"image": img2d}
        transform = self.get_single_image_transform()
        data = transform(data)
        
        return data["image"]  # torch.Tensor (1, 224, 224)

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
        if img_file and os.path.exists(img_file[0]):
            img = self.preprocess_single_image(img_file[0]).unsqueeze(0)
        elif image_url.endswith(".nii") or image_url.endswith(".nii.gz"):
            with tempfile.NamedTemporaryFile(suffix=".nii.gz") as temp_file:
                urllib.request.urlretrieve(image_url, temp_file.name)
                img = self.preprocess_single_image(temp_file.name)
        else:
            raise ValueError("Provide either a valid local NIfTI file or a URL to a .nii/.nii.gz file.")

        # Inference
        self.model.eval() 
        with torch.no_grad():
            img = img.to(self.device)

            outputs = self.model(img)

            pred = outputs.argmax(dim=1)

        return (
            self.classification_to_string(pred),
            None,
            "Use this result to respond to this prompt:\n" + prompt,
        )
    
if __name__ == "__main__":
    model = ExpertMPS()
    output = '<MPS()>'
    print(model.mentioned_by(output))

    image_path = ["/home/hufsaim/VLM/VLM/data/tmp/1012729/1012729_T1GD_stripped.nii.gz"]
    prompt = 'What pulse sequence is this mri?'

    expert_output, _, inst = model.run(
        img_file=image_path,
        prompt=prompt
    )
    print(expert_output, inst)