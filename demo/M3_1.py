import torch
import os
import nibabel as nib
import numpy as np
from PIL import Image
from experts.expert_monai_brats import ExpertBrats
from experts.expert_monai_vista3d import ExpertVista3D
from experts.expert_torchxrayvision import ExpertTXRV
from llava.model.builder import load_pretrained_model
from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token
from llava.conversation import conv_templates, SeparatorStyle


CT_Sample_1 = "/home/hufsaim/VLM/VLM/m3/demo/example_data/ct_sample_0.nii.gz",
CT_Sample_2 = "/home/hufsaim/VLM/VLM/m3/demo/example_data/ct_liver_0.nii.gz",
MRI_Sample = [
        "/home/hufsaim/VLM/VLM/m3/demo/example_data/mri_Brats18_2013_31_1_t1.nii.gz",
        "/home/hufsaim/VLM/VLM/m3/demo/example_data/mri_Brats18_2013_31_1_t1ce.nii.gz",
        "/home/hufsaim/VLM/VLM/m3/demo/example_data/mri_Brats18_2013_31_1_t2.nii.gz",
        "/home/hufsaim/VLM/VLM/m3/demo/example_data/mri_Brats18_2013_31_1_flair.nii.gz",
]
Chest_X_ray_Sample_1 = "/home/hufsaim/VLM/VLM/m3/demo/example_data/cxr_00026451_030.jpg",
Chest_X_ray_Sample_2 = "/home/hufsaim/VLM/VLM/m3/demo/example_data/cxr_00029943_005.jpg",

EXAMPLE_PROMPTS_3D = [
    ["Segment the visceral structures in the current image."],
    ["Can you identify any liver masses or tumors?"],
    ["Segment the entire image."],
    ["What's in the scan?"],
    ["Segment the muscular structures in this image."],
    ["Could you please isolate the cardiovascular system in this image?"],
    ["Separate the gastrointestinal region from the surrounding tissue in this image."],
    ["Can you assist me in segmenting the bony structures in this image?"],
    ["Describe the image in detail"],
    ["Segment the image using BRATS"],
]

EXAMPLE_PROMPTS_2D = [
    ["What abnormalities are seen in this image?"],
    ["Is there evidence of edema in this image?"],
    ["Is there pneumothorax?"],
    ["What type is the lung opacity?"],
    ["Which view is this image taken?"],
    ["Is there evidence of cardiomegaly in this image?"],
    ["Is the atelectasis located on the left side or right side?"],
    ["What level is the cardiomegaly?"],
    ["Describe the image in detail"],
]


MODEL_CARDS = "Here is a list of available expert models:\n<BRATS(args)> Modality: MRI, Task: segmentation, Overview: A pre-trained model for volumetric (3D) segmentation of brain tumor subregions from multimodal MRIs based on BraTS 2018 data, Accuracy: Tumor core (TC): 0.8559 - Whole tumor (WT): 0.9026 - Enhancing tumor (ET): 0.7905 - Average: 0.8518, Valid args are: None\n<VISTA3D(args)> Modality: CT, Task: segmentation, Overview: domain-specialized interactive foundation model developed for segmenting and annotating human anatomies with precision, Accuracy: 127 organs: 0.792 Dice on average, Valid args are: 'everything', 'hepatic tumor', 'pancreatic tumor', 'lung tumor', 'bone lesion', 'organs', 'cardiovascular', 'gastrointestinal', 'skeleton', or 'muscles'\n<VISTA2D(args)> Modality: cell imaging, Task: segmentation, Overview: model for cell segmentation, which was trained on a variety of cell imaging outputs, including brightfield, phase-contrast, fluorescence, confocal, or electron microscopy, Accuracy: Good accuracy across several cell imaging datasets, Valid args are: None\n<CXR(args)> Modality: chest x-ray (CXR), Task: classification, Overview: pre-trained model which are trained on large cohorts of data, Accuracy: Good accuracy across several diverse chest x-rays datasets, Valid args are: None\nGive the model <NAME(args)> when selecting a suitable expert model.\n"

def load_nifti_image(nifti_path, slice_axis=2):
    nifti_img = nib.load(nifti_path)
    volume = nifti_img.get_fdata()

    slice_idx = volume.shape[slice_axis] // 2
    if slice_axis == 0:
        slice_img = volume[slice_idx, :, :]
    elif slice_axis == 1:
        slice_img = volume[:, slice_idx, :]
    else:
        slice_img = volume[:, :, slice_idx]

    slice_norm = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img))
    slice_image = Image.fromarray(np.uint8(slice_norm * 255)).convert('RGB')

    return slice_image

class M3Inference:
    def __init__(self, model_path="MONAI/Llama3-VILA-M3-8B", conv_mode="llama_3"):
        model_name = model_path.split("/")[-1]
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(model_path, model_name)
        self.conv_mode = conv_mode

    def inference(self, 
        image_path, 
        prompt,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ):
        answer = []
        conv = conv_templates[self.conv_mode].copy()

        answer.append({"USER" : prompt})

        prompt_text = conv.get_prompt()

        if isinstance(image_path, list):
            images = []
            for img_path in image_path:
                if img_path.endswith(('.nii', '.nii.gz')):
                    image = load_nifti_image(img_path)
                else:
                    image = Image.open(img_path).convert('RGB')
                images.append(image)
            images_tensor = process_images(images, self.image_processor, self.model.config).to(self.model.device, dtype=torch.float16)

            image_tokens = " ".join(["<image>"] * len(images))
            full_prompt = f"{MODEL_CARDS}\n{image_tokens}\n{prompt}"
        else: 
            if image_path.endswith(('.nii', '.nii.gz')):
                image = load_nifti_image(image_path)
            else:
                image = Image.open(image_path).convert('RGB')
            images_tensor = process_images([image], self.image_processor, self.model.config).to(self.model.device, dtype=torch.float16)

            full_prompt = f"{MODEL_CARDS}\n<image>\n{prompt}"

        media_input = {"image": [img for img in images_tensor]} if images_tensor is not None else None
        media_config = {"image": {}} if images_tensor is not None else {}

        conv.append_message(conv.roles[0], full_prompt)
        conv.append_message(conv.roles[1], "")

        prompt_text = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt_text, self.tokenizer, return_tensors="pt").unsqueeze(0).to(self.model.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                media=media_input,
                media_config=media_config,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                pad_token_id=self.tokenizer.eos_token_id,
            )

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if output.endswith(stop_str):
            output = output[:-len(stop_str)].strip()

        answer.append({"VILA-M3" : output})
        
        expert_model = None
        for expert_cls in [ExpertBrats, ExpertVista3D, ExpertTXRV]:
            expert = expert_cls()
            if expert.mentioned_by(output):
                expert_model = expert
                break

        expert_response, expert_image = None, None
        if expert_model:
            try:
                if isinstance(image_path, list):
                    expert_img_file = image_path
                else:
                    expert_img_file = [image_path]

                slice_index = 77

                if all(f.endswith((".nii", ".nii.gz")) for f in expert_img_file):
                    try:
                        nib_img = nib.load(expert_img_file[0])
                        shape = nib_img.shape
                        if len(shape) == 3:
                            slice_index = shape[2] // 2 
                            print(f"[DEBUG] Auto-calculated slice_index: {slice_index}")
                        else:
                            print(f"[WARNING] Unexpected shape {shape} for NIfTI file.")
                    except Exception as e:
                        print(f"[WARNING] Failed to load NIfTI file for slice index: {e}")
                        slice_index = 77  # fallback
                
                expert_response, expert_image, instruction = expert_model.run(
                    image_url=expert_img_file,
                    input=output,
                    output_dir="/home/hufsaim/VLM/VLM/m3/demo/expert_result",
                    img_file=expert_img_file,
                    slice_index=slice_index,
                    prompt=prompt,
                )
                answer.append({"Expert" : expert_response})
                answer.append({"Expert Image" : expert_image})
                if instruction:
                    conv = conv_templates[self.conv_mode].copy()
                    image_tokens = "\n".join(["<image>"] * (len(image_path) if isinstance(image_path, list) else 1))
                    updated_prompt = f"{expert_response}\n{instruction}\n{image_tokens}"
                    conv.append_message(conv.roles[0], updated_prompt)
                    conv.append_message(conv.roles[1], "")
                    updated_prompt_text = conv.get_prompt()

                    answer.append({"Expert": instruction})

                    input_ids = tokenizer_image_token(updated_prompt_text, self.tokenizer, return_tensors="pt").unsqueeze(0).to(self.model.device)

                    with torch.inference_mode():
                        updated_output_ids = self.model.generate(
                            input_ids,
                            media=media_input,
                            media_config=media_config,
                            do_sample=True if temperature > 0 else False,
                            temperature=temperature,
                            top_p=top_p,
                            max_new_tokens=max_tokens,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria],
                            pad_token_id=self.tokenizer.eos_token_id,
                        )
                    output = self.tokenizer.batch_decode(updated_output_ids, skip_special_tokens=True)[0].strip()
                    if output.endswith(stop_str):
                        output = output[:-len(stop_str)].strip()

                    answer.append({"VILA-M3": output})

            except Exception as e:
                expert_response = f"Expert model encountered an error: {e}"

        return answer

if __name__ == "__main__":
    inference_model = M3Inference()

    image_paths = MRI_Sample
    prompt = EXAMPLE_PROMPTS_3D[2]

    results = inference_model.inference(image_paths, prompt)

    for result in results:
        for role, content in result.items():
            if role == "Expert Image":
                print(f"{role} Path: {content}")
            else:
                print(f"{role}: {content}")