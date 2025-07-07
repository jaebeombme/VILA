import torch
import os
import json
import nibabel as nib
import numpy as np
from tqdm import tqdm
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
from llava.conversation import conv_templates, SeparatorStyle
from huggingface_hub import snapshot_download
from typing import Any, Dict, List, Optional, Tuple, Union

# MODEL_PATH="/home/hufsaim/.cache/huggingface/hub/models--MONAI--Llama3-VILA-M3-8B/snapshots/df60e0276e2ae10624c86dabe909847a03b2a5cb" # Pretrained VILA-M3
# MODEL_PATH="/home/hufsaim/VLM/VLM/m3/train/checkpoints/loramps/checkpoint-1200" # MPS
MODEL_PATH="/home/hufsaim/VLM/VLM/m3/train/checkpoints/lorattcc2/checkpoint-16" # TC
MODEL_BASE="/home/hufsaim/.cache/huggingface/hub/models--MONAI--Llama3-VILA-M3-8B/snapshots/df60e0276e2ae10624c86dabe909847a03b2a5cb"
SOURCE="local"

OUTPUT_PATH="/home/hufsaim/VLM/VLM/m3/demo/0704/wo_expert"
JSON_FILE="/home/hufsaim/VLM/VLM/m3/demo/0704/vila_infer_input_val.json"

BASE_FILE_NAME = "after_ft"

MODEL_CARDS = None

def get_largest_tumor_slice(label_path):
    label_data = nib.load(label_path).get_fdata()
    binary_mask = (label_data > 0).astype(np.uint8)
    z_sums = np.sum(binary_mask, axis=(0, 1))
    return int(np.argmax(z_sums))

def match_modalities_to_paths(image_paths: List[str]) -> Dict[str, str]:
    """
    Given a list of image paths and expected modality names,
    returns a dictionary mapping each modality name to its corresponding file path.
    
    Parameters:
        image_paths (List[str]): List of image file paths.
        modality_names (List[str]): List of modality keywords to match.
    
    Returns:
        Dict[str, str]: Dictionary mapping modality names to file paths.
    """
    modality_names = ['t1', 't1ce', 't2', 'flair', 'label', 'seg']
    modality_paths = {}
    for name in modality_names:
        matched = next(
            (p for p in image_paths if name in os.path.basename(p).lower()), None
        )
        if matched:
            modality_paths[name] = matched
    return modality_paths

def load_nifti_image_2d_tensor(
    modality_paths: Dict[str, str],
    process_images,  # image processor function
    image_processor,  # typically model.image_processor
    model_config,     # model.config
    device,            # target torch.device
    using_label: bool = True
):
    """
    Load 2D slices from multiple modalities (except label/seg),
    extract the largest tumor slice, convert to tensor using processor.
    """
    # Step 1: Find label path and tumor slice index
    label_path = next((v for k, v in modality_paths.items() if "label" in k.lower() or "seg" in k.lower()), None)
    if label_path is None:
        raise ValueError("No label/seg modality found in paths.")
    
    slice_index = get_largest_tumor_slice(label_path)

    # Step 2: Load slice for each modality (excluding label/seg)
    slice_images = []
    for name, path in modality_paths.items():
        if not using_label:
            if "label" in name.lower() or "seg" in name.lower():
                continue

        img = nib.load(path)
        data = img.get_fdata()

        try:
            slice_2d = data[slice_index]
        except IndexError:
            # 슬라이스 인덱스가 잘못되었을 경우 중앙 슬라이스 사용
            print(f"[Warning] slice_index {slice_index} is out of bounds for {name}, using center slice instead.")
            slice_index = data.shape[0] // 2
            slice_2d = data[slice_index]

        # Normalize and convert to PIL image
        slice_2d = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d) + 1e-6)
        slice_img = Image.fromarray((slice_2d * 255).astype(np.uint8)).convert("RGB")
        slice_images.append(slice_img)

    # Step 3: Convert to tensor using processor
    images_tensor = process_images(slice_images, image_processor, model_config).to(
        device=device, dtype=torch.float16
    )
    return images_tensor  # shape: [N, 3, H, W] or as defined by processor

def load_nifti_image(nifti_path, slice_axis=2):
    """
    Convert nifti file to png for use as input for VILA-M3 model
    """
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

def save_result_to_json(result, output_dir, base_filename=BASE_FILE_NAME):
    os.makedirs(output_dir, exist_ok=True)
    counter = 1

    filename = f"{base_filename}_{counter}.json"
    filepath = os.path.join(output_dir, filename)

    while os.path.exists(filepath):
        filename = f"{base_filename}_{counter}.json"
        filepath = os.path.join(output_dir, filename)
        counter += 1

    formatted_results = []

    for sample_id, info_list in result.items():
        combined = {"id": sample_id}
        for entry in info_list:
            combined.update(entry)
        formatted_results.append(combined)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(formatted_results, f, ensure_ascii=False, indent=4)

def load_json_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data
    
class M3Inference:
    def __init__(self, source=SOURCE, model_path=MODEL_PATH, conv_mode="llama_3"):
        self.source = source
        if source == "local" or source == "huggingface":
            # TODO: allow setting the device
            self.conv_mode = conv_mode
            if source == "huggingface":
                model_path = snapshot_download(model_path)
            model_name = get_model_name_from_path(model_path)
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path, model_name, MODEL_BASE
            )
        else:
            raise NotImplementedError(f"Source {source} is not supported.")

    def inference(self, 
        image_path, 
        prompt,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 0.9,
        output_path: str = None,
        **kwargs,
    ):  
        # print(f'[DEBUG] Inferce')
        answer = []
        conv = conv_templates[self.conv_mode].copy()

        answer.append({"USER" : prompt})
        answer.append({"GT" : kwargs['answer']})
        answer.append({"image path": image_path})

        prompt_text = conv.get_prompt()

        if isinstance(image_path, list):
            # ------ 2D 처리 (ImageEncoder용) -------
            modality_paths = match_modalities_to_paths(image_path)
            images_tensor = load_nifti_image_2d_tensor(
                modality_paths=modality_paths,
                process_images=process_images,
                image_processor=self.image_processor,
                model_config=self.model.config,
                device=self.model.device,
            )
            # print(f'[DEBUG] dtype image tensor : {images_tensor.dtype}')
            media_input = {"image": [img for img in images_tensor]}
            media_config = {"image": {}}

            num_image = len(media_input["image"])
            image_tokens = " ".join(["<image>"] * num_image)
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

        # VILA-M3 Model Inference
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

        return answer
    
def run_batch_inference(
        image_dir=None, 
        modality=None, 
        prompt="What type of brain tumor is present in this MRI image? And if there is a tumor in this image, segment the tumor area.",
        pairs=None,
        output_path=None,
        temperature=0.0,
        top_p=0.9,
    ):
    inference_result = {}
    inference_model = M3Inference(model_path=MODEL_PATH)

    if pairs:
        for pair in tqdm(pairs, desc="Processing Cases",leave=True):
            case_id = pair["id"]
            image_path = pair["image_path"]
            question = pair.get("question", prompt)
            answer = pair["answer"]
            # 단일 path도 리스트로 감쌈
            if isinstance(image_path, str):
                image_path = [image_path]
            inference_result[case_id] = inference_model.inference(
                image_path=image_path, 
                prompt=question, 
                output_path=output_path,
                temperature=temperature, 
                top_p=top_p,
                answer=answer
                )
    else:
        raise ValueError('[Error] Please enter the correct format for the JSON file.')

    return inference_result

if __name__ == '__main__':
    pairs = []
    if JSON_FILE:
        data = load_json_data(JSON_FILE)

        for sample in data:
            sample_id = sample["id"]
            image_path = sample["image_path"]
            question = sample["USER"]
            answer = sample["GT"]

            pairs.append({"id": sample_id, "image_path": image_path, "question": question, "answer": answer})
    else:
        raise ValueError("JSON_FILE path must be specified.")

    results = run_batch_inference(pairs=pairs)
    save_result_to_json(results, OUTPUT_PATH)