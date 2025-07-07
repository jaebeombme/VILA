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
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
from experts.expert_monai_brats import ExpertBrats
from experts.expert_monai_vista3d import ExpertVista3D
from experts.expert_torchxrayvision import ExpertTXRV
from experts.expert_hd_glio import ExpertHDGLIO
from experts.expert_mps import ExpertMPS

# MODEL_PATH="/home/hufsaim/.cache/huggingface/hub/models--MONAI--Llama3-VILA-M3-8B/snapshots/df60e0276e2ae10624c86dabe909847a03b2a5cb"
MODEL_PATH="/home/hufsaim/VLM/VLM/m3/train/checkpoints/lora/checkpoint-1200"
MODEL_BASE="/home/hufsaim/.cache/huggingface/hub/models--MONAI--Llama3-VILA-M3-8B/snapshots/df60e0276e2ae10624c86dabe909847a03b2a5cb"
SOURCE="local"

OUTPUT_PATH = "/home/hufsaim/VLM/VLM/m3/demo/0701/result"
JSON_FILE = "/home/hufsaim/VLM/VLM/m3/demo/0701/TumorVolume.json"
SLICE_SAVE_PATH = "/home/hufsaim/VLM/VLM/m3/demo/sliced_images" 
DATA_ROOT_DIR=None

MODEL_CARDS = """Here is a list of available expert models:\n<BRATS(args)> Modality: MRI, Task: segmentation, Overview: A pre-trained model for volumetric (3D) segmentation of brain tumor subregions from multimodal MRIs based on BraTS 2018 data, Accuracy: Tumor core (TC): 0.8559 - Whole tumor (WT): 0.9026 - Enhancing tumor (ET): 0.7905 - Average: 0.8518, Valid args are: None\n<VISTA3D(args)> Modality: CT, Task: segmentation, Overview: domain-specialized interactive foundation model developed for segmenting and annotating human anatomies with precision, Accuracy: 127 organs: 0.792 Dice on average, Valid args are: 'everything', 'hepatic tumor', 'pancreatic tumor', 'lung tumor', 'bone lesion', 'organs', 'cardiovascular', 'gastrointestinal', 'skeleton', or 'muscles'\n<VISTA2D(args)> Modality: cell imaging, Task: segmentation, Overview: model for cell segmentation, which was trained on a variety of cell imaging outputs, including brightfield, phase-contrast, fluorescence, confocal, or electron microscopy, Accuracy: Good accuracy across several cell imaging datasets, Valid args are: None\n<CXR(args)> Modality: chest x-ray (CXR), Task: classification, Overview: pre-trained model which are trained on large cohorts of data, Accuracy: Good accuracy across several diverse chest x-rays datasets, Valid args are: None\n<MPS(args)>Modality: MRITask: classificationOverview: A deep learning-based classification model that predicts the MRI sequence type (e.g., T1, T2, FLAIR, T1CE, STIR, PD) from input MR images. It is designed to automatically annotate sequence labels for unlabeled MR scans, improving downstream processing and dataset curation. The model was trained on a diverse multi-institutional dataset covering common brain imaging protocols with varying acquisition parameters.Accuracy:T1: 94.2% T2: 95.8 %FLAIR: 93.7% T1CE: 92.1% Average: 92.7%Valid args are: None\n Give the model <NAME(args)> when selecting a suitable expert model.\n"""

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

def inject_media_tokens(prompt: str, tokenizer, media_input: dict):
    prefix_ids = []
    for media_type, items in media_input.items():
        token_id = tokenizer.media_token_ids[media_type]
        prefix_ids.extend([token_id] * len(items))
    encoded = tokenizer(prompt, return_tensors="pt").input_ids[0]
    injected = torch.cat([
        torch.tensor(prefix_ids, dtype=torch.long),
        encoded
    ])
    return injected.unsqueeze(0)

def collect_mri_paths(case_dir):
    """
    Function to adjust the input order when inputting MRI data to pass it to the BraTS model input
    """
    modalities = ['t1', 't1ce', 't2', 'flair']
    image_paths = []

    for modality in modalities:
        for ext in ('.nii', '.nii.gz'):
            fname = next((f for f in os.listdir(case_dir) if f.lower().endswith(ext) and f"{modality}." in f.lower()), None)
            gt_name = str(f for f in os.listdir(case_dir) if f.lower().endswith(ext) and f"{'seg'}." in f.lower())
            if fname:
                image_paths.append(os.path.join(case_dir, fname))
                break

    return image_paths if len(image_paths) == 4 else None, gt_name if gt_name else None

def resize_video_tensor(video_tensor: torch.Tensor, target_size: int = 384) -> torch.Tensor:
    """
    Resize each frame in the video tensor to [3, target_size, target_size].

    Args:
        video_tensor: Tensor of shape [M, T, 3, H, W]
        target_size: Target spatial dimension (default: 384 for SigLIP)

    Returns:
        Tensor of shape [M, T, 3, target_size, target_size]
    """
    M, T, C, H, W = video_tensor.shape
    video_tensor = video_tensor.view(M * T, C, H, W)
    video_tensor = F.interpolate(video_tensor, size=(target_size, target_size), mode="bilinear", align_corners=False)
    video_tensor = video_tensor.view(M, T, C, target_size, target_size)
    return video_tensor

def load_nifti_image_2d_tensor(
    modality_paths: Dict[str, str],
    process_images,  # image processor function
    image_processor,  # typically model.image_processor
    model_config,     # model.config
    device,            # target torch.device
    using_label: bool = False
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

def load_mri_volume(
    modality_paths: Dict[str, str],
    num_slices: int = 16,
    as_rgb: bool = True,
    stack_modality: bool = True,
    device: torch.device = "cpu",
    using_label: bool = False,
) -> torch.Tensor:
    all_modalities = []

    # Detect label path
    label_path = next((v for k, v in modality_paths.items() if "label" in k.lower() or "seg" in k.lower()), None)
    if label_path is None:
        raise ValueError("No label path found with 'label' or 'seg' in key.")
    slice_index = get_largest_tumor_slice(label_path)

    for name, path in modality_paths.items():
        if not using_label:
            if "label" in name.lower() or "seg" in name.lower():
                continue

        nii = nib.load(path).get_fdata()
        if nii.ndim == 4:
            nii = nii[0]

        D = nii.shape[0]
        start = max(0, slice_index - num_slices // 2)
        end = min(D, slice_index + num_slices // 2)
        slices = nii[start:end]

        slice_tensors = []
        for s in slices:
            s = (s - s.min()) / (s.max() - s.min() + 1e-6)
            img = Image.fromarray((s * 255).astype(np.uint8)).convert("RGB")
            img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).to(dtype=torch.float16) / 255.0
            slice_tensors.append(img_tensor)

        modality_tensor = torch.stack(slice_tensors, dim=0).to(device)  # [T, 3, H, W]
        all_modalities.append(modality_tensor)

    video_tensor = torch.stack(all_modalities, dim=0)  # [M, T, 3, H, W]
    video_tensor = resize_video_tensor(video_tensor, target_size=384)
    return video_tensor

def save_result_to_json(result, output_dir, base_filename="inference_result"):
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
        print(f'[DEBUG] results saved at {filepath}')

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
        dimension: str = "2D",
        key: bool = False,
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
            if image_path[0].endswith(('.nii', '.nii.gz')):
                modality_paths = match_modalities_to_paths(image_path)
                if dimension == "3D":
                    video_tensor = load_mri_volume(
                        modality_paths=modality_paths,
                        num_slices=8,
                        device=self.model.device,
                        using_label=key
                    )
                    media_input = {"video": [video_tensor[i] for i in range(video_tensor.shape[0])]}
                    media_config = {"video": {}}

                    num_videos = len(media_input["video"])
                    video_tokens = " ".join(["<video>"] * num_videos)
                    full_prompt = f"{MODEL_CARDS}\n{video_tokens}\n{prompt}"

                elif dimension == "2D":
                    # ------ 2D 처리 (ImageEncoder용) -------
                    images_tensor = load_nifti_image_2d_tensor(
                        modality_paths=modality_paths,
                        process_images=process_images,
                        image_processor=self.model.image_processor,
                        model_config=self.model.config,
                        device=self.model.device,
                        using_label=key
                    )
                    # print(f'[DEBUG] dtype image tensor : {images_tensor.dtype}')
                    media_input = {"image": [img for img in images_tensor]}
                    media_config = {"image": {}}

                    num_image = len(media_input["image"])
                    image_tokens = " ".join(["<image>"] * num_image)
                    full_prompt = f"{MODEL_CARDS}\n{image_tokens}\n{prompt}"

                else:
                    raise ValueError(f"Unknown dimension type: {dimension}")

            else:
                # 일반 RGB 이미지 처리
                image = Image.open(image_path).convert('RGB')
                images_tensor = process_images([image], self.image_processor, self.model.config).to(
                    self.model.device, dtype=torch.float16
                )
                media_input = {"image": [img for img in images_tensor]}
                media_config = {"image": {}}
                full_prompt = f"{MODEL_CARDS}\n<image>\n{prompt}"
                

        conv.append_message(conv.roles[0], full_prompt)
        conv.append_message(conv.roles[1], "")

        prompt_text = conv.get_prompt()

        if "video" in media_input:
            input_ids = inject_media_tokens(prompt_text, self.tokenizer, media_input).to(self.model.device)
        else:
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

        # # Check Tirrger
        # expert_model = None
        # for expert_cls in [ExpertBrats, ExpertVista3D, ExpertTXRV, ExpertHDGLIO, ExpertMPS]:
        #     expert = expert_cls()
        #     if expert.mentioned_by(output):
        #         expert_model = expert
        #         break
        
        # expert_response, expert_image = None, None
        # if expert_model:
        #     # print(f'[DEBUG] Expert Model Name : {expert_model.model_name}')
        #     try:
        #         slice_index = 0
        #         if isinstance(image_path, list):
        #             expert_img_file = image_path
        #         else:
        #             expert_img_file = [image_path]

        #         if all(f.endswith((".nii", ".nii.gz")) for f in expert_img_file):
        #             try:
        #                 nib_img = nib.load(expert_img_file[0])
        #                 shape = nib_img.shape
        #                 if len(shape) == 3:
        #                     slice_index = shape[2] // 2 
        #                     # print(f"[DEBUG] Auto-calculated slice_index: {slice_index}")
        #                 else:
        #                     print(f"[WARNING] Unexpected shape {shape} for NIfTI file.")
        #             except Exception as e:
        #                 print(f"[WARNING] Failed to load NIfTI file for slice index: {e}")
        #                 slice_index = 77  # fallback
                
        #         # Expert Model Inference
        #         # print(f'[DEBUG] Expert Model Inference')
        #         expert_response, expert_image_path, instruction = expert_model.run(
        #             img_file=expert_img_file,
        #             image_url='',
        #             input=output,
        #             output_dir=output_path,
        #             slice_index=slice_index,
        #             prompt=prompt,
        #         )

        #         answer.append({"Expert Response" : expert_response})
        #         # print(f'[DEBUG] Expert Output : {expert_response}')
        #         if expert_image_path:
        #             answer.append({"Expert Image Path" : expert_image_path})
                
        #         if instruction:
        #             answer.append({"Expert Instruction": instruction})
        #             if "video" in media_input:
        #                 conv = conv_templates[self.conv_mode].copy()
        #                 image_tokens = "\n".join(["<video>"] * (len(image_path) if isinstance(image_path, list) else 1))
        #                 updated_prompt = f"{expert_response}\n{instruction}\n{image_tokens}"
        #                 conv.append_message(conv.roles[0], updated_prompt)
        #                 conv.append_message(conv.roles[1], "")
        #                 updated_prompt_text = conv.get_prompt()
        #                 input_ids = inject_media_tokens(updated_prompt_text, self.tokenizer, media_input).to(self.model.device)
        #             else:
        #                 conv = conv_templates[self.conv_mode].copy()
        #                 image_tokens = "\n".join(["<image>"] * (len(image_path) if isinstance(image_path, list) else 1))
        #                 updated_prompt = f"{expert_response}\n{instruction}\n{image_tokens}"
        #                 conv.append_message(conv.roles[0], updated_prompt)
        #                 conv.append_message(conv.roles[1], "")
        #                 updated_prompt_text = conv.get_prompt()
        #                 input_ids = tokenizer_image_token(updated_prompt_text, self.tokenizer, return_tensors="pt").unsqueeze(0).to(self.model.device)

        #             print(f'[DEBUG] Expert Model Instruction : {instruction}')
        #             # VILA-M3 Model Inference With Expert Model Output Image
        #             with torch.inference_mode():
        #                 updated_output_ids = self.model.generate(
        #                     input_ids,
        #                     media=media_input,
        #                     media_config=media_config,
        #                     do_sample=True if temperature > 0 else False,
        #                     temperature=temperature,
        #                     top_p=top_p,
        #                     max_new_tokens=max_tokens,
        #                     use_cache=True,
        #                     stopping_criteria=[stopping_criteria],
        #                     pad_token_id=self.tokenizer.eos_token_id,
        #                 )
        #             output = self.tokenizer.batch_decode(updated_output_ids, skip_special_tokens=True)[0].strip()
        #             if output.endswith(stop_str):
        #                 output = output[:-len(stop_str)].strip()
                    
        #             print(f'[DEBUG] VILA-M3 Final Response : {output}')
        #             answer.append({"VILA-M3 Final": output})

        #     except Exception as e:
        #         print('[DEBUG] Failed')
        #         expert_response = f"Expert model encountered an error: {e}"
        #         answer.append({"Expert" : expert_response})
        #         print(f'[DEBUG] {expert_response}\n\n')
        
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
            dimension = pair["dimension"]
            key = pair["key"]
            # 단일 path도 리스트로 감쌈
            if isinstance(image_path, str):
                image_path = [image_path]
            inference_result[case_id] = inference_model.inference(
                image_path=image_path, 
                prompt=question, 
                output_path=output_path,
                temperature=temperature, 
                top_p=top_p,
                answer=answer,
                dimension=dimension,
                key=key
                )

    else:
        def is_image_file(filename):
            return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.nii', '.nii.gz'))

        if isinstance(image_dir, list):
            image_paths = [p for p in image_dir if is_image_file(p)]
            if image_paths:
                inference_result["multi_image_input"] = inference_model.inference(
                    image_path=image_paths, prompt=prompt, output_path=output_path
                )

        elif isinstance(image_dir, str) and os.path.isfile(image_dir) and is_image_file(image_dir):
            # 단일 파일
            image_paths = [image_dir]
            inference_result["single_image"] = inference_model.inference(
                image_path=image_paths, prompt=prompt, output_path=output_path
            )

        elif isinstance(image_dir, str) and os.path.isdir(image_dir):
            # 디렉토리 처리
            for case_name in os.listdir(image_dir):
                case_path = os.path.join(image_dir, case_name)
                image_paths = []

                if os.path.isfile(case_path) and is_image_file(case_name):
                    image_paths = [case_path]

                elif os.path.isdir(case_path):
                    if modality and modality.upper() == 'MRI':
                        image_paths, _ = collect_mri_paths(case_path)
                    else:
                        files = sorted(f for f in os.listdir(case_path) if is_image_file(f))
                        image_paths = [os.path.join(case_path, f) for f in files]

                if not image_paths:
                    continue

                print(f"[DEBUG] Running inference on: {case_name}")
                inference_result[case_name] = inference_model.inference(
                    image_path=image_paths, prompt=prompt, output_path=output_path, temperature=temperature, top_p=top_p
                )

        else:
            print(f"[ERROR] Invalid image_dir: {image_dir}")

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
            dimension = sample.get("Dimension", "2D")
            key = bool(sample.get('Using Label', 0))

            if isinstance(image_path, list):
                image_path = [os.path.join(DATA_ROOT_DIR, p) if not os.path.isabs(p) else p for p in image_path]
            elif isinstance(image_path, str):
                if not os.path.isabs(image_path):
                    image_path = os.path.join(DATA_ROOT_DIR, image_path)
            else:
                raise TypeError(f"Unexpected image_path type: {type(image_path)} (id: {sample_id})")

            pairs.append({"id": sample_id, "image_path": image_path, "question": question, "answer": answer, "dimension": dimension, "key": key})
    else:
        raise ValueError("JSON_FILE path must be specified.")

    results = run_batch_inference(pairs=pairs, output_path=OUTPUT_PATH)
    save_result_to_json(results, OUTPUT_PATH)