import torch
import os
import json
import nibabel as nib
import numpy as np
from tqdm import tqdm
from PIL import Image
from experts.expert_monai_brats import ExpertBrats
from experts.expert_monai_vista3d import ExpertVista3D
from experts.expert_torchxrayvision import ExpertTXRV
from llava.model.builder import load_pretrained_model
from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token
from llava.conversation import conv_templates, SeparatorStyle

MODEL_PATH="MONAI/Llama3-VILA-M3-8B"
MODEL_CARDS = "Here is a list of available expert models:\n<BRATS(args)> Modality: MRI, Task: segmentation, Overview: A pre-trained model for volumetric (3D) segmentation of brain tumor subregions from multimodal MRIs based on BraTS 2018 data, Accuracy: Tumor core (TC): 0.8559 - Whole tumor (WT): 0.9026 - Enhancing tumor (ET): 0.7905 - Average: 0.8518, Valid args are: None\n<VISTA3D(args)> Modality: CT, Task: segmentation, Overview: domain-specialized interactive foundation model developed for segmenting and annotating human anatomies with precision, Accuracy: 127 organs: 0.792 Dice on average, Valid args are: 'everything', 'hepatic tumor', 'pancreatic tumor', 'lung tumor', 'bone lesion', 'organs', 'cardiovascular', 'gastrointestinal', 'skeleton', or 'muscles'\n<VISTA2D(args)> Modality: cell imaging, Task: segmentation, Overview: model for cell segmentation, which was trained on a variety of cell imaging outputs, including brightfield, phase-contrast, fluorescence, confocal, or electron microscopy, Accuracy: Good accuracy across several cell imaging datasets, Valid args are: None\n<CXR(args)> Modality: chest x-ray (CXR), Task: classification, Overview: pre-trained model which are trained on large cohorts of data, Accuracy: Good accuracy across several diverse chest x-rays datasets, Valid args are: None\nGive the model <NAME(args)> when selecting a suitable expert model.\n"
OUTPUT_PATH = "/home/hufsaim/VLM/VLM/m3/demo/result"
INPUT_DIR = "/home/hufsaim/VLM/VLM/m3/BraTS/tmp"
MODALITY="MRI"
JSON_FILE = None #"/home/hufsaim/VLM/VLM/m3/BrainTumorMRI/dataset.json"
json_dir='/home/hufsaim/VLM/VLM/m3/BrainTumorMRI'
slice_index = 77

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

def save_result_to_json(result, output_dir, base_filename="inference_result"):
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{base_filename}.json"
    filepath = os.path.join(output_dir, filename)

    counter = 1
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
    def __init__(self, model_path="MONAI/Llama3-VILA-M3-8B", conv_mode="llama_3"):
        model_name = model_path.split("/")[-1]
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(model_path, model_name, device="cuda:0")
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
        answer.append({"image path": image_path})

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
        
        # Check Tirrger
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
                
                # Expert Model Inference
                expert_response, expert_image_path, instruction = expert_model.run(
                    image_url=expert_img_file,
                    input=output,
                    output_dir="/home/hufsaim/VLM/VLM/m3/demo/expert_result",
                    img_file=expert_img_file,
                    slice_index=slice_index,
                    prompt=prompt,
                )

                answer.append({"Expert" : expert_response})
                if expert_image_path:
                    answer.append({"Expert Image Path" : expert_image_path})
                
                if instruction:
                    conv = conv_templates[self.conv_mode].copy()
                    image_tokens = "\n".join(["<image>"] * (len(image_path) if isinstance(image_path, list) else 1))
                    updated_prompt = f"{expert_response}\n{instruction}\n{image_tokens}"
                    conv.append_message(conv.roles[0], updated_prompt)
                    conv.append_message(conv.roles[1], "")
                    updated_prompt_text = conv.get_prompt()

                    answer.append({"Expert": instruction})

                    input_ids = tokenizer_image_token(updated_prompt_text, self.tokenizer, return_tensors="pt").unsqueeze(0).to(self.model.device)

                    # VILA-M3 Model Inference With Expert Model Output Image
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
    
def run_batch_inference(
        image_dir=None, 
        modality=None, 
        prompt="Segment the entire image.",
        pairs=None
    ):
    inference_result = {}
    inference_model = M3Inference(model_path=MODEL_PATH)

    if pairs:
        for pair in tqdm(pairs, desc="Processing Cases"):
            id = pair["id"]
            image_path = pair["image_path"]
            question = pair["question"]

            inference_result[id] = inference_model.inference(image_path=image_path, prompt=question)
    else:
        if os.path.isdir(image_dir):
            for case_name in os.listdir(image_dir):
                case_path = os.path.join(image_dir, case_name)
                image_paths = []

                if os.path.isfile(case_path) and case_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.nii', '.nii.gz')):
                    image_paths = case_path
                elif os.path.isdir(case_path):
                    if modality and modality.upper() == 'MRI':
                        image_paths, _ = collect_mri_paths(case_path)
                    else:
                        files = [f for f in os.listdir(case_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.nii', '.nii.gz'))]
                        if files:
                            image_paths = os.path.join(case_path, files[0])
                        else:
                            continue
                if not image_paths:
                    continue
                print(f"[DEFUG] Running inference on: {case_name}")
                inference_result[case_name] = inference_model.inference(image_path=image_paths, prompt=prompt)
        else:
            inference_result["id"] = inference_model.inference(image_dir, prompt)
    
    return inference_result


if __name__ == "__main__":
    results = []
    if JSON_FILE:
        data = load_json_data(JSON_FILE)
        pairs = []
        for sample in data:
            id = sample["id"]
            image_path = os.path.join(json_dir, sample["image_path"])
            question = sample["question"]
            pairs.append({"id": id, "image_path": image_path, "question": question})
        
        results = run_batch_inference(pairs=pairs)
    else:
        results = run_batch_inference(INPUT_DIR, modality=MODALITY)

    save_result_to_json(results, OUTPUT_PATH)
