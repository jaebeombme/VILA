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

MODEL_PATH="/home/hufsaim/VLM/VLM/Llama3-VILA-M3-8B/snapshots/df60e0276e2ae10624c86dabe909847a03b2a5cb"
DATA_ROOT_DIR = "/home/hufsaim/VLM/VLM/data/0527"
OUTPUT_PATH = "/home/hufsaim/VLM/VLM/m3/demo/0527"
JSON_FILE = "/home/hufsaim/VLM/VLM/data/0527/question1.json"
SLICE_SAVE_PATH = "/home/hufsaim/VLM/VLM/m3/demo/sliced_images"

MODEL_CARDS = """Here is a list of available expert models:\n
<MPS(args)>
Modality: MRI
Task: classification
Overview: A deep learning-based classification model that predicts the MRI sequence type (e.g., T1, T2, FLAIR, T1CE, STIR, PD) from input MR images. It is designed to automatically annotate sequence labels for unlabeled MR scans, improving downstream processing and dataset curation. The model was trained on a diverse multi-institutional dataset covering common brain imaging protocols with varying acquisition parameters.
Accuracy:
T1: 94.2%
T2: 95.8%
FLAIR: 93.7%
T1CE: 92.1%
Average: 92.7%
Valid args are: None\n
Give the model <NAME(args)> when selecting a suitable expert model.\n"""

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

def save_result_to_json(result, output_dir, base_filename="inference_result_wo_expert"):
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
    def __init__(self, model_path=MODEL_PATH, conv_mode="llama_3"):
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(model_path, model_name, device="cuda")
        self.conv_mode = conv_mode

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

            if isinstance(image_path, list):
                image_path = [os.path.join(DATA_ROOT_DIR, p) if not os.path.isabs(p) else p for p in image_path]
            elif isinstance(image_path, str):
                if not os.path.isabs(image_path):
                    image_path = os.path.join(DATA_ROOT_DIR, image_path)
            else:
                raise TypeError(f"Unexpected image_path type: {type(image_path)} (id: {sample_id})")

            pairs.append({"id": sample_id, "image_path": image_path, "question": question, "answer": answer})
    else:
        raise ValueError("JSON_FILE path must be specified.")

    results = run_batch_inference(pairs=pairs)
    save_result_to_json(results, OUTPUT_PATH)