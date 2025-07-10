def build_input_prompt(prompt: str, modality: str, img_file, use_model_cards: bool, model_card_text: str) -> str:
    """
    Construct the input prompt for VILA-M3 based on modality and image configuration.
    
    Args:
        prompt (str): User's text prompt
        modality (str): Modality type, e.g., 'x-ray', 'ct', 'mri'
        img_file (str or list): Path(s) to the image(s)
        use_model_cards (bool): Whether to prepend the model card
        model_card_text (str): Model card content
    
    Returns:
        str: The constructed prompt
    """
    # Start with model card if enabled
    model_cards = model_card_text if use_model_cards else ""

    # Determine modality-specific prefix
    if isinstance(img_file, list) and modality.lower() == "mri" and len(img_file) == 4:
        special_token = "T1(contrast enhanced): <image>, T1: <image>, T2: <image>, FLAIR: <image> "
        mod_msg = "These are different MRI modalities.\n"
        prefix = special_token + mod_msg
        # No need to add <image> in prompt for this case (already handled)
        if "<image>" in prompt:
            prompt = prompt.replace("<image>", "")
    elif modality.lower() in ["x-ray", "ct"]:
        prefix = f"This is a {modality.upper()} image.\n"
        if "<image>" not in prompt:
            prompt = "<image>" + prompt
    else:
        # For unknown or single-modality MRI, etc.
        prefix = ""
        if "<image>" not in prompt:
            prompt = "<image>" + prompt

    # Final prompt
    full_prompt = model_cards + prefix + prompt
    return full_prompt
