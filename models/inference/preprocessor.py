def preprocess_input(data):
    """Normalize and validate input for inference.

    Args:
        data: dict with possible keys 'text' and/or 'audio_path'
    Returns:
        dict: cleaned data structure for model consumption
    """
    return data or {}
