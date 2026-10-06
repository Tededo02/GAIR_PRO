MODEL_FAMILIES = ("rigid", "superflex")


def validate_model_family(model_family: str) -> str:
    if model_family not in MODEL_FAMILIES:
        raise ValueError(f"model_family must be one of {MODEL_FAMILIES}, got {model_family!r}")
    return model_family


def parameter_count(model_family: str) -> int:
    return 11 if validate_model_family(model_family) == "rigid" else 19
