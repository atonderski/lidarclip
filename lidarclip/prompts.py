import itertools


# Use same prompts as prior art for objects
# Copied from https://github.com/chongzhou96/MaskCLIP/blob/master/tools/maskclip_utils/prompt_engineering.py
# The codebase for https://arxiv.org/abs/2112.01071
OBJECT_PROMPT_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
    "there is a {} in the scene.",
    "there is the {} in the scene.",
    "this is a {} in the scene.",
    "this is the {} in the scene.",
    "this is one {} in the scene.",
]

# Generate weather and period prompts by permuting these attributes
QUALITY_MODIFIERS = ["bad", "good", "clean", "dirty", "cropped", "close-up"]
FORMAT_MODIFIERS = [
    "photo",
]
SCENE_MODIFIERS = ["environment", "scene", "road", "street", "intersection"]
CAPTURE_MODIFIERS = [
    "taken",
    "captured",
]

WEATHER_PROMPT_TEMPLATES = set()
for quality, format, scene, capture in itertools.product(
    QUALITY_MODIFIERS, FORMAT_MODIFIERS, SCENE_MODIFIERS, CAPTURE_MODIFIERS
):
    # Example: a good photo of a rainy environment
    WEATHER_PROMPT_TEMPLATES.add(f"a {quality} {format} of a {{}} {scene}.")
    WEATHER_PROMPT_TEMPLATES.add(f"a {quality} {format} {capture} on a {{}} day.")
    WEATHER_PROMPT_TEMPLATES.add(f"a {quality} {format} {capture} in a {{}} {scene}.")
    WEATHER_PROMPT_TEMPLATES.add(f"a {quality} {format} of many things in a {{}} {scene}.")

PERIOD_PROMPT_TEMPLATES = set()
for quality, format, scene, capture in itertools.product(
    QUALITY_MODIFIERS, FORMAT_MODIFIERS, SCENE_MODIFIERS, CAPTURE_MODIFIERS
):
    if quality == "bright" or quality == "dark":
        continue  # Do not bias day/night by brightness in prompt
    # Example: a good photo taken at night
    PERIOD_PROMPT_TEMPLATES.add(f"a {quality} {format} {capture} at {{}}.")
    # Example: a good photo of a scene taken at night
    PERIOD_PROMPT_TEMPLATES.add(f"a {quality} {format} of a {scene} {capture} at {{}}.")
    PERIOD_PROMPT_TEMPLATES.add(
        f"a {quality} {format} of many things in a {scene} {capture} at {{}}."
    )
    PERIOD_PROMPT_TEMPLATES.add(f"a {quality} {format} of the {scene} {capture} at {{}}.")
PERIOD_PROMPT_TEMPLATES = list(set(PERIOD_PROMPT_TEMPLATES))  # Remove duplicates

EMPTY_PROMPTS = set()
BUSY_PROMPTS = set()
for quality, format, scene in itertools.product(
    QUALITY_MODIFIERS, FORMAT_MODIFIERS, SCENE_MODIFIERS
):
    # Example: a good photo of a busy environment
    for busy_modifier in ("busy", "crowded", "full"):
        BUSY_PROMPTS.add(
            f"a {quality} {format} of extremely {busy_modifier} traffic during rush hour with a large number of nearby vehicles."
        )

    for empty_modifier in ("empty", "deserted", "abandoned"):
        EMPTY_PROMPTS.add(
            f"a {quality} {format} of a completely {empty_modifier} {scene} with no vehicles in sight."
        )
