"""
instruction_templates.py — the ONE source of task-instruction strings, shared by
demo collection (now) and closed-loop inference (later). Training and inference
MUST use the same templates, so both import from here.

Each product maps to a human-readable name; templates are paraphrases of the same
goal (pick the named item, place it in the basket). One template is drawn per
episode so the VLA keys off the task, not one exact sentence.
"""
import random

# body name -> natural language item name
ITEM_NAMES = {
    "milk_carton":  "milk carton",
    "cola_can":     "can of cola",
    "bread_loaf":   "loaf of bread",
    "cereal_box":   "box of cereal",
    "water_bottle": "water bottle",
}

TEMPLATES = [
    "pick up the {item} and place it in the basket",
    "grab the {item} and put it in the basket",
    "take the {item} and drop it in the basket",
    "put the {item} into the basket",
    "collect the {item} and place it in the basket",
    "pick the {item} off the shelf and put it in the basket",
]


def instruction_for(product, rng=None):
    """Return one instruction string for `product` (a body name), randomized."""
    item = ITEM_NAMES.get(product, product.replace("_", " "))
    template = (rng or random).choice(TEMPLATES)
    return template.format(item=item)
