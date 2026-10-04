"""
train_smolvla.py — fine-tune SmolVLA with ONLY the vision encoder unfrozen (~86M
params), keeping the language backbone frozen and the action expert trainable.

Why: LeRobot's --policy.freeze_vision_encoder / --policy.train_expert_only flags
can't express this combination, and the full unfrozen VLM (~403M) won't fit 12 GB
(forces batch_size=2, trains too noisily). Vision-only adapts the visual encoder
to our MuJoCo render style while fitting a real batch size on an RTX A2000.
(Adapted from my earlier VLA_UR10e project.)

Usage: same flags as `python -m lerobot.scripts.lerobot_train`; the freezing logic
below overrides --policy.freeze_vision_encoder / --policy.train_expert_only.
"""
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel


def set_requires_grad_vision_only(self):
    # Freeze the whole VLM, then unfreeze only vision_model params.
    self.vlm.eval()
    for name, params in self.vlm.named_parameters():
        params.requires_grad = "vision_model" in name
    self.get_vlm_model().vision_model.train()
    # Keep the action expert trainable except its lm_head (upstream behavior).
    for name, params in self.lm_expert.named_parameters():
        if "lm_head" in name:
            params.requires_grad = False


SmolVLMWithExpertModel.set_requires_grad = set_requires_grad_vision_only

from lerobot.scripts.lerobot_train import train  # noqa: E402

if __name__ == "__main__":
    train()
