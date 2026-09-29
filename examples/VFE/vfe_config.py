from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


so100_config = {
    # Video: current frame only; keys must match "video" entries in meta/modality.json
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_front", "cam_outside"],  # front third-person view + wrist egocentric
    ),
    # State: current proprioceptive reading; keys must match "state" entries in meta/modality.json
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm",  # joint positions
            "left_hand",  # gripper state
        ],
    ),
    # Action: 16-step prediction horizon; one ActionConfig per modality key
    "action": ModalityConfig(
        delta_indices=list(range(0, 32, 2)),  # predict 16 future steps
        modality_keys=[
            "left_arm",
            "left_hand",
        ],
        action_configs=[
            # single_arm: RELATIVE = delta from current state (better generalization)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,  # joint-space, not end-effector
                format=ActionFormat.DEFAULT,
            ),
            # gripper: ABSOLUTE = target position (binary open/close works better absolute)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    # Language: task instruction from annotation field in the dataset
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(so100_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)


# Bimanual VFE robot (dual arm + dual hand), configured for RECAP advantage-conditioned
# training. Matches the dataset's meta/modality.json layout, e.g.:
#   "state":  {"left_arm": {...}, "right_arm": {...}, "left_hand": {...}, "right_hand": {...}}
#   "action": {"left_arm": {...}, "right_arm": {...}, "left_hand": {...}, "right_hand": {...}}
#   "video":  {"cam_front": {...}, "cam_outside": {...}}
#   "reward": {"current": {"original_key": "next.reward"}}
# The "reward" key opts this config into RECAP: LeRobotEpisodeLoader reads its
# original_key from modality.json as the per-frame success/failure column, instead
# of the default "next.done".
vfe_recap_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_front", "cam_outside"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm",
            "left_hand",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 32, 2)),  # predict 16 future steps
        modality_keys=[
            "left_arm",
            "left_hand",
        ],
        action_configs=[
            # arms: RELATIVE = delta from current state (better generalization)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # hands: ABSOLUTE = target position (binary open/close works better absolute)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
    # RECAP: per-frame outcome signal. modality_keys=["current"] pairs with
    # modality.json's "reward.current.original_key" to locate the raw column.
    "reward": ModalityConfig(
        delta_indices=[0],
        modality_keys=["current"],
    ),
}

register_modality_config(vfe_recap_config, embodiment_tag=EmbodimentTag.VFE_RECAP)
