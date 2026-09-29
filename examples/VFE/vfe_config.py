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


# VFE robot, configured for RECAP advantage-conditioned training. Matches the
# dataset's meta/modality.json layout, e.g.:
#   "state":  {"left_arm": {...}, "left_hand": {...}}
#   "action": {"left_arm": {...}, "left_hand": {...}}
#   "video":  {"cam_front": {...}, "cam_outside": {...}}
#
# RECAP itself needs no "reward" entry here: the per-frame outcome column is
# read directly from the dataset's meta/modality.json by LeRobotEpisodeLoader
# (see _reward_column(), which honors a "reward": {"current": {"original_key":
# "next.reward"}} section there, falling back to "next.done"). Declaring a
# "reward" ModalityConfig key in this dict would instead make the generic
# per-step extraction in extract_step_data() look for a literal "reward.current"
# dataframe column, which is never populated, and crash with a KeyError.
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
}

register_modality_config(vfe_recap_config, embodiment_tag=EmbodimentTag.VFE_RECAP)
