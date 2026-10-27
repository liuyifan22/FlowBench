"""Dataclasses serialized into each eval ``sample.json``.

Hierarchy
---------
EvalSample
  └── ViewSample  (one agent-eye camera + one translation edit)
        ├── CameraPose
        ├── MoveAction
        └── Question[]   (Modification 4: typically one category-matched question)

Coordinates are AI2-THOR world meters unless noted. Camera-frame axes are
documented in ``camera_search.camera_basis``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, Union


Direction = Literal["left", "right", "forward", "back", "up", "down"]
QaCategory = Literal[
    "occlusion",
    "closer_to",
    "contact_wall",
    "affordance",
    "left_right_landmark",
    "feasibility",
    "support",
    "closest_three",
    "count_near",
    "near_object",
]
AnswerType = Literal["yes_no", "choice", "multi_choice", "set", "open"]
AnswerValue = Optional[Union[bool, str, list[str]]]


@dataclass
class Vec3:
    x: float
    y: float
    z: float

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "Vec3":
        return cls(x=float(d["x"]), y=float(d["y"]), z=float(d["z"]))

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass
class CameraPose:
    """Agent body pose for ``TeleportFull`` (agent-eye camera follows)."""

    position: Vec3
    rotation_y: float
    horizon: float
    standing: bool = True
    score: float = 0.0
    camera_position: Optional[Vec3] = None

    def teleport_kwargs(self) -> dict[str, Any]:
        return {
            "position": self.position.to_dict(),
            "rotation": {"x": 0.0, "y": self.rotation_y, "z": 0.0},
            "horizon": self.horizon,
            "standing": self.standing,
        }


@dataclass
class ObjectRef:
    object_id: str
    object_type: str
    position: Vec3
    rotation: dict[str, float]
    distance: Optional[float] = None
    visible: Optional[bool] = None

    @classmethod
    def from_metadata(cls, obj: dict[str, Any]) -> "ObjectRef":
        return cls(
            object_id=obj["objectId"],
            object_type=obj["objectType"],
            position=Vec3.from_dict(obj["position"]),
            rotation=dict(obj["rotation"]),
            distance=obj.get("distance"),
            visible=obj.get("visible"),
        )


@dataclass
class MoveAction:
    object_id: str
    object_type: str
    direction: Direction
    meters: float
    delta_world: Vec3
    position_before: Vec3
    position_after: Vec3
    rotation_before: dict[str, float]
    distance_before: float
    angular_size_proxy: float
    still_visible_in_rgb1: bool
    text: str
    # Modification 5: category-specific GT hints (wall collision, planned occludee, …).
    edit_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Question:
    category: QaCategory
    question_id: str
    text: str
    answer: AnswerValue
    answer_type: AnswerType = "yes_no"
    gt_meta: dict[str, Any] = field(default_factory=dict)
    gt_complete: bool = False
    # Modification 12: SoM-aware wording (mark tokens); None when SoM disabled.
    text_som: Optional[str] = None


@dataclass
class ViewSample:
    view_id: str
    camera: CameraPose
    move: MoveAction
    rgb0_path: str
    rgb1_path: str
    questions: list[Question]
    vlm_rank_notes: Optional[str] = None
    occlusion_method: Optional[str] = None
    target_category: Optional[str] = None
    # Modification 12: SoM-annotated frames (same camera / edit).
    rgb0_som_path: Optional[str] = None
    rgb1_som_path: Optional[str] = None
    # Modification 20+: per-RGB aux maps (instance seg + depth).
    seg0_path: Optional[str] = None
    seg1_path: Optional[str] = None
    seg0_ids_path: Optional[str] = None
    seg1_ids_path: Optional[str] = None
    depth0_path: Optional[str] = None
    depth1_path: Optional[str] = None
    # Camera-level copy: {"rgb0": {seg_id: {category, instance_id, ...}}, "rgb1": {...}}.
    seg_id_to_label: Optional[dict[str, Any]] = None


@dataclass
class EvalSample:
    sample_id: str
    dataset: str
    scene: str
    views: list[ViewSample]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
