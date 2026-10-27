"""Pipeline constants and ``PipelineConfig``.

See README development log (Modification 1–5).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
TMP_DIR = Path(os.environ.get("CIT_TMP_DIR", Path.home() / ".cache" / "cit" / "tmp"))
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "ithor_eval"

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
WIDTH = 800
HEIGHT = 600
FOV = 90
VISIBILITY_DISTANCE = 9.0

# ---------------------------------------------------------------------------
# Mover on-screen size (Modification 9)
# ---------------------------------------------------------------------------
# At native 800×600, reject movers whose instance mask has fewer pixels than this.
MOVER_MASK_MIN_PIXELS = 1000
# Occlusion occludee (B) must also be large enough on screen to ask about.
OCCLUDEE_MASK_MIN_PIXELS = 150

# ---------------------------------------------------------------------------
# Obviousness / move magnitude
# ---------------------------------------------------------------------------
DIST_MIN = 1.5
DIST_MAX = 4.5
ANGULAR_SIZE_MIN = 0.08  # move_meters / distance_to_camera

# ---------------------------------------------------------------------------
# Referring (Modification 4, 8)
# ---------------------------------------------------------------------------
ORDINAL_MAX_SAME_TYPE = 4
# Modification 8: when count exceeds the ordinal cap, leftmost/rightmost
# instances remain referable (see referring.instance_referring_allowed).

# ---------------------------------------------------------------------------
# Affordance (Modification 8)
# ---------------------------------------------------------------------------
# Chairs with a meaningful yaw only; stools/ottomans/sofas have ambiguous facing.
AFFORDANCE_CHAIR_TYPES = frozenset({"Chair", "ArmChair"})

# ---------------------------------------------------------------------------
# Support QA (Modification 3 / 10)
# ---------------------------------------------------------------------------
# AABB bottom must be at least this high before the move (not floor furniture).
SUPPORT_QA_MIN_BOTTOM_Y_M = 0.30
# Treat AABB bottoms below this as "on / near the floor".
SUPPORT_QA_FLOOR_BOTTOM_Y_M = 0.15
# After gravity settle: this much drop in bottom/center y ⇒ dropped off support.
SUPPORT_DROP_HEIGHT_M = 0.20

# ---------------------------------------------------------------------------
# Contact-wall QA (Modification 3–5, 8)
# ---------------------------------------------------------------------------
# Modification 8: enlarge eligibility to objects within 1.5 m of a wall.
WALL_SELECT_NEAR_M = 1.5
# Fast find_edit (ProcTHOR): tighter near-wall band for one-shot slides.
WALL_SELECT_NEAR_M_FAST = 0.8
# Stop this far short of the scene bound so teleports don't penetrate.
WALL_STOP_MARGIN_M = 0.1
# Modification 8+/24: fraction of wall edits that slide *away* from the nearest
# wall (intended answer "no"). Kept very low — completed "away" samples are easy
# and otherwise dominate the observed Yes/No ratio (selection bias).
# Overridable via ``PipelineConfig.wall_away_prob`` (iTHOR balance profile).
WALL_AWAY_PROB = 0.05
# Target fraction of contact_wall answers that should be "no" (yes = 1 - this).
# v6 observed ~36% yes with 0.25; push harder toward yes attempts.
# Overridable via ``PipelineConfig.contact_target_no_frac``.
CONTACT_TARGET_NO_FRAC = 0.08
# Legacy alias used by older eligibility helpers.
WALL_QA_NEAR_M = WALL_SELECT_NEAR_M
# Total slide toward wall (split into WALL_MOVE_STEPS sub-teleports).
# Include short slides so objects that leave the view after a long push still
# yield a valid toward-wall example (Modification 8).
WALL_MOVE_METERS = (0.15, 0.25, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5)
WALL_MOVE_STEPS = 4
# Answer "yes" if edge clearance < this. Looser than 0.15 so toward-wall slides
# that stop just short of the bound still count (v6 median clearance was ~1.6m
# on Nos; Yes needs clearance under threshold).
WALL_CONTACT_M = 0.22
# Fast closer_to: validate at most this many metadata plans in sim.
# Floor of 3 motion enumerations is enforced in find_closer_to_flip_edit.
CLOSER_TO_FAST_MAX_TRIES = 6
# Place occluder this far in front of B along the camera→B ray (meters).
OCCLUDE_RAY_FRONT_MARGIN_M = 0.12
# Prefer-no: lateral offset off the ray so cover stays below threshold.
OCCLUDE_RAY_NO_OFFSET_M = 0.22

# ---------------------------------------------------------------------------
# Occlusion QA (Modification 5: cover small B with large A)
# ---------------------------------------------------------------------------
OCCLUDE_THRESHOLD = 0.30  # Modification 5: cover of B's rgb0 region by A in rgb1
# Target fraction of occlusion answers that should be "no" / not-occluded.
# Mod 24: prefer positive (occluded) pairs — cover search already exists.
OCCLUDE_TARGET_NO_FRAC = 0.2
DEFAULT_CHAIR_FOOTPRINT_M = 0.55
# Scale factors on the pixel-planned lateral meters (search for best cover).
OCCLUDE_COVER_SCALE = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25)
# Extra pull of A toward the camera so A ends up in front of B (meters).
OCCLUDE_COVER_BACK_M = (0.0, 0.08, 0.15, 0.25, 0.40, 0.55)
LARGE_OCCLUDER_TYPES = (
    "Chair",
    "Stool",
    "Ottoman",
    "ArmChair",
    "Microwave",
    "Laptop",
    "Box",
    "Pillow",
    "HousePlant",
    "BasketBall",
    "TeddyBear",
    "Toaster",
    "CoffeeMachine",
    "Pot",
    "Pan",
    "GarbageCan",
)
SMALL_OCCLUDEE_TYPES = (
    "Apple",
    "Cup",
    "Mug",
    "Bowl",
    "Plate",
    "Pen",
    "Pencil",
    "CellPhone",
    "CreditCard",
    "Watch",
    "Candle",
    "KeyChain",
    "RemoteControl",
    "SoapBottle",
    "DishSponge",
    "Egg",
    "Tomato",
    "Bread",
    "Statue",
    "Vase",
    "AlarmClock",
    "Book",
    "Newspaper",
    "ButterKnife",
    "Fork",
    "Spoon",
    "PepperShaker",
    "SaltShaker",
    "Glassbottle",
    "WineBottle",
    "SprayBottle",
    "TissueBox",
    "ToiletPaper",
    "HandTowel",
    "Cloth",
)
# Max 3D distance between A and B when proposing a cover move.
OCCLUDE_PAIR_MAX_DIST_M = 2.25
OCCLUDE_COVER_METERS_MIN = 0.15
OCCLUDE_COVER_METERS_MAX = 1.50
# Prefer A/B height difference below this when ranking pairs (meters).
OCCLUDE_PAIR_MAX_HEIGHT_DIFF_M = 0.85

# ---------------------------------------------------------------------------
# Closest-three QA (Modification 4)
# ---------------------------------------------------------------------------
CLOSEST_THREE_POOL = 6
CLOSEST_THREE_ANSWER = 3
# In closer_to ABCD anchors and closest_three / count_near pools (size 6),
# never show more than one instance of each of these look-alike types.
ANCHOR_TYPE_CAP_ONCE = frozenset({"ShelvingUnit", "Drawer"})

# ---------------------------------------------------------------------------
# Preferred move targets
# ---------------------------------------------------------------------------
PREFERRED_OBJECT_TYPES = (
    "Chair",
    "Stool",
    "Ottoman",
    "Laptop",
    "Box",
    "Pillow",
    "AlarmClock",
    "Book",
    "Bowl",
    "Cup",
    "Mug",
    "Plate",
    "Statue",
    "Vase",
    "RemoteControl",
    "Egg",
    "HousePlant",
    "CellPhone",
    "Candle",
    "BasketBall",
    "BaseballBat",
    "TeddyBear",
    "Apple",
    "Tomato",
    "SoapBottle",
)

# Legacy sittable set (Mod 2). Affordance (Mod 8) uses AFFORDANCE_CHAIR_TYPES only.
SITTABLE_TYPES = frozenset({"Chair", "ArmChair", "Stool", "Ottoman", "Sofa"})

# Modification 8 debug / subset builds: four iTHOR scenes (one per room type).
V8_SCENES = (
    "FloorPlan1",
    "FloorPlan201",
    "FloorPlan301",
    "FloorPlan401",
)
V8_CATEGORIES = ("affordance", "contact_wall")

# Distractors for closer_to MCQ that are often absent from a given room.
ABSENT_DISTRACTOR_TYPES = (
    "Tablet",
    "Microwave",
    "Toaster",
    "Television",
    "Toilet",
    "Laptop",
    "Apple",
    "BasketBall",
    "AlarmClock",
    "TeddyBear",
)

MOVE_DIRECTIONS = ("left", "right", "forward", "back")
MOVE_METERS_CANDIDATES = (0.6, 0.5, 0.4, 0.25)
# Vertical *down* probes for feasibility QA (Modification 10: no upward probes).
FEASIBILITY_VERTICAL_METERS = (0.6, 0.5, 0.4)
FEASIBILITY_ORTHOGONAL_PROB = 0.85
# Include vertical drop in move command text when |Δy| exceeds this.
# Modification 10: never describe / apply *upward* moves (against gravity).
MOVE_TEXT_VERTICAL_MIN_M = 0.08
ALLOW_MOVE_UP = False
# Feasibility may still probe *down*; upward probes are banned (Mod 10).
FEASIBILITY_PROBE_DIRECTIONS = MOVE_DIRECTIONS + ("down",)
# Wall slides: small collision-checked steps (forceAction=False).
WALL_SLIDE_STEP_M = 0.05
# Reject wall poses whose AABB penetrates past the scene bound by more than this.
WALL_PENETRATION_TOL_M = 0.02
# Support QA: release kinematics and settle under gravity after the slide.
SUPPORT_GRAVITY_MAX_STEPS = 80
SUPPORT_GRAVITY_TIMESTEP = 0.05
SUPPORT_GRAVITY_MIN_STEPS = 12  # always advance a bit even if "at rest"
SUPPORT_GRAVITY_PROB = 1.0  # always settle for support category (Mod 10)
# Modification 11: binary-refine a blocked slide along the ray (halve step, 4 iters).
# Example: 1.0 fail → 0.5 fail → 0.25 ok → +0.125 ok → total 0.375 m applied.
BINARY_SLIDE_ITERS = 4
# Categories that settle pickupable movers under gravity after the slide (Mod 11).
# Occlusion / wall contact stay kinematic so cover / wall GT stay controllable.
GRAVITY_SETTLE_CATEGORIES = frozenset(
    {
        "support",
        "feasibility",
        "closer_to",
        "closest_three",
        "count_near",
        "near_object",
        "left_right_landmark",
    }
)

# ---------------------------------------------------------------------------
# Modification 12 — Set-of-Marks + answer-flip quotas + wall visibility
# ---------------------------------------------------------------------------
SOM_MARK_RADIUS = 10
SOM_MARK_FILL = (220, 20, 20)
SOM_MARK_OUTLINE = (255, 255, 255)
SOM_TEXT_FILL = (255, 255, 255)
SOM_MARK_MIN_SEP_PX = 24
# Badge center offset from object mask centroid, in mark-radii units.
# Was 0.8 for procthor_v7; default 1.2 for v8+ (less occlusion of small objects).
SOM_MARK_OFFSET_RADII = 1.2
SOM_MOVER_MASK_MIN_PIXELS = 30
SOM_OCCLUDEE_MASK_MIN_PIXELS = 20
WALL_AFTER_MASK_MIN_PIXELS = 30
ANSWER_FLIP_TRAIN_KEEP_IDENTICAL_FRAC = 0.50
ANSWER_FLIP_BENCH_IDENTICAL_FRAC = 0.10
ANSWER_FLIP_PRIORITY_CATEGORIES = frozenset(
    {"left_right_landmark", "closer_to", "occlusion", "contact_wall"}
)
ANSWER_FLIP_GEN_IDENTICAL_FRAC = 0.10

# ---------------------------------------------------------------------------
# Modification 15/20 — flip-planned closer_to / left_right + Round-2 fixes
# ---------------------------------------------------------------------------
# v7.2: larger gap so B↔C flips are visually obvious (was 0.08).
CLOSER_TO_GAP_M = 0.18
CLOSER_TO_MIN_MOVE_M = 0.25
CLOSER_TO_MAX_MOVE_M = 1.05
CLOSER_TO_MIN_BC_SEP_M = 0.30
# Fractions of |BC| past the B–C midpoint toward C (v7.2: further than mid).
CLOSER_TO_FLIP_FRAC_PAST_MID = (0.45, 0.55, 0.60, 0.50, 0.65, 0.40)
LEFT_RIGHT_MIN_CROSS_DEG = 4.0
LEFT_RIGHT_MIN_SPAN_DEG = 6.0
# v7.2: left/right GT uses mask-centroid pixel *u*; require a clear pixel gap.
LEFT_RIGHT_MIN_SPAN_PX = 36.0
LEFT_RIGHT_MIN_CROSS_PX = 12.0
# v7.2: push slides ~0.1 m further than v7.1 meter options.
LEFT_RIGHT_EXTRA_METERS = 0.10
# Prefer answer-flipping slides, but keep ~20% same-side (non-flip) to reduce shortcuts.
LEFT_RIGHT_REQUIRE_FLIP = False
LEFT_RIGHT_NONFLIP_FRAC = 0.20
FLIP_PLAN_FORCE_EDIT = True
# Feasibility: bias slightly toward collision ("no"); longer probes help find nos.
FEASIBILITY_TARGET_NO_FRAC = 0.55
FEASIBILITY_PROBE_METERS = (1.5, 1.25, 1.0, 0.75, 0.6, 0.5, 0.4)
# Legacy fast split (v9 and earlier): short↔Yes / long↔No — a language shortcut.
# Kept only for ablations; prefer continuous shared sampling below.
FEASIBILITY_FAST_YES_METERS = (0.4, 0.5)
FEASIBILITY_FAST_NO_METERS = (1.5, 1.25, 1.0)
# Discrete fallback grid (ablations / non-continuous mode).
FEASIBILITY_FAST_SHARED_METERS = (0.4, 0.5, 0.75, 1.0, 1.25, 1.5)
# v10.1+: sample probe length from a continuous range (meter-first), not a fixed
# grid. Same support for Yes and No intents. Mild counter-skew is optional; keep
# powers near 1.0 so both polarities remain findable (long-favoring Yes-intent
# starves feasible probes on ProcTHOR).
FEASIBILITY_PROBE_M_MIN = 0.40
FEASIBILITY_PROBE_M_MAX = 1.50
FEASIBILITY_PROBE_CONTINUOUS = True
# m = min + (max-min) * U**power. power=1 → uniform on [min,max].
# Slight Yes→shorter / No→longer helps findability while meter-first keeps
# accepted means much closer than polarity-pool picking.
FEASIBILITY_PROBE_YES_POWER = 1.15  # Yes-intent: slight short bias (easier Yes)
FEASIBILITY_PROBE_NO_POWER = 0.85  # No-intent: slight long bias (easier No)
# Prefer picking a probe whose direction also has the opposite polarity in-pool
# (blocks "all Yes are forward / all No are back" shortcuts within a scene).
FEASIBILITY_BALANCE_DIRECTION_POLARITY = True
FEASIBILITY_FAST_MAX_ATTEMPTS = 12
FEASIBILITY_COMPOUND_METERS = (0.5, 0.75, 1.0, 1.25)
# Contact_wall v10+: always slide toward the nearest wall. Intentional Nos use a
# half-length slide (see CONTACT_HALF_MOTION_NO_FRAC); full slides aim for Yes
# but may fail and become No (helps label balance without away-from-wall leaks).
CONTACT_ALWAYS_TOWARD = True
# Intentional half-No coin flip. Observed accepted half% is lower than this
# (failed half attempts skip), and full-slide misses add extra Nos — keep in
# sync with ITHOR_CONTACT_TARGET_NO_FRAC (~45%).
CONTACT_HALF_MOTION_NO_FRAC = 0.45
# Half-motion No geometry (v10): slide ``0.5 * clearance - CONTACT_HALF_MOVE_SUB_M``
# toward the wall; require expected leftover clearance ≥ CONTACT_HALF_MIN_LEFTOVER_M.
# Kept loose so half-Nos are not starved; the post-move contact check still
# decides Yes vs No (QA Yes uses WALL_CONTACT_M).
CONTACT_HALF_MOVE_SUB_M = 0.05
CONTACT_HALF_MIN_LEFTOVER_M = 0.08
# v10 contact: room-polygon wall pre-filter (metadata-only) + ranked candidates.
CONTACT_ROOM_WALL_PREFILTER = True
CONTACT_ROOM_WALL_MAX_M = 2.5
CONTACT_FAST_MAX_CANDIDATES = 8
CONTACT_FAST_SINGLE_CAMERA = False
CONTACT_FEAS_FIRST = True
# Occlusion question wording (ablation winner on ithor_v4/som).
OCCLUDE_PROMPT_VARIANT = "partial_or_totally_visible"
# Pin occludee SoM badge to rgb0 location on both frames (avoid flying (2)).
SOM_PIN_OCCLUDEE_TO_RGB0 = True
MODIFICATION_TAG = "mod30_ithor_v7_2_pixel_lr_stronger_flips"

# ---------------------------------------------------------------------------
# iTHOR balance profile (Mod 29 / v7.1): aim ~45–55% Yes on yes/no categories.
# closer_to left unchanged (MCQ4).
#
# Note: category-driven builds ask ≤1 Q/category/scene, so the running balance
# starts tied every scene — ``*_target_no_frac`` is effectively P(prefer No).
# Contact Yes is hard under fast slides (~10% success when preferred); set a
# low No-target so prefer-Yes dominates and completed Yes% can reach ~50%.
# ---------------------------------------------------------------------------
ITHOR_OCCLUDE_TARGET_NO_FRAC = 0.50
# v10 contact: ~45% intentional half-motion Nos (tune with CONTACT_HALF_MOTION_NO_FRAC).
ITHOR_CONTACT_TARGET_NO_FRAC = 0.45
# Away slides disabled under CONTACT_ALWAYS_TOWARD; kept for legacy overrides.
ITHOR_WALL_AWAY_PROB = 0.0
# Use a wide eligibility band even under fast_find_edit (fast default is 0.8 m).
ITHOR_FAST_WALL_SELECT_NEAR_M = 2.5
ITHOR_FEASIBILITY_TARGET_NO_FRAC = 0.50
ITHOR_LEFT_RIGHT_TARGET_NO_FRAC = 0.50
# Ranked contact candidates (not a tiny top-2 cap); enough tries to walk the list.
ITHOR_FAST_CONTACT_MAX_TRIES = 24
ITHOR_FAST_OCCLUSION_MAX_TRIES = 56
ITHOR_FAST_FEASIBILITY_MAX_ATTEMPTS = 40
# Default fast caps when PipelineConfig overrides are unset (ProcTHOR).
FAST_CONTACT_MAX_TRIES = 24
FAST_OCCLUSION_MAX_TRIES = 16
WALL_BANNED_MOVE_TYPES = frozenset(
    {
        "SinkBasin",
        "Sink",
        "Bathtub",
        "BathtubBasin",
        "Toilet",
        "ShowerGlass",
        "ShowerDoor",
        "LightSwitch",
        "Sofa",
        "ArmChair",
    }
)
FORBIDDEN_MOVE_TYPES = frozenset(
    {
        "SinkBasin",
        "Sink",
        "Sofa",
        "ArmChair",
        "LightSwitch",
        "CD",
        "ShowerGlass",
        "ShowerDoor",
        "ShowerHead",
        "Bathtub",
        "BathtubBasin",
        "Toilet",
        # Too small / hard to see as primary movers (always banned, incl. SoM).
        "KeyChain",
        "Watch",
        "Pen",
        "Pencil",
    }
)
CHOICE_MASK_MIN_PIXELS = 40
MOVER_MASK_MIN_PIXELS_MOD20 = 80

# Dual MCQ roots
ITHOR_V4_NAME = "ithor_v4"
ITHOR_V4_NATURAL_NAME = "ithor_v4/natural"
ITHOR_V4_SOM_NAME = "ithor_v4/som"
ITHOR_V20_NAME = "ithor_v20"
ITHOR_V20_NATURAL_NAME = "ithor_v20/natural"
ITHOR_V20_SOM_NAME = "ithor_v20/som"
ITHOR_V5_NAME = "ithor_v5"
ITHOR_V5_NATURAL_NAME = "ithor_v5/natural"
ITHOR_V5_SOM_NAME = "ithor_v5/som"
PROCTHOR_V24_NAME = "procthor_v24"
PROCTHOR_V24_NATURAL_NAME = "procthor_v24/natural"
PROCTHOR_V24_SOM_NAME = "procthor_v24/som"
PROCTHOR_V7_NAME = "procthor_v7"
PROCTHOR_V7_NATURAL_NAME = "procthor_v7/natural"
PROCTHOR_V7_SOM_NAME = "procthor_v7/som"
PROCTHOR_V8_NAME = "procthor_v8"
PROCTHOR_V8_NATURAL_NAME = "procthor_v8/natural"
PROCTHOR_V8_SOM_NAME = "procthor_v8/som"
DEFAULT_DATASET_PARENT = Path(
    os.environ.get("CIT_DATA_ROOT", "data/cit")
)

# Modification 15/20 default categories (legacy eight kept as QA_CATEGORIES_V12).
# Mod-30 / procthor_v8 / build_v30: five core + three distance categories.
QA_CATEGORIES = (
    "occlusion",
    "closer_to",
    "contact_wall",
    "left_right_landmark",
    "feasibility",
    "closest_three",
    "count_near",
    "near_object",
)
QA_CATEGORIES_V12 = (
    "occlusion",
    "closer_to",
    "contact_wall",
    "affordance",
    "left_right_landmark",
    "feasibility",
    "support",
    "closest_three",
)
QA_CATEGORIES_V5 = (
    "occlusion",
    "closer_to",
    "contact_wall",
    "left_right_landmark",
    "feasibility",
)

VLM_BASE_URL = "http://127.0.0.1:19857/v1"
VLM_MODEL = "Qwen3-VL-32B-Instruct"
VLM_API_KEY = "EMPTY"

DEFAULT_ITHOR_SCENE = "FloorPlan201"

# Modification 4 scene set (kitchen / living / bedroom / bathroom).
V4_SCENES = (
    "FloorPlan1",
    "FloorPlan2",
    "FloorPlan201",
    "FloorPlan202",
    "FloorPlan301",
    "FloorPlan302",
    "FloorPlan401",
    "FloorPlan402",
)

# Full iTHOR split: kitchens 1–30, living 201–230, bedrooms 301–330, baths 401–430.
ITHOR_FULL_SCENES: tuple[str, ...] = tuple(
    [f"FloorPlan{i}" for i in range(1, 31)]
    + [f"FloorPlan{i}" for i in range(201, 231)]
    + [f"FloorPlan{i}" for i in range(301, 331)]
    + [f"FloorPlan{i}" for i in range(401, 431)]
)


@dataclass
class PipelineConfig:
    """Runtime knobs for ``build_one_eval_sample`` / CLI."""

    scene: str = DEFAULT_ITHOR_SCENE
    out_dir: Path = field(default_factory=lambda: DEFAULT_OUT_DIR)
    width: int = WIDTH
    height: int = HEIGHT
    fov: int = FOV
    visibility_distance: float = VISIBILITY_DISTANCE
    dist_min: float = DIST_MIN
    dist_max: float = DIST_MAX
    angular_size_min: float = ANGULAR_SIZE_MIN
    occlude_threshold: float = OCCLUDE_THRESHOLD
    # Modification 4 default: one camera; one view per QA category.
    n_views: int = 1
    questions_per_view: int = 1
    # Floor positions for camera FPS subsample (was 40 stride; now 20 FPS).
    max_camera_positions: int = 20
    # Fast mode: FPS down to this many positions (ProcTHOR).
    fast_camera_max_positions: int = 15
    # Yaw steps: full quality uses 8 (45°); fast mode uses 4 (90°).
    camera_n_yaws: int = 8
    fast_camera_n_yaws: int = 4
    vlm_base_url: str = VLM_BASE_URL
    vlm_model: str = VLM_MODEL
    use_vlm_ranking: bool = True
    # Retries when contacting the VLM before starting a sample (Mod 11 regen).
    # 0 = do not attempt a connect (use with vlm_optional / --no-vlm template mode).
    vlm_connect_retries: int = 8
    vlm_connect_retry_delay_s: float = 5.0
    # If True, VLM connect failure (or retries=0) is non-fatal: keep GT template
    # question text and continue exporting MCQ rows. CoT polish is skipped.
    vlm_optional: bool = False
    seed: int = 0
    # When True: for each QA category, pick a category-specific mover and ask
    # exactly one question of that type (Mod 4).
    category_driven: bool = True
    # Dataset backend: "ithor" (FloorPlan string) or "procthor" (house dict).
    dataset: str = "ithor"
    # ProcTHOR-only knobs (ignored for ithor).
    house_index: Optional[int] = None
    house_split: str = "train"
    # None = latest ProcTHOR-10K (needs AI2-THOR 5.0+). Pin an old revision only
    # when pairing with an older THOR build.
    procthor_revision: Optional[str] = None
    max_rooms: int = 0  # 0 = all rooms (legacy per-room mode only)
    # ProcTHOR camera mode:
    #   "apartment" (v8): per-room propose (2–3+ views) then global select with
    #     hard room coverage + diversity; emit one sample per camera, fall back
    #     across the apartment camera pool for failed categories → ≈8y QAs.
    #   "per_room" (v7): one sample per room with an in-room camera pool.
    procthor_camera_mode: str = "apartment"
    # y for apartment mode (default 10 → ideally 80 questions with 8 categories).
    procthor_cameras_per_apartment: int = 10
    # Per-room proposal budget before global select (raised automatically when
    # n_rooms * budget < y so studios can still fill).
    procthor_proposals_per_room: int = 3
    # FPS subsample size inside each room polygon during apartment proposal
    # (10 spots × 8 yaws ≈ 80 teleports/room before keeping 2–3 proposals).
    procthor_max_positions_per_room: int = 10
    # Added to view score when a candidate is in a not-yet-covered room.
    procthor_room_diversity_bonus: float = 1000.0
    # Min camera-xyz separation for in-room proposal and global select.
    procthor_camera_min_separation: float = 1.0
    # Legacy per-room knobs (used when procthor_camera_mode == "per_room").
    # ``0`` cameras = auto max(8, n_categories).
    procthor_cameras_per_room: int = 0
    procthor_samples_per_room: int = 1
    # Attach Mod-7 CoT after writing sample.json.
    # Default off: warm/pretended CoT are expensive VLM calls; enable via build
    # ``--with-cot`` / ``--with-pretended-cot`` when needed. VLM question wording
    # still runs when ``use_vlm_ranking=True``.
    attach_cot: bool = False
    # When attach_cot is True, also generate pretended CoT (rgb0-only rewrite).
    attach_pretended_cot: bool = False
    # Mod 27 / risky v7: question wording is text-only (no rgb0/rgb1). Images
    # rarely changed the draft and dominated wall time (~75s/call).
    vlm_word_use_images: bool = False
    # Mod 27 / risky v7: skip VLM referring polish; keep geometric ordinal phrase.
    vlm_referring_polish: bool = False
    # Mod 28 fast mode (ProcTHOR default): closed-form occlusion/contact, fewer
    # closer_to tries, fast feasibility probes, shrunk camera search (15 FPS × 4
    # yaws). None → True for dataset=="procthor", False for ithor.
    fast_find_edit: Optional[bool] = None
    # Pose snapshot restore between categories (skip full house reset).
    # Independent kill switch: set False if rewind quality looks bad while
    # keeping other fast knobs. None → same default as use_fast_mode(cfg).
    snapshot_restore: Optional[bool] = None
    # ---- Yes/No balance targets (None → module globals) ----
    occlude_target_no_frac: Optional[float] = None
    contact_target_no_frac: Optional[float] = None
    feasibility_target_no_frac: Optional[float] = None
    left_right_target_no_frac: Optional[float] = None
    # ---- Wall movement knobs (contact_wall) ----
    # P(slide away from wall) when prefer_contact is unset; else balance drives it.
    # Ignored when ``contact_always_toward`` is True (v10+).
    wall_away_prob: Optional[float] = None
    # Near-wall eligibility band under fast_find_edit (None → WALL_SELECT_NEAR_M_FAST).
    fast_wall_select_near_m: Optional[float] = None
    # v10+: always slide toward nearest wall (no away-from-wall Nos).
    contact_always_toward: Optional[bool] = None
    # Fraction of contact edits that intentionally use half motion (No).
    # None → CONTACT_HALF_MOTION_NO_FRAC when always-toward is on.
    contact_half_motion_no_frac: Optional[float] = None
    # v10 contact: room-wall metadata pre-filter + ranked-candidate search.
    contact_room_wall_prefilter: Optional[bool] = None
    contact_room_wall_max_m: Optional[float] = None
    contact_fast_max_candidates: Optional[int] = None
    contact_fast_single_camera: Optional[bool] = None
    contact_feas_first: Optional[bool] = None
    # Runtime: ProcTHOR rooms list for room-wall contact pre-filter.
    procthor_house_rooms: Optional[list] = None
    # ---- Fast-path try / probe caps (None → FAST_* / FEASIBILITY_FAST_MAX_*) ----
    fast_contact_max_tries: Optional[int] = None
    fast_occlusion_max_tries: Optional[int] = None
    fast_feasibility_max_attempts: Optional[int] = None
    # Subset of QA_CATEGORIES for category-driven builds (None = all eight).
    categories: Optional[tuple[str, ...]] = None
    # Optional flat MCQ training export (dataset.jsonl + images/); sample.json
    # / per-view PNGs are always kept for debug + visualization.
    export_mcq: bool = False
    mcq_jsonl: Optional[Path] = None  # default: out_dir / dataset.jsonl
    mcq_images_dir: Optional[Path] = None  # default: out_dir / images
    mcq_flush_every: int = 10
    # Modification 12: also write SoM images + text_som on every view.
    enable_som: bool = True
    out_dir_som: Optional[Path] = None
    answer_flip_mode: str = "train"
    answer_flip_identical_frac: Optional[float] = None
    modification: str = MODIFICATION_TAG
    # Modification 20: simulator GT rigid-depth object flow (masks + dense flow).
    enable_gt_flow: bool = True
    # Modification 20+: save instance segmentation + depth for every rendered RGB,
    # plus a camera-level seg_id → {category, instance_id} map.
    enable_aux_maps: bool = True


def use_fast_find_edit(cfg: PipelineConfig) -> bool:
    """ProcTHOR defaults to fast mode; iTHOR keeps the slow high-quality path.

    Controls: find_edit (occlusion/contact/closer), feasibility probes, and
    camera search size. Alias: ``use_fast_mode``.
    """
    flag = getattr(cfg, "fast_find_edit", None)
    if flag is not None:
        return bool(flag)
    return str(getattr(cfg, "dataset", "") or "") == "procthor"


def use_fast_mode(cfg: PipelineConfig) -> bool:
    """Alias for ``use_fast_find_edit`` (Mod 28 umbrella fast flag)."""
    return use_fast_find_edit(cfg)


def use_snapshot_restore(cfg: PipelineConfig) -> bool:
    """Whether to rewind poses between categories instead of ``controller.reset``.

    Flag: ``PipelineConfig.snapshot_restore``
      - ``True`` / ``False``: force on/off (kill switch if quality is bad)
      - ``None`` (default): follow ``use_fast_mode(cfg)`` (on for ProcTHOR)
    """
    flag = getattr(cfg, "snapshot_restore", None)
    if flag is not None:
        return bool(flag)
    return use_fast_mode(cfg)
