"""FlowBench: counterfactual spatial questions generated in AI2-THOR.

Library used by the scripts under ``scripts/`` and ``training/``:

- scene control and edit search: ``thor_env``, ``camera_search``, ``edit_search``, ``category_edits``
- question generation: ``qa``, ``distance_categories``, ``occlusion``, ``referring``, ``pipeline``
- ground truth: ``gt_object_flow``, ``spatial_metrics``, ``scene_integrity``
- Set-of-Marks and text: ``som``, ``color_som``, ``move_text``, ``mcq_export``, ``dataset_export``
- curation and release: ``curation``, ``answer_balance``, ``release_text``
- reasoning and evaluation: ``cot``, ``vlm_client``, ``benchmark_eval``, ``eval_formats``

See the repository ``README.md`` for the full pipeline.
"""

__version__ = "1.0.0"
