# Publics in Motion

Spatial-behavior research artifacts and video heatmap experiments from a Design Analytics project.

## Start here

- [Project presentation](PublicsInMotion_DAFinalPresentation.pdf)
- [Final report](Publics%20In%20Motion%20-%20Design%20Analytics%20Final%20Project%20Report.docx)
- [Workflow notes](workflow.docx)
- [`data/`](data/) — exported summary tables.
- [`script/`](script/) — two OpenCV video-overlay experiments.

This repository preserves selected project artifacts. It is not a complete reproduction package: the original input video and all upstream processing steps are not included.

## What the scripts measure

| Script | Method | Interpretation |
| --- | --- | --- |
| `flow_speed_heatmap.py` | Dense optical-flow magnitude with decaying accumulation | Relative image motion, not calibrated pedestrian speed |
| `presence_density_heatmap.py` | HSV thresholding of green markers with decaying accumulation | Marker presence, not general-purpose person detection |

Both normalize their heatmap per frame. Colors are relative visualizations, not an absolute scale for comparisons across frames or recordings. Original AI-assistance comments are retained in the source.

## Run a heatmap experiment

Use Python 3 with OpenCV and NumPy:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Supply your own appropriately permissioned video. In the chosen script, set `video_path` and `output_path` to your local paths, then run:

```sh
python3 script/flow_speed_heatmap.py
# Or, for video containing green tracking markers:
python3 script/presence_density_heatmap.py
```

Existing defaults refer to the author's original workstation and must be changed before running. Use distinct input and output paths. For the presence visualization, green objects other than tracking markers will also be detected.
