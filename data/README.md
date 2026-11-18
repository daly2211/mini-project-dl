# Data

Expected layout:

```
data/raw/
├── metadata.csv               # WILDS Waterbirds v1.0 metadata
├── 001.Black_footed_Albatross/
├── ...                        # 200 CUB species folders
├── cub_annotations/
│   ├── images.txt             # CUB image id -> filename
│   └── bounding_boxes.txt     # CUB image id -> x y width height
```

`metadata.csv` columns: `img_id, img_filename, y, split, place, place_filename`.
`y` is the bird label (0 = landbird, 1 = waterbird), `place` is the background
(0 = land, 1 = water). `split` is 0/1/2 and is mapped to train/val/test. Group
id is `2*y + place`:

| group | bird | background | split role |
|---|---|---|---|
| 0 | landbird | land | majority |
| 1 | landbird | water | minority |
| 2 | waterbird | land | minority |
| 3 | waterbird | water | majority |

WILDS does not distribute bounding boxes, so `src/dataset.py` joins
CUB-200-2011 boxes by filename (the Waterbirds images are CUB images). CUB boxes
are `x, y, width, height` in original-image pixels and are used as an
approximation of the bird region for the background attribution ratio.

Sources:
- Waterbirds: <https://huggingface.co/datasets/pangwei/wilds-waterbirds>
  (Sagawa et al., 2020; WILDS release v1.0).
- CUB annotations: <https://github.com/ZF4444/MMAL-Net/tree/master/datasets/CUB_200_2011>
  (CUB-200-2011, Wah et al., 2011).

Raw data is not committed to this repository. Both datasets are for
non-commercial research/educational use.
