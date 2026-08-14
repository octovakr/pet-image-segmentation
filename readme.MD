## Project Summary

Codes for training pet-image tri-class segmentation model and generating submission file
(Yonsei 2026 Spring, CAS3120 MACHINE LEARNING Programming Assignment)

## Main files

- `data.py` : data loader & preprocessing
- `main.py` : training & inference
- `make_submission.py` : turn the prediction files into a csv file
- `ensemble_and_postprocessed/` : random seed ensemble (different data split) & postprocessing(filling holes inside the boundary-loop of the prediction)

## Requirements

- Python 3.8+
- Install required packaged referring to the import statements.

## Run Examples

- train:

```
python main.py
```

- generate submission file:

```
python make_submission.py
```

## Saved Model / Results

Best model saved at:
2026/05/29 16:28
(mIoU on kaggle: 0.83996)

ensemble_and_postprocessed saved at:
2026/05/31 18:32
(mIoU on kaggle: 0.83506)
