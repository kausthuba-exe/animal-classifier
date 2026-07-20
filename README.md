# Animal Classifier

A simple convolutional neural network, trained from scratch (no pretrained weights), that classifies images into six categories: **dog, cat, human, pig, lion, tiger**.

## Project structure

```
├── train.py           # builds and trains the CNN
├── predict.py          # runs inference on a single image
├── data/                # training images, organized by class (not tracked in git)
│   ├── dog/
│   ├── cat/
│   ├── Humans/
│   ├── pig/
│   ├── lion/
│   └── tiger/
├── animal_classifier.keras   # saved trained model (not tracked in git)
└── class_names.txt           # class labels used by predict.py
```

## Setup

```bash
pip install tensorflow
```

## Training

Place labeled images in `data/<class_name>/`, then run:

```bash
python train.py
```

This trains a small CNN (3 convolutional blocks + dense head) for 15 epochs and saves:
- `animal_classifier.keras` — the trained model
- `class_names.txt` — the class labels, in the order the model was trained on

## Predicting

```bash
python predict.py path/to/image.jpg
```

Outputs the predicted class along with a confidence breakdown across all six categories.

## Notes

- Model is trained from scratch — no transfer learning or pretrained weights.
- Recommended: at least 100–200+ images per class, roughly balanced, for reasonable accuracy.
- `data/` and `*.keras` are excluded from version control via `.gitignore` since datasets and trained models can be large — re-run `train.py` locally to regenerate them.