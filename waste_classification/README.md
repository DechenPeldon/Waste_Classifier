# Waste Classification

A lightweight Django app that classifies images of waste into categories (paper, plastic, organic, glass, e-waste, etc.) using a Keras/TensorFlow model.

This repository contains:
- `classifier/` — Django app with views, models and the ML integration.
- `classifier/ml_model/` — trained model and `labels.txt` (authoritative label list).
- `templates/`, `static/`, and Django project files to run the web UI.

**Quick Goals**
- Use `classifier/ml_model/labels.txt` as the authoritative label list.
- Prefer a TensorFlow SavedModel for cross-platform reliability (`classifier/ml_model/saved_model`).
- Keep web UI simple: upload an image or use camera capture, then view the classification result.

**Features**
- Image upload and camera capture support.
- Model inference via TensorFlow SavedModel (wrapped to provide `.predict()` compatibility).
- Friendly descriptions for recognized waste classes (editable in code or move to a JSON file).

**Prerequisites**
- Python 3.10+ (the repo was tested with Python 3.10 on Windows)
- A virtual environment (recommended)

**Setup (Windows using Bash)**
```bash
cd "/c/Users/Dell/Desktop/New folder/waste_classification"
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Database & Static files**
```bash
python manage.py migrate
python manage.py collectstatic --noinput
```

**Run (development)**
```bash
python manage.py runserver
# Open http://127.0.0.1:8000/ in your browser
```

**How to use**
- Open the home page.
- Upload an image or use your camera capture (if available in your browser).
- The app returns a result page showing the predicted label and a short description.

**Model & Labels**
- `classifier/ml_model/labels.txt` — this file is the authoritative list of labels; each non-empty line is treated as one label (the app strips numeric prefixes like `0 ` when parsing).
- `classifier/ml_model/saved_model/` — the preferred model format for runtime inference (portable across platforms).
- `classifier/ml_model/keras_model.h5` — legacy H5 model exported from training. The app used a tolerant loader while converting to SavedModel; the H5 can be kept as a backup or removed once SavedModel is in place.

**If you need to re-export a SavedModel locally**
Use this snippet (it handles the DepthwiseConv2D `groups` issue if the H5 was exported on a different TF/Keras):

```bash
python - <<'PY'
import tensorflow as tf
from pathlib import Path
MODEL_FILE = Path('classifier/ml_model/keras_model.h5')
SAVED_DIR = Path('classifier/ml_model/saved_model')

# Optionally adjust this wrapper to match any other problematic layer types
try:
    try:
        model = tf.keras.models.load_model(str(MODEL_FILE))
    except Exception:
        class DepthwiseConv2DWrapper(tf.keras.layers.DepthwiseConv2D):
            @classmethod
            def from_config(cls, config):
                config = dict(config)
                config.pop('groups', None)
                return super().from_config(config)
        model = tf.keras.models.load_model(str(MODEL_FILE), custom_objects={'DepthwiseConv2D': DepthwiseConv2DWrapper})

    # Use tf.saved_model.save to export a legacy SavedModel that can be loaded by TF APIs
    if SAVED_DIR.exists():
        import shutil
        shutil.rmtree(SAVED_DIR)
    tf.saved_model.save(model, str(SAVED_DIR))
    print('SavedModel exported to', SAVED_DIR)
except Exception as e:
    print('Export failed:', e)
    raise
PY
```

**Troubleshooting**
- "Unrecognized keyword arguments passed to DepthwiseConv2D: {'groups': 1}" when loading an H5
  - Cause: model H5 was exported by a different Keras/TensorFlow version and includes keys not expected by the runtime.
  - Fix: re-export to SavedModel on the original environment, or use the tolerant wrapper shown above to load-and-export.

- "cannot write mode RGBA as JPEG"
  - Fix: images are converted to RGB before saving; ensure uploads are valid image files.

- Model shows "Model Unavailable"
  - Confirm `classifier/ml_model/saved_model/` exists and is readable by the app process.
  - If not present, create it using the export snippet above or ensure `keras_model.h5` exists and re-run the export.

**Developer notes**
- Labels: the code normalizes and strips numeric prefixes from `labels.txt`. If you'd like to clean the file visually, feel free to remove the numeric indices — the app will work either way.
- Descriptions: current human-friendly descriptions live in `classifier/models.py` inside `get_waste_description()`; consider moving them to `classifier/ml_model/descriptions.json` for easier editing.
- Production: serve the SavedModel via TF Serving or a model-serving endpoint if you scale beyond a simple Django-hosted inference.

**Contributing**
- Fork, branch, and open a PR. Keep changes small and test with `python manage.py runserver` locally.

**License & Contact**
- This project doesn't include a license file by default. Add one if you plan to publish the repo.
- If you want me to make further changes (cleanup, delete the H5, or externalize descriptions), tell me which step to take next.
