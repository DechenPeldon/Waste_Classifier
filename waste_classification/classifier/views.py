import base64
from io import BytesIO
from django.shortcuts import render, get_object_or_404
from django.core.files.base import ContentFile
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from PIL import Image
from .models import ClassificationResult
import time
import tensorflow as tf
from pathlib import Path
import logging
import re

# Teachable Machine model (Keras .h5) and labels loader
MODEL = None
CLASS_NAMES = []
MODEL_INPUT_SIZE = (224, 224)  # default, will attempt to infer from model
MODEL_DIR = Path(__file__).resolve().parent / 'ml_model'
MODEL_FILE = MODEL_DIR / 'keras_model.h5'
LABELS_FILE = MODEL_DIR / 'labels.txt'
SAVED_MODEL_DIR = MODEL_DIR / 'saved_model'

def load_model_and_labels():
    """Lazily load the Keras model and labels from the `ml_model` folder."""
    global MODEL, CLASS_NAMES, MODEL_INPUT_SIZE
    if MODEL is not None:
        return MODEL

    # Prefer SavedModel directory (portable across TF versions). We exported
    # a SavedModel during setup; load it when present. If it's not present,
    # keep MODEL as None so callers can handle the missing model case.
    try:
        if SAVED_MODEL_DIR.exists():
            # Load TF SavedModel and wrap it to expose a .predict(numpy_array) API
            try:
                loaded = tf.saved_model.load(str(SAVED_MODEL_DIR))

                class SavedModelWrapper:
                    def __init__(self, loaded):
                        self._loaded = loaded
                        # Choose serving_default if available
                        signatures = getattr(self._loaded, 'signatures', None) or {}
                        if 'serving_default' in signatures:
                            self._fn = signatures['serving_default']
                        else:
                            # pick first available signature
                            vals = list(signatures.values())
                            if vals:
                                self._fn = vals[0]
                            else:
                                self._fn = None

                    def predict(self, x):
                        # x is expected as a numpy array
                        if self._fn is None:
                            raise RuntimeError('SavedModel has no callable signatures')
                        # Determine signature input names (kwargs) if present
                        sig_kwargs = self._fn.structured_input_signature[1]
                        import numpy as _np

                        if sig_kwargs:
                            # pass as the first kwarg name
                            name = next(iter(sig_kwargs.keys()))
                            tf_input = tf.constant(x, dtype=tf.float32)
                            out = self._fn(**{name: tf_input})
                        else:
                            tf_input = tf.constant(x, dtype=tf.float32)
                            out = self._fn(tf_input)

                        # signature outputs may be a dict or tensor
                        if isinstance(out, dict):
                            out_vals = list(out.values())
                            result = out_vals[0].numpy()
                        else:
                            try:
                                result = out.numpy()
                            except Exception:
                                # Fallback: convert to numpy via tf.convert_to_tensor
                                result = tf.convert_to_tensor(out).numpy()
                        return result

                MODEL = SavedModelWrapper(loaded)
                logging.getLogger(__name__).info('Loaded SavedModel and created wrapper')
            except Exception:
                logging.getLogger(__name__).exception('Failed to load SavedModel')
                MODEL = None

            # Can't reliably infer input_shape from a SavedModel wrapper; keep default

            # Load class labels (one per line) -- use only labels.txt
            CLASS_NAMES = []
            if LABELS_FILE.exists():
                try:
                    with open(LABELS_FILE, 'r', encoding='utf-8') as f:
                        for line in f:
                            raw = line.strip()
                            if not raw:
                                continue
                            cleaned = re.sub(r'^\s*\d+\s*[\.\-:)]*\s*', '', raw)
                            cleaned = cleaned.strip()
                            if cleaned:
                                CLASS_NAMES.append(cleaned)
                except Exception as e:
                    logging.getLogger(__name__).exception('Failed reading labels.txt: %s', e)
        else:
            MODEL = None
    except Exception:
        logging.getLogger(__name__).exception('Failed to load SavedModel')
        MODEL = None

    return MODEL

def preprocess_image(image, target_size=None):
    """Preprocess image for the Teachable Machine model.

    This resizes to the model input size and scales pixel values to [0,1].
    Adjust this if your model expects a different preprocessing pipeline.
    """
    if target_size is None:
        target_size = MODEL_INPUT_SIZE

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = image.resize(target_size)
    img_array = np.array(image).astype('float32')
    # Teachable Machine-exported Keras models typically expect values in [0,1]
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def predict_waste(image):
    """Predict waste class from image using the loaded Teachable Machine model."""
    model = load_model_and_labels()

    if model is None:
        # Model missing — return an explicit label rather than None.
        logging.getLogger(__name__).warning('Model file not loaded; returning Model Unavailable')
        return 'Model Unavailable', 0.0

    # Note: we always return a label from the model (mapping via labels.txt when available).
    # Any prediction index will be mapped to a label from `labels.txt` when present.

    try:
        preprocessed = preprocess_image(image)
        preds = model.predict(preprocessed)

        # preds shape usually (1, num_classes)
        probs = preds[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx]) * 100.0

        # Map prediction index to label using labels loaded from labels.txt
        if idx < len(CLASS_NAMES):
            label = CLASS_NAMES[idx]
        else:
            # Fallback to numeric label if labels.txt doesn't contain the index
            label = str(idx)

        # Return a cleaned, human-friendly label (ensure string)
        return str(label), confidence

    except Exception as e:
        logging.getLogger(__name__).exception('Error during prediction: %s', e)
        # On unexpected errors return a safe string label (avoid None)
        return 'Unknown', 0.0

def home(request):
    """Home page view"""
    return render(request, 'home.html')

def about(request):
    """About page view"""
    return render(request, 'about.html')

@csrf_exempt
def classify_waste(request):
    """Handle waste classification from uploaded image or camera"""
    if request.method == 'POST':
        try:
            is_camera = False
            
            # Check if image is from camera (base64) or file upload
            if 'image' in request.FILES:
                # File upload
                uploaded_file = request.FILES['image']
                image = Image.open(uploaded_file)
                
            elif 'image_data' in request.POST:
                # Camera capture (base64)
                is_camera = True
                image_data = request.POST['image_data']
                
                # Remove data URL prefix if present
                if 'base64,' in image_data:
                    image_data = image_data.split('base64,')[1]
                
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))
            else:
                return JsonResponse({'error': 'No image provided'}, status=400)
            
            # Predict
            predicted_class, confidence = predict_waste(image)

            # Save image
            img_io = BytesIO()
            # Ensure image is in RGB mode before saving as JPEG (PIL can't write RGBA as JPEG)
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')

            # For grayscale 'L' we still save as JPEG; PIL will handle it.
            image.save(img_io, format='JPEG', quality=95)
            img_io.seek(0)
            
            # Create result object
            result = ClassificationResult()
            # Ensure predicted_class is always a string (never None)
            result.predicted_class = str(predicted_class)
            result.confidence = confidence
            result.is_camera = is_camera
            # Use a stable filename that doesn't rely on result.id (which is None until saved)
            filename = f'waste_{int(time.time()*1000)}.jpg'
            result.image.save(filename, ContentFile(img_io.read()), save=False)
            result.save()
            
            return JsonResponse({
                'success': True,
                'result_id': result.id,
                'redirect_url': f'/result/{result.id}/'
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def result(request, result_id):
    """Display classification result"""
    result = get_object_or_404(ClassificationResult, id=result_id)
    waste_info = result.get_waste_description()
    
    context = {
        'result': result,
        'waste_info': waste_info,
    }
    
    return render(request, 'result.html', context)


# Warm-load the model on module import so the development server initializes the
# model at startup rather than waiting for the first request. This reduces the
# chance a request sees `Model Unavailable` due to lazy loading on Windows.
try:
    load_model_and_labels()
except Exception:
    logging.getLogger(__name__).exception('Warm model load failed')