from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import WasteClassification
from .utils import classifier, get_waste_info
import os
import base64
from io import BytesIO
from PIL import Image

def home(request):
    return render(request, 'classifier/home.html')

def about(request):
    return render(request, 'classifier/about.html')

@csrf_exempt
def classify_image(request):
    if request.method == 'POST':
        try:
            # Handle file upload
            if 'image' in request.FILES:
                image_file = request.FILES['image']
                fs = FileSystemStorage()
                filename = fs.save(image_file.name, image_file)
                image_path = fs.path(filename)
                image_url = fs.url(filename)
            
            # Handle camera capture (base64)
            elif 'image_data' in request.POST:
                image_data = request.POST['image_data']
                format, imgstr = image_data.split(';base64,')
                ext = format.split('/')[-1]
                
                image_bytes = base64.b64decode(imgstr)
                image = Image.open(BytesIO(image_bytes))
                
                fs = FileSystemStorage()
                filename = f'camera_capture_{os.urandom(8).hex()}.{ext}'
                
                # Save the image
                image_buffer = BytesIO()
                image.save(image_buffer, format='JPEG')
                image_buffer.seek(0)
                
                saved_filename = fs.save(filename, image_buffer)
                image_path = fs.path(saved_filename)
                image_url = fs.url(saved_filename)
            else:
                return JsonResponse({'error': 'No image provided'}, status=400)
            
            # Perform classification
            prediction = classifier.predict(image_path)
            
            if prediction:
                # Save to database
                classification = WasteClassification.objects.create(
                    image=image_url.lstrip('/media/'),
                    predicted_class=prediction['class'],
                    confidence=prediction['confidence']
                )
                
                # Get waste information
                waste_info = get_waste_info(prediction['class'])
                
                # Store in session for result page
                request.session['last_prediction'] = {
                    'id': classification.id,
                    'class': prediction['class'],
                    'confidence': prediction['confidence'],
                    'image_url': image_url,
                    'all_predictions': prediction['all_predictions'],
                    'info': waste_info
                }
                
                return JsonResponse({
                    'success': True,
                    'redirect_url': '/result/'
                })
            else:
                return JsonResponse({'error': 'Classification failed'}, status=500)
                
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=400)

def result(request):
    prediction_data = request.session.get('last_prediction')
    
    if not prediction_data:
        return redirect('home')
    
    context = {
        'prediction': prediction_data
    }
    
    return render(request, 'classifier/result.html', context)