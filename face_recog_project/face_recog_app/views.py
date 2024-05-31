from django.shortcuts import render
from django.http import HttpResponse
from face_recog_app.forms import FaceRecognitionForm
from face_recog_app.computervision import register_face, face_detect
from django.conf import settings
from face_recog_app.models import FaceRecognition
import os
# Create your views here.

def index(request):
    form = FaceRecognitionForm()


    if request.method == 'POST':
        form = FaceRecognitionForm(request.POST or None, request.FILES or None)
        if form.is_valid():
            # id = form.cleaned_data['id']
            # image = form.cleaned_data['image']
            uploaded_image = form.save(commit=True)
            image_path = uploaded_image.image.path

            # Process the image here
            result = register_face(id=uploaded_image.id, image_path=image_path, live=False)

            context = {
                'result': result,
                'success': result['success']
            }

            return render(request, 'result.html', context)
    else:
        form = FaceRecognitionForm()


    return render(request, 'index.html', {'form': form})
