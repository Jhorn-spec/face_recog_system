from django import forms
from face_recog_app.models import FaceRecognition

class FaceRecognitionForm(forms.ModelForm):
    class Meta:
        model = FaceRecognition
        fields = ['id', 'image']
