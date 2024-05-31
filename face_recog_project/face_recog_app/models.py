from django.db import models

# Create your models here.
class FaceRecognition(models.Model):
    id = models.CharField(max_length=50, primary_key=True)
    image = models.ImageField(upload_to='images/')

    def __str__(self):
        return str(self.id)