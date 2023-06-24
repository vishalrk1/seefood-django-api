from django.db import models

# Create your models here.
class ImageClassificationModel(models.Model):
    image = models.ImageField(upload_to='images/')
    classifiedLabel = models.CharField(max_length=255)
