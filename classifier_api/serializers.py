from rest_framework import serializers
from .models import ImageClassificationModel

class ImageClassificationResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageClassificationModel
        fields = ('image', 'classifiedLabel')