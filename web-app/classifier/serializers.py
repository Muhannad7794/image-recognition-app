# web-app/classifier/serializers.py
from rest_framework import serializers


class ImageUploadSerializer(serializers.Serializer):
    """
    Serializer for accepting an image file upload.
    """

    image = serializers.ImageField()
