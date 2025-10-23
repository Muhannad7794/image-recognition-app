# web-app/classifier/views.py
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiTypes
from rest_framework.parsers import MultiPartParser, FormParser
from PIL import Image
import numpy as np
# will uncomment this when the model exists
# from tflite_runtime.interpreter import Interpreter
from .serializers import ImageUploadSerializer


# --- Placeholder Inference Function ---
# Will replace this with the real model logic once it is ready.
def get_model_prediction(image):
    """
    A placeholder function that simulates a model prediction.
    """
    # Get image size (just to show processing)
    width, height = image.size
    print(f"Processing image with size: {width}x{height}")

    # Return a FAKE prediction
    predictions = {
        "prediction": "Mug",
        "confidence": 0.85,
        "message": "This is a placeholder response. Model not yet loaded.",
    }
    return predictions


# ----------------------------------------


@extend_schema(tags=["Classification"])
class ClassificationViewSet(viewsets.ViewSet):
    """
    A ViewSet for uploading an image to classify.
    """

    serializer_class = ImageUploadSerializer
    parser_classes = (MultiPartParser, FormParser)

    @extend_schema(
        summary="Classify an Image",
        description="Upload an image (from file or webcam) to get a classification.",
        request=ImageUploadSerializer,
        responses={200: '{"prediction": "Mug", "confidence": 0.85}'},
    )
    @action(detail=False, methods=["post"], url_path="classify")
    def classify_image(self, request):
        serializer = ImageUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        # Get the uploaded image from the serializer
        image_file = serializer.validated_data["image"]

        try:
            # Open the image using Pillow
            image = Image.open(image_file)

            # --- Run Inference ---
            # Just calling the placeholder function for now.
            # Later, will replace this with the real TFLite model.
            prediction_data = get_model_prediction(image)
            # ---------------------

            return Response(prediction_data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {"error": f"Failed to process image: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
