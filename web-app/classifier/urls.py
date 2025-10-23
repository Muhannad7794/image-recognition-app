# web-app/classifier/urls.py
from rest_framework.routers import DefaultRouter
from .views import ClassificationViewSet

# Create a router and register our viewset with it.
router = DefaultRouter()
router.register(r"v1", ClassificationViewSet, basename="classification")

# The API URLs are now determined automatically by the router.
# This will create the '/api/v1/classify/' endpoint.
urlpatterns = router.urls
