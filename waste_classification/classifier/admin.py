from django.contrib import admin
from .models import ClassificationResult

@admin.register(ClassificationResult)
class ClassificationResultAdmin(admin.ModelAdmin):
    list_display = ['id', 'predicted_class', 'confidence', 'is_camera', 'created_at']
    list_filter = ['predicted_class', 'is_camera', 'created_at']
    search_fields = ['predicted_class']
    readonly_fields = ['created_at']
    ordering = ['-created_at']