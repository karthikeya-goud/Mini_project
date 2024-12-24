from django.contrib import admin

# Register your models here.
from recognition.models import Video,Metadata

class VideoAdmin(admin.ModelAdmin):
    list_display=["video_file","uploaded_at"]

class MetadataAdmin(admin.ModelAdmin):
    list_display=['video','detected_actions','confidence_scores','timestamp']

admin.site.register(Video,VideoAdmin)
admin.site.register(Metadata,MetadataAdmin)