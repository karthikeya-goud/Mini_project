from django.db import models

class Video(models.Model):
    video_file = models.FileField(upload_to='videos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class Metadata(models.Model):
    video = models.OneToOneField(Video, on_delete=models.CASCADE)
    detected_actions = models.JSONField(default=list)
    confidence_scores = models.JSONField(default=list)
    timestamp = models.DateTimeField(auto_now_add=True)