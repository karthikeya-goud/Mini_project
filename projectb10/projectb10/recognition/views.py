import cv2
import numpy as np
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from .forms import VideoUploadForm
from .models import Video, Metadata
from .utils import extract_frames, predict_action
from .human_action_model import model, CLASSES

def index(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.save()
            video_path = video.video_file.path

            # Extract frames and predict action
            frames = extract_frames(video_path, img_size=64, sequence_length=40)
            if frames:
                detected_action, confidence = predict_action(np.array(frames), model, CLASSES)
                Metadata.objects.create(
                    video=video,
                    detected_actions=[detected_action],
                    confidence_scores=[confidence]
                )
            else:
                Metadata.objects.create(
                    video=video,
                    detected_actions=["No action detected"],
                    confidence_scores=[0.0]
                )

            return redirect('results', video_id=video.id)
    else:
        form = VideoUploadForm()
    return render(request, 'recognition/index.html', {'form': form})

def results(request, video_id):
    video = get_object_or_404(Video, id=video_id)
    metadata = get_object_or_404(Metadata, video=video)
    return render(request, 'recognition/results.html', {'video': video, 'metadata': metadata})

def fetch_previous_results(request):
    if request.method == 'GET':
        videos = Video.objects.all().order_by('-uploaded_at')
        results = []
        for video in videos:
            metadata = Metadata.objects.filter(video=video).first()
            if metadata:
                results.append({
                    'video_id': video.id,
                    'video_url': video.video_file.url,
                    'detected_actions': metadata.detected_actions,
                    'confidence_scores': metadata.confidence_scores,
                    'timestamp': metadata.timestamp
                })
        return JsonResponse(results, safe=False)