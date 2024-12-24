import cv2
import numpy as np

def extract_frames(video_path, img_size, sequence_length):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)

    if not video_reader.isOpened():
        print(f"Failed to open video: {video_path}")
        return frames_list

    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / sequence_length), 1)

    for frame_counter in range(sequence_length):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (img_size, img_size))
        normalized_frame = resized_frame.astype('float32') / 255.0
        frames_list.append(normalized_frame)

    video_reader.release()
    return frames_list

def predict_action(frame_sequence, model, CLASSES):
    frame_sequence = np.expand_dims(frame_sequence, axis=0)
    pred = model.predict(frame_sequence)
    action_idx = np.argmax(pred)
    confidence = float(pred[0][action_idx])
    return CLASSES[action_idx], confidence