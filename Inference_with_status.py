import warnings
warnings.filterwarnings("ignore")
import os
import torch
import torchaudio
import numpy as np
from moviepy import *
from PIL import Image, ImageDraw
import face_alignment
import cv2

from look2hear.models import Dolphin
from look2hear.datas.transform import get_preprocessing_pipelines

from face_detection_utils import detect_faces

# Import functions from original Inference.py
from Inference import (
    linear_interpolate, warp_img, apply_transform, cut_patch, convert_bgr2gray,
    save2npz, read_video, face2head, bb_intersection_over_union, 
    landmarks_interpolate, crop_patch, convert_video_fps, extract_audio, merge_video_audio
)

def detectface_with_status(video_input_path, output_path, detect_every_N_frame, scalar_face_detection, number_of_speakers, status_callback=None):
    """Face detection with status updates"""
    device = torch.device('cuda' if torch.cuda.get_device_name() else 'cpu')
    if status_callback:
        status_callback({'status': f'Running on device: {device}', 'progress': 0.0})
    
    os.makedirs(os.path.join(output_path, 'faces'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'landmark'), exist_ok=True)

    landmarks_dic = {}
    faces_dic = {}
    boxes_dic = {}
    
    for i in range(number_of_speakers):
        landmarks_dic[i] = []
        faces_dic[i] = []
        boxes_dic[i] = []

    video_clip = VideoFileClip(video_input_path)
    if status_callback:
        status_callback({'status': f"Video: {video_clip.w}x{video_clip.h}, {video_clip.fps}fps", 'progress': 0.05})
    
    frames = [Image.fromarray(frame) for frame in video_clip.iter_frames()]
    total_frames = len(frames)
    if status_callback:
        status_callback({'status': f'Processing {total_frames} frames', 'progress': 0.1})
    
    video_clip.close()
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    
    for i, frame in enumerate(frames):
        if status_callback and i % 10 == 0:
            status_callback({'status': f'Tracking frame: {i+1}/{total_frames}', 'progress': 0.1 + 0.3 * (i / total_frames)})
        
        # Detect faces every N frames
        if i % detect_every_N_frame == 0:
            frame_array = np.array(frame)

            detected_boxes, _ = detect_faces(
                frame_array,
                threshold=0.9,
                allow_upscaling=False,
            )

            if detected_boxes is None or len(detected_boxes) == 0:
                detected_boxes, _ = detect_faces(
                    frame_array,
                    threshold=0.7,
                    allow_upscaling=True,
                )

            if detected_boxes is not None and len(detected_boxes) > 0:
                detected_boxes = np.asarray(detected_boxes, dtype=np.float32)
                areas = (detected_boxes[:, 2] - detected_boxes[:, 0]) * (detected_boxes[:, 3] - detected_boxes[:, 1])
                sort_idx = np.argsort(areas)[::-1]
                detected_boxes = detected_boxes[sort_idx][:number_of_speakers]
                detected_boxes = face2head(detected_boxes, scalar_face_detection)
                detected_boxes = [box for box in detected_boxes]
            else:
                detected_boxes = []

        # Process the detection results (same as original)
        if i == 0:
            # First frame - initialize tracking
            if len(detected_boxes) < number_of_speakers:
                raise ValueError(f"First frame must detect at least {number_of_speakers} faces, but only found {len(detected_boxes)}")
            
            # Assign first detections to speakers in order
            for j in range(number_of_speakers):
                box = detected_boxes[j]
                face = frame.crop((box[0], box[1], box[2], box[3])).resize((224,224))
                preds = fa.get_landmarks(np.array(face))
                
                if preds is None:
                    raise ValueError(f"Face landmarks not detected in initial frame for speaker {j}")
                
                faces_dic[j].append(face)
                landmarks_dic[j].append(preds)
                boxes_dic[j].append(box)
        else:
            # For subsequent frames, match detected boxes to speakers
            matched_speakers = set()
            speaker_boxes = [None] * number_of_speakers
            
            # Match each detected box to the most likely speaker
            for box in detected_boxes:
                iou_scores = []
                for speaker_id in range(number_of_speakers):
                    if speaker_id in matched_speakers:
                        iou_scores.append(-1)  # Already matched
                    else:
                        last_box = boxes_dic[speaker_id][-1]
                        iou_score = bb_intersection_over_union(box, last_box)
                        iou_scores.append(iou_score)
                
                if max(iou_scores) > 0:  # Valid match found
                    best_speaker = iou_scores.index(max(iou_scores))
                    speaker_boxes[best_speaker] = box
                    matched_speakers.add(best_speaker)
            
            # Process each speaker
            for speaker_id in range(number_of_speakers):
                if speaker_boxes[speaker_id] is not None:
                    # Use detected box
                    box = speaker_boxes[speaker_id]
                else:
                    # Use previous box for this speaker
                    box = boxes_dic[speaker_id][-1]
                
                # Extract face and landmarks
                face = frame.crop((box[0], box[1], box[2], box[3])).resize((224,224))
                preds = fa.get_landmarks(np.array(face))
                
                if preds is None:
                    # Use previous landmarks if detection fails
                    preds = landmarks_dic[speaker_id][-1]
                
                faces_dic[speaker_id].append(face)
                landmarks_dic[speaker_id].append(preds)
                boxes_dic[speaker_id].append(box)
    
    # Verify all speakers have same number of frames
    frame_counts = [len(boxes_dic[s]) for s in range(number_of_speakers)]
    if status_callback:
        status_callback({'status': f"Frame counts per speaker: {frame_counts}", 'progress': 0.4})
    
    assert all(count == len(frames) for count in frame_counts), f"Inconsistent frame counts: {frame_counts}"
    
    # Continue with saving videos and landmarks...
    for s in range(number_of_speakers):
        if status_callback:
            status_callback({'status': f'Saving tracked video for speaker {s+1}', 'progress': 0.4 + 0.1 * (s / number_of_speakers)})
        
        frames_tracked = []
        for i, frame in enumerate(frames):
            frame_draw = frame.copy()
            draw = ImageDraw.Draw(frame_draw)
            draw.rectangle(boxes_dic[s][i], outline=(255, 0, 0), width=6) 
            frames_tracked.append(frame_draw)
            
        # Save tracked video
        tracked_frames = [np.array(frame) for frame in frames_tracked]
        if tracked_frames:
            tracked_clip = ImageSequenceClip(tracked_frames, fps=25.0)
            tracked_video_path = os.path.join(output_path, 'video_tracked' + str(s+1) + '.mp4')
            tracked_clip.write_videofile(tracked_video_path, codec='libx264', audio=False, logger=None)
            tracked_clip.close()

    # Save landmarks
    for i in range(number_of_speakers):
        # Create landmark directory if it doesn't exist
        landmark_dir = os.path.join(output_path, 'landmark')
        os.makedirs(landmark_dir, exist_ok=True)
        save2npz(os.path.join(landmark_dir, 'speaker' + str(i+1)+'.npz'), data=landmarks_dic[i])
        
        # Save face video
        face_frames = [np.array(frame) for frame in faces_dic[i]]
        if face_frames:
            face_clip = ImageSequenceClip(face_frames, fps=25.0)
            face_video_path = os.path.join(output_path, 'faces', 'speaker' + str(i+1) + '.mp4')
            face_clip.write_videofile(face_video_path, codec='libx264', audio=False, logger=None)
            face_clip.close()

    # Output video path
    parts = video_input_path.split('/')
    video_name = parts[-1][:-4]
    filename_dir = os.path.join(output_path, 'filename_input')
    os.makedirs(filename_dir, exist_ok=True)
    csvfile = open(os.path.join(filename_dir, str(video_name) + '.csv'), 'w')
    for i in range(number_of_speakers):
        csvfile.write('speaker' + str(i+1)+ ',0\n')
    csvfile.close()
    return os.path.join(filename_dir, str(video_name) + '.csv')


def crop_mouth_with_status(video_direc, landmark_direc, filename_path, save_direc, status_callback=None, convert_gray=False, testset_only=False):
    """Crop mouth with status updates"""
    lines = open(filename_path).read().splitlines()
    lines = list(filter(lambda x: 'test' in x, lines)) if testset_only else lines

    for filename_idx, line in enumerate(lines):
        filename, person_id = line.split(',')
        
        if status_callback:
            status_callback({'status': f'Processing speaker{int(person_id)+1}', 'progress': 0.5 + 0.1 * filename_idx / len(lines)})

        video_pathname = os.path.join(video_direc, filename+'.mp4')
        landmarks_pathname = os.path.join(landmark_direc, filename+'.npz')
        
        # Create mouthroi directory if it doesn't exist
        os.makedirs(save_direc, exist_ok=True)
        dst_pathname = os.path.join(save_direc, filename+'.npz')

        multi_sub_landmarks = np.load(landmarks_pathname, allow_pickle=True)['data']
        if len(multi_sub_landmarks) == 0:
            print(f"No landmarks found for {filename}, skipping crop.")
            continue

        landmark_frame_count = len(multi_sub_landmarks)
        cap = cv2.VideoCapture(video_pathname)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        if frame_count > 0 and frame_count != landmark_frame_count:
            print(
                f"Frame count mismatch for {filename}: video has {frame_count} frames, "
                f"landmarks have {landmark_frame_count} entries. Adjusting to match."
            )
            if frame_count < landmark_frame_count:
                multi_sub_landmarks = multi_sub_landmarks[:frame_count]
            else:
                pad_count = frame_count - landmark_frame_count
                pad = np.repeat(multi_sub_landmarks[-1:], pad_count, axis=0)
                multi_sub_landmarks = np.concatenate((multi_sub_landmarks, pad), axis=0)

        landmarks = [None] * len(multi_sub_landmarks)
        for frame_idx in range(len(landmarks)):
            try:
                landmarks[frame_idx] = multi_sub_landmarks[frame_idx][int(person_id)]
            except (IndexError, TypeError):
                continue

        # Pre-process landmarks: interpolate frames not being detected
        preprocessed_landmarks = landmarks_interpolate(landmarks)
        if not preprocessed_landmarks:
            continue

        # Crop
        mean_face_landmarks = np.load('assets/20words_mean_face.npy')
        sequence = crop_patch(mean_face_landmarks, video_pathname, preprocessed_landmarks, 12, 48, 68, 96, 96)
        assert sequence is not None, "cannot crop from {}.".format(filename)

        # Save
        data = convert_bgr2gray(sequence) if convert_gray else sequence[...,::-1]
        save2npz(dst_pathname, data=data)


def process_video_with_status(input_file, output_path, number_of_speakers=2, 
                             detect_every_N_frame=8, scalar_face_detection=1.5,
                             config_path="checkpoints/vox2/conf.yml",
                             cuda_device=None, status_callback=None):
    """Main processing function with status updates"""
    
    # Set CUDA device if specified
    if cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Convert video to 25fps
    if status_callback:
        status_callback({'status': 'Converting video to 25fps', 'progress': 0.0})
    
    temp_25fps_file = os.path.join(output_path, 'temp_25fps.mp4')
    convert_video_fps(input_file, temp_25fps_file, target_fps=25)
    
    # Detect faces
    if status_callback:
        status_callback({'status': 'Detecting faces and tracking speakers', 'progress': 0.1})
    
    filename_path = detectface_with_status(
        video_input_path=temp_25fps_file, 
        output_path=output_path, 
        detect_every_N_frame=detect_every_N_frame, 
        scalar_face_detection=scalar_face_detection, 
        number_of_speakers=number_of_speakers,
        status_callback=status_callback
    )
    torch.cuda.empty_cache()
    # Extract audio
    if status_callback:
        status_callback({'status': 'Extracting audio from video', 'progress': 0.5})
    
    audio_output = os.path.join(output_path, 'audio.wav')
    extract_audio(temp_25fps_file, audio_output, sample_rate=16000)
    
    # Crop mouth
    if status_callback:
        status_callback({'status': 'Cropping mouth regions', 'progress': 0.55})
    
    crop_mouth_with_status(
        video_direc=os.path.join(output_path, "faces"), 
        landmark_direc=os.path.join(output_path, "landmark"), 
        filename_path=filename_path, 
        save_direc=os.path.join(output_path, "mouthroi"), 
        convert_gray=True, 
        testset_only=False,
        status_callback=status_callback
    )
    
    # Load model
    if status_callback:
        status_callback({'status': 'Loading Dolphin model', 'progress': 0.6})
    torch.cuda.empty_cache()
    audiomodel = Dolphin.from_pretrained("JusperLee/Dolphin")
    # audiomodel.cuda()
    audiomodel.eval()
    
    # Process each speaker
    with torch.no_grad():
        for i in range(number_of_speakers):
            if status_callback:
                status_callback({'status': f'Processing audio for speaker {i+1}', 'progress': 0.65 + 0.25 * (i / number_of_speakers)})
            
            mouth_roi_path = os.path.join(output_path, "mouthroi", f"speaker{i+1}.npz")
            mouth_roi = np.load(mouth_roi_path)["data"]
            mouth_roi = get_preprocessing_pipelines()["val"](mouth_roi)
            
            mix, sr = torchaudio.load(audio_output)
            mix = mix.mean(dim=0)
            
            window_size = 4 * sr 
            hop_size = int(4 * sr)
            
            all_estimates = []
            
            # Sliding window processing
            start_idx = 0
            window_count = 0
            while start_idx < len(mix):
                end_idx = min(start_idx + window_size, len(mix))
                window_mix = mix[start_idx:end_idx]
                
                start_frame = int(start_idx / sr * 25)
                end_frame = int(end_idx / sr * 25)
                end_frame = min(end_frame, len(mouth_roi))
                window_mouth_roi = mouth_roi[start_frame:end_frame]
                
                est_sources = audiomodel(window_mix[None], 
                                    torch.from_numpy(window_mouth_roi[None, None]).float())
                
                all_estimates.append({
                    'start': start_idx,
                    'end': end_idx,
                    'estimate': est_sources[0].cpu()
                })
                
                window_count += 1
                if status_callback:
                    progress = 0.65 + 0.25 * (i / number_of_speakers) + 0.25 / number_of_speakers * (window_count * hop_size / len(mix))
                    status_callback({'status': f'Processing audio window {window_count} for speaker {i+1}', 'progress': min(progress, 0.9)})
                
                start_idx += hop_size
                
                if start_idx >= len(mix):
                    break
                torch.cuda.empty_cache()
            
            output_length = len(mix)
            merged_output = torch.zeros(1, output_length)
            weights = torch.zeros(output_length)
            
            for est in all_estimates:
                window_len = est['end'] - est['start']
                hann_window = torch.hann_window(window_len)
                
                merged_output[0, est['start']:est['end']] += est['estimate'][0, :window_len] * hann_window
                weights[est['start']:est['end']] += hann_window
            
            merged_output[:, weights > 0] /= weights[weights > 0]
            
            audio_save_path = os.path.join(output_path, f"speaker{i+1}_est.wav")
            torchaudio.save(audio_save_path, merged_output, sr)

    # Merge video with separated audio for each speaker
    torch.cuda.empty_cache()
    if status_callback:
        status_callback({'status': 'Merging videos with separated audio', 'progress': 0.9})
    
    output_files = []
    for i in range(number_of_speakers):
        video_input = os.path.join(output_path, f"video_tracked{i+1}.mp4")
        audio_input = os.path.join(output_path, f"speaker{i+1}_est.wav")
        video_output = os.path.join(output_path, f"s{i+1}.mp4")
        
        merge_video_audio(video_input, audio_input, video_output)
        output_files.append(video_output)
    
    # Clean up temporary file
    if os.path.exists(temp_25fps_file):
        os.remove(temp_25fps_file)
    
    if status_callback:
        status_callback({'status': 'Processing completed!', 'progress': 1.0})
    
    return output_files
