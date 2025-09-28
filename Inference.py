import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import face_alignment
import torch
import torchaudio
import numpy as np
import cv2
from PIL import Image, ImageDraw
from moviepy import *
from collections import deque                                                 
from skimage import transform as tf
import yaml

from look2hear.models import Dolphin
from look2hear.datas.transform import get_preprocessing_pipelines

from face_detection_utils import detect_faces

# -- Landmark interpolation:
def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx-start_idx):
        landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
    return landmarks

# -- Face Transformation
def warp_img(src, dst, img, std_size):
    tform = tf.estimate_transform('similarity', src, dst)  # find the transformation matrix
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)  # wrap the frame image
    warped = warped * 255  # note output from wrap is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped, tform

def apply_transform(transform, img, std_size):
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    warped = warped * 255  # note output from wrap is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped

# -- Crop
def cut_patch(img, landmarks, height, width, threshold=5):

    center_x, center_y = np.mean(landmarks, axis=0)

    if center_y - height < 0:                                                
        center_y = height                                                    
    if center_y - height < 0 - threshold:                                    
        raise Exception('too much bias in height')                           
    if center_x - width < 0:                                                 
        center_x = width                                                     
    if center_x - width < 0 - threshold:                                     
        raise Exception('too much bias in width')                            
                                                                             
    if center_y + height > img.shape[0]:                                     
        center_y = img.shape[0] - height                                     
    if center_y + height > img.shape[0] + threshold:                         
        raise Exception('too much bias in height')                           
    if center_x + width > img.shape[1]:                                      
        center_x = img.shape[1] - width                                      
    if center_x + width > img.shape[1] + threshold:                          
        raise Exception('too much bias in width')                            
                                                                             
    cutted_img = np.copy(img[ int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                         int(round(center_x) - round(width)): int(round(center_x) + round(width))])
    return cutted_img

# -- RGB to GRAY
def convert_bgr2gray(data):
    return np.stack([cv2.cvtColor(_, cv2.COLOR_BGR2GRAY) for _ in data], axis=0)


def save2npz(filename, data=None):
    assert data is not None, "data is {}".format(data)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    np.savez_compressed(filename, data=data)
    
def read_video(filename):
    """Read video frames using MoviePy for better compatibility"""
    try:
        video_clip = VideoFileClip(filename)
        for frame in video_clip.iter_frames():
            # Convert RGB to BGR to match cv2 format
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            yield frame_bgr
        video_clip.close()
    except Exception as e:
        print(f"Error reading video {filename}: {e}")
        return

def face2head(boxes, scale=1.5):
    new_boxes = []
    for box in boxes:
        width = box[2] - box[0]
        height= box[3] - box[1]
        width_center = (box[2] + box[0]) / 2
        height_center = (box[3] + box[1]) / 2
        square_width = int(max(width, height) * scale)
        new_box = [width_center - square_width/2, height_center - square_width/2, width_center + square_width/2, height_center + square_width/2]
        new_boxes.append(new_box)
    return new_boxes

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def detectface(video_input_path, output_path, detect_every_N_frame, scalar_face_detection, number_of_speakers):
    device = torch.device('cuda' if torch.cuda.get_device_name() else 'cpu')
    print('Running on device: {}'.format(device))
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
    print("Video statistics: ", video_clip.w, video_clip.h, (video_clip.w, video_clip.h), video_clip.fps)
    frames = [Image.fromarray(frame) for frame in video_clip.iter_frames()]
    print('Number of frames in video: ', len(frames))
    video_clip.close()
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    
    for i, frame in enumerate(frames):
        print('\rTracking frame: {}'.format(i + 1), end='')
        
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
                detected_boxes = detected_boxes[:number_of_speakers]
                detected_boxes = face2head(detected_boxes, scalar_face_detection)
            else:
                detected_boxes = []
        
        # Process the detection results
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
    print(f"\nFrame counts per speaker: {frame_counts}")
    assert all(count == len(frames) for count in frame_counts), f"Inconsistent frame counts: {frame_counts}"
    
    # Continue with saving videos and landmarks...
    for s in range(number_of_speakers):
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
        save2npz(os.path.join(output_path, 'landmark', 'speaker' + str(i+1)+'.npz'), data=landmarks_dic[i])
        
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
    if not os.path.exists(os.path.join(output_path, 'filename_input')):
        os.mkdir(os.path.join(output_path, 'filename_input'))
    csvfile = open(os.path.join(output_path, 'filename_input', str(video_name) + '.csv'), 'w')
    for i in range(number_of_speakers):
        csvfile.write('speaker' + str(i+1)+ ',0\n')
    csvfile.close()
    return os.path.join(output_path, 'filename_input', str(video_name) + '.csv')


def crop_patch(mean_face_landmarks, video_pathname, landmarks, window_margin, start_idx, stop_idx, crop_height, crop_width, STD_SIZE=(256, 256)):

    """Crop mouth patch
    :param str video_pathname: pathname for the video_dieo
    :param list landmarks: interpolated landmarks
    """
    
    stablePntsIDs = [33, 36, 39, 42, 45]

    frame_idx = 0
    frame_gen = read_video(video_pathname)
    while True:
        try:
            frame = frame_gen.__next__() ## -- BGR
        except StopIteration:
            break
        if frame_idx == 0:
            q_frame, q_landmarks = deque(), deque()
            sequence = []

        q_landmarks.append(landmarks[frame_idx])
        q_frame.append(frame)
        if len(q_frame) == window_margin:
            smoothed_landmarks = np.mean(q_landmarks, axis=0)
            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frame.popleft()
            # -- affine transformation
            trans_frame, trans = warp_img( smoothed_landmarks[stablePntsIDs, :],
                                           mean_face_landmarks[stablePntsIDs, :],
                                           cur_frame,
                                           STD_SIZE)
            trans_landmarks = trans(cur_landmarks)
            # -- crop mouth patch
            sequence.append( cut_patch( trans_frame,
                                        trans_landmarks[start_idx:stop_idx],
                                        crop_height//2,
                                        crop_width//2,))
        if frame_idx == len(landmarks)-1:
            #deal with corner case with video too short
            if len(landmarks) < window_margin:
                smoothed_landmarks = np.mean(q_landmarks, axis=0)
                cur_landmarks = q_landmarks.popleft()
                cur_frame = q_frame.popleft()

                # -- affine transformation
                trans_frame, trans = warp_img(smoothed_landmarks[stablePntsIDs, :],
                                            mean_face_landmarks[stablePntsIDs, :],
                                            cur_frame,
                                            STD_SIZE)
                trans_landmarks = trans(cur_landmarks)
                # -- crop mouth patch
                sequence.append(cut_patch( trans_frame,
                                trans_landmarks[start_idx:stop_idx],
                                crop_height//2,
                                crop_width//2,))

            while q_frame:
                cur_frame = q_frame.popleft()
                # -- transform frame
                trans_frame = apply_transform( trans, cur_frame, STD_SIZE)
                # -- transform landmarks
                trans_landmarks = trans(q_landmarks.popleft())
                # -- crop mouth patch
                sequence.append( cut_patch( trans_frame,
                                            trans_landmarks[start_idx:stop_idx],
                                            crop_height//2,
                                            crop_width//2,))
            return np.array(sequence)
        frame_idx += 1
    return None

def landmarks_interpolate(landmarks):
    
    """Interpolate landmarks
    param list landmarks: landmarks detected in raw videos
    """

    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx-1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks

def crop_mouth(video_direc, landmark_direc, filename_path, save_direc, convert_gray=False, testset_only=False):
    lines = open(filename_path).read().splitlines()
    lines = list(filter(lambda x: 'test' in x, lines)) if testset_only else lines

    for filename_idx, line in enumerate(lines):

        filename, person_id = line.split(',')
        print('idx: {} \tProcessing.\t{}'.format(filename_idx, filename))

        video_pathname = os.path.join(video_direc, filename+'.mp4')
        landmarks_pathname = os.path.join(landmark_direc, filename+'.npz')
        dst_pathname = os.path.join( save_direc, filename+'.npz')

        # if os.path.exists(dst_pathname):
        #    continue

        multi_sub_landmarks = np.load(landmarks_pathname, allow_pickle=True)['data']
        landmarks = [None] * len(multi_sub_landmarks)
        for frame_idx in range(len(landmarks)):
            try:
                #landmarks[frame_idx] = multi_sub_landmarks[frame_idx][int(person_id)]['facial_landmarks'] #original for LRW
                landmarks[frame_idx] = multi_sub_landmarks[frame_idx][int(person_id)] #VOXCELEB2
            except (IndexError, TypeError):
                continue

        # -- pre-process landmarks: interpolate frames not being detected.
        preprocessed_landmarks = landmarks_interpolate(landmarks)
        if not preprocessed_landmarks:
            continue

        # -- crop
        mean_face_landmarks = np.load('assets/20words_mean_face.npy')
        sequence = crop_patch(mean_face_landmarks, video_pathname, preprocessed_landmarks, 12, 48, 68, 96, 96)
        assert sequence is not None, "cannot crop from {}.".format(filename)

        # -- save
        data = convert_bgr2gray(sequence) if convert_gray else sequence[...,::-1]
        save2npz(dst_pathname, data=data)

def convert_video_fps(input_file, output_file, target_fps=25):
    """Convert video to target FPS using moviepy"""
    video = VideoFileClip(input_file)
    video_fps = video.fps
    
    if video_fps != target_fps:
        video.write_videofile(
            output_file,
            fps=target_fps,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
        )
    else:
        # If already at target fps, just copy
        import shutil
        shutil.copy2(input_file, output_file)
    
    video.close()
    print(f'Video has been converted to {target_fps} fps and saved to {output_file}')

def extract_audio(video_file, audio_output_file, sample_rate=16000):
    """Extract audio from video using moviepy"""
    video = VideoFileClip(video_file)
    audio = video.audio
    
    # Save audio with specified sample rate
    audio.write_audiofile(audio_output_file, fps=sample_rate, nbytes=2, codec='pcm_s16le')
    
    video.close()
    audio.close()

def merge_video_audio(video_file, audio_file, output_file):
    """Merge video and audio using moviepy"""
    video = VideoFileClip(video_file)
    audio = AudioFileClip(audio_file)
    
    # Attach audio (MoviePy v2 renamed set_audio to with_audio)
    set_audio_fn = getattr(video, "set_audio", None)
    if callable(set_audio_fn):
        final_video = set_audio_fn(audio)
    else:
        with_audio_fn = getattr(video, "with_audio", None)
        if not callable(with_audio_fn):
            video.close()
            audio.close()
            raise AttributeError("VideoFileClip object lacks both set_audio and with_audio methods")
        final_video = with_audio_fn(audio)
    
    # Write the result
    final_video.write_videofile(output_file, codec='libx264', audio_codec='aac', temp_audiofile='temp-audio.m4a', remove_temp=True)
    
    # Clean up
    video.close()
    audio.close()
    final_video.close()

def process_video(input_file, output_path, number_of_speakers=2, 
                  detect_every_N_frame=8, scalar_face_detection=1.5,
                  config_path="checkpoints/vox2/conf.yml",
                  cuda_device=None):
    """Main processing function for video speaker separation"""
    
    # Set CUDA device if specified
    if cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Convert video to 25fps
    temp_25fps_file = os.path.join(output_path, 'temp_25fps.mp4')
    convert_video_fps(input_file, temp_25fps_file, target_fps=25)
    
    # Detect faces
    filename_path = detectface(video_input_path=temp_25fps_file, 
                              output_path=output_path, 
                              detect_every_N_frame=detect_every_N_frame, 
                              scalar_face_detection=scalar_face_detection, 
                              number_of_speakers=number_of_speakers)
    
    # Extract audio
    audio_output = os.path.join(output_path, 'audio.wav')
    extract_audio(temp_25fps_file, audio_output, sample_rate=16000)
    
    # Crop mouth
    crop_mouth(video_direc=os.path.join(output_path, "faces"), 
               landmark_direc=os.path.join(output_path, "landmark"), 
               filename_path=filename_path, 
               save_direc=os.path.join(output_path, "mouthroi"), 
               convert_gray=True, 
               testset_only=False)
    
    # Load model
    audiomodel = Dolphin.from_pretrained("JusperLee/Dolphin")
    
    audiomodel.cuda()
    audiomodel.eval()
    
    # Process each speaker
    with torch.no_grad():
        for i in range(number_of_speakers):
            mouth_roi = np.load(os.path.join(output_path, "mouthroi", f"speaker{i+1}.npz"))["data"]
            mouth_roi = get_preprocessing_pipelines()["val"](mouth_roi)
            
            mix, sr = torchaudio.load(audio_output)
            mix = mix.cuda().mean(dim=0)
            
            window_size = 4 * sr 
            hop_size = 4 * sr 
            
            all_estimates = []
            
            # 滑动窗口处理
            start_idx = 0
            while start_idx < len(mix):
                end_idx = min(start_idx + window_size, len(mix))
                window_mix = mix[start_idx:end_idx]
                
                start_frame = int(start_idx / sr * 25)
                end_frame = int(end_idx / sr * 25)
                end_frame = min(end_frame, len(mouth_roi))
                window_mouth_roi = mouth_roi[start_frame:end_frame]
                
                est_sources = audiomodel(window_mix[None], 
                                    torch.from_numpy(window_mouth_roi[None, None]).float().cuda())
                
                all_estimates.append({
                    'start': start_idx,
                    'end': end_idx,
                    'estimate': est_sources[0].cpu()
                })
                
                start_idx += hop_size
                
                if start_idx >= len(mix):
                    break
            
            output_length = len(mix)
            merged_output = torch.zeros(1, output_length)
            weights = torch.zeros(output_length)
            
            for est in all_estimates:
                window_len = est['end'] - est['start']
                hann_window = torch.hann_window(window_len)
                
                merged_output[0, est['start']:est['end']] += est['estimate'][0, :window_len] * hann_window
                weights[est['start']:est['end']] += hann_window
            
            merged_output[:, weights > 0] /= weights[weights > 0]
            
            torchaudio.save(os.path.join(output_path, f"speaker{i+1}_est.wav"), merged_output, sr)

    
    # Merge video with separated audio for each speaker
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
    
    return output_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Speaker Separation using Dolphin model')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory path (default: creates directory based on input filename)')
    parser.add_argument('--speakers', '-s', type=int, default=2,
                        help='Number of speakers to separate (default: 2)')
    parser.add_argument('--detect-every-n', type=int, default=8,
                        help='Detect faces every N frames (default: 8)')
    parser.add_argument('--face-scale', type=float, default=1.5,
                        help='Face detection bounding box scale factor (default: 1.5)')
    parser.add_argument('--cuda-device', type=int, default=0,
                        help='CUDA device ID to use (default: 0, set to -1 for CPU)')
    parser.add_argument('--config', type=str, default="checkpoints/vox2/conf.yml",
                        help='Path to model configuration file')
    
    args = parser.parse_args()
    
    # 验证输入文件是否存在
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        exit(1)
    
    # 如果没有指定输出路径，基于输入文件名创建输出目录
    if args.output is None:
        input_basename = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(os.path.dirname(args.input), input_basename + "_output")
    
    # 设置CUDA设备
    cuda_device = args.cuda_device if args.cuda_device >= 0 else None
    
    print(f"Processing video: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Number of speakers: {args.speakers}")
    print(f"CUDA device: {cuda_device if cuda_device is not None else 'CPU'}")
    
    # 处理视频
    output_files = process_video(
        input_file=args.input,
        output_path=args.output,
        number_of_speakers=args.speakers,
        detect_every_N_frame=args.detect_every_n,
        scalar_face_detection=args.face_scale,
        config_path=args.config,
        cuda_device=cuda_device
    )
    
    print("\nProcessing completed!")
    print("Output files:")
    for i, output_file in enumerate(output_files):
        print(f"  Speaker {i+1}: {output_file}")
