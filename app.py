#!/usr/bin/env python3
"""
Audio-visual Speech Separation Gradio App - Hugging Face Space Version
Automatically detects and separates all speakers in videos
"""

import warnings
warnings.filterwarnings("ignore")
import os
import gradio as gr
import numpy as np
import shutil
import tempfile
import time
import sys
import threading
from PIL import Image, ImageDraw, ImageFont
from moviepy import *
import spaces

from face_detection_utils import detect_faces

# Use HF Space's temp directory
TEMP_DIR = os.environ.get('TMPDIR', '/tmp')

# Shared state for relaying GPU-side status back to the UI thread.
GPU_PROGRESS_STATE = {"progress": 0.0, "status": "Processing on GPU..."}
GPU_PROGRESS_LOCK = threading.Lock()

class LogCollector:
    """Collect logs in a list"""
    def __init__(self):
        self.logs = []
        
    def add(self, message):
        if message and message.strip():
            timestamp = time.strftime("%H:%M:%S")
            self.logs.append(f"[{timestamp}] {message.strip()}")
    
    def get_text(self, last_n=None):
        if last_n:
            return "\n".join(self.logs[-last_n:])
        return "\n".join(self.logs)

# Global log collector for capturing print statements
GLOBAL_LOG = LogCollector()

class StdoutCapture:
    """Capture stdout and add to log"""
    def __init__(self, original):
        self.original = original
        
    def write(self, text):
        self.original.write(text)
        if text.strip():
            GLOBAL_LOG.add(text.strip())
    
    def flush(self):
        self.original.flush()

def remove_duplicate_faces(boxes, probs, iou_threshold=0.5):
    """Remove duplicate face detections using IoU (Intersection over Union)"""
    if len(boxes) <= 1:
        return boxes, probs
    
    # Calculate IoU between all pairs of boxes
    def calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    # Keep track of which boxes to keep
    keep = []
    used = set()
    
    # Sort by confidence (if available) or by area
    if probs is not None:
        sorted_indices = np.argsort(probs)[::-1]
    else:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_indices = np.argsort(areas)[::-1]
    
    for i in sorted_indices:
        if i in used:
            continue
        
        keep.append(i)
        used.add(i)
        
        # Mark overlapping boxes as used
        for j in range(len(boxes)):
            if j != i and j not in used:
                iou = calculate_iou(boxes[i], boxes[j])
                if iou > iou_threshold:
                    used.add(j)
    
    # Return filtered boxes and probs
    keep = sorted(keep)  # Maintain original order
    filtered_boxes = boxes[keep]
    filtered_probs = probs[keep] if probs is not None else None
    
    return filtered_boxes, filtered_probs

def process_detected_faces(boxes, probs, frame_rgb, frame_pil):
    """Process detected faces and return face images"""
    face_images = []
    full_frame_annotated = frame_rgb.copy()

    if boxes is None or len(boxes) == 0:
        return [], 0, full_frame_annotated, "No faces detected"

    boxes = np.asarray(boxes, dtype=np.float32)

    # Filter by confidence if available
    if probs is not None:
        # Keep faces with confidence > 0.9
        confident_indices = probs > 0.9
        boxes = boxes[confident_indices]
        probs = probs[confident_indices]
        print(f"After filtering by confidence: {len(boxes)} faces")

    if len(boxes) == 0:
        return [], 0, full_frame_annotated, "No faces passed the confidence filter"

    # Remove duplicate detections
    boxes, probs = remove_duplicate_faces(boxes, probs, iou_threshold=0.3)
    print(f"After removing duplicates: {len(boxes)} faces")

    if len(boxes) == 0:
        return [], 0, full_frame_annotated, "No faces remained after duplicate removal"

    # Sort boxes by area (larger faces first)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_indices = np.argsort(areas)[::-1]
    boxes = boxes[sorted_indices]
    
    # Annotate full frame
    full_frame_pil = Image.fromarray(full_frame_annotated)
    draw = ImageDraw.Draw(full_frame_pil)
    
    # Try to use a better font
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    # Extract face images and annotate
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    for i, box in enumerate(boxes):
        color = colors[i % len(colors)]
        
        # Draw bounding box
        draw.rectangle(box.tolist(), outline=color, width=4)
        label = f"Speaker {i+1}"
        
        # Draw label
        if font:
            draw.text((box[0] + 5, box[1] - 20), label, fill=color, font=font)
        
        # Extract face with margin
        margin = 50
        x1 = max(0, int(box[0] - margin))
        y1 = max(0, int(box[1] - margin))
        x2 = min(frame_rgb.shape[1], int(box[2] + margin))
        y2 = min(frame_rgb.shape[0], int(box[3] + margin))
        
        face_crop = frame_rgb[y1:y2, x1:x2]
        # Resize maintaining aspect ratio
        face_crop = Image.fromarray(face_crop)
        face_crop.thumbnail((250, 250), Image.Resampling.LANCZOS)
        face_crop = np.array(face_crop)
        
        face_images.append(face_crop)
    
    full_frame_annotated = np.array(full_frame_pil)
    return face_images, len(boxes), full_frame_annotated, None

@spaces.GPU(duration=60, enable_queue=True)
def detect_faces_gpu(frame_pil):
    """GPU-accelerated face detection"""
    print("Detecting faces with RetinaFace")

    frame_array = np.array(frame_pil)

    boxes, probs = detect_faces(
        frame_array,
        threshold=0.9,
        allow_upscaling=False,
    )

    if boxes is None or len(boxes) == 0:
        print("No faces detected at high threshold, relaxing criteria...")
        boxes, probs = detect_faces(
            frame_array,
            threshold=0.7,
            allow_upscaling=True,
        )

    return boxes, probs

def detect_and_extract_all_faces(video_path):
    """Detect all faces in the first frame and extract them"""
    print("Starting face detection...")
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file does not exist at path: {video_path}")
        return [], 0, None, f"Video file not found: {video_path}"
    
    print(f"Video path: {video_path}")
    print(f"File size: {os.path.getsize(video_path) / 1024 / 1024:.2f} MB")
    
    # Use moviepy to read video
    print("Opening video with moviepy...")
    try:
        clip = VideoFileClip(video_path)
        
        # Get video properties
        fps = clip.fps
        duration = clip.duration
        total_frames = int(fps * duration)
        
        print(f"Video info: FPS: {fps}, Duration: {duration}s, Total frames: {total_frames}")
        
        # Get first frame
        frame = clip.get_frame(0)  # MoviePy returns RGB
        frame_rgb = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        
        print(f"Successfully read frame with moviepy: {frame_rgb.shape}")
        
        # Close the clip to free resources
        clip.close()
        
        # Convert to PIL for downstream processing
        frame_pil = Image.fromarray(frame_rgb)

        # Detect faces using RetinaFace
        print("Detecting faces with RetinaFace...")
        boxes, probs = detect_faces(
            frame_rgb,
            threshold=0.9,
            allow_upscaling=False,
        )

        if boxes is None or len(boxes) == 0:
            print("No faces detected at high threshold, trying relaxed settings...")
            boxes, probs = detect_faces(
                frame_rgb,
                threshold=0.7,
                allow_upscaling=True,
            )

        if boxes is not None and len(boxes) > 0:
            print(f"Detected {len(boxes)} faces")
            return process_detected_faces(boxes, probs, frame_rgb, frame_pil)
        else:
            return [], 0, frame_rgb, "No faces detected in the first frame"
                
    except Exception as e:
        print(f"MoviePy failed: {e}")
        import traceback
        traceback.print_exc()
        return [], 0, None, f"Failed to open video file. Error: {str(e)}"

@spaces.GPU(duration=300, enable_queue=True)
def process_video_gpu(video_file, temp_dir, num_speakers):
    """GPU-accelerated video processing"""
    try:
        from Inference_with_status import process_video_with_status
        
        # Define status callback inside GPU function
        def gpu_status_callback(message):
            status_text = message.get('status', 'Processing...')
            print(f"GPU Processing: {status_text}")
            progress_value = message.get('progress')
            with GPU_PROGRESS_LOCK:
                GPU_PROGRESS_STATE["status"] = status_text
                if progress_value is not None:
                    try:
                        numeric_progress = float(progress_value)
                        GPU_PROGRESS_STATE["progress"] = min(max(numeric_progress, 0.0), 1.0)
                    except (TypeError, ValueError):
                        pass
        
        output_files = process_video_with_status(
            input_file=video_file,
            output_path=temp_dir,
            number_of_speakers=num_speakers,
            detect_every_N_frame=8,
            scalar_face_detection=1.5,
            status_callback=gpu_status_callback
        )
        return output_files
    except ImportError:
        from Inference import process_video
        print("Using standard process_video (status callbacks not available)")
        output_files = process_video(
            input_file=video_file,
            output_path=temp_dir,
            number_of_speakers=num_speakers,
            detect_every_N_frame=8,
            scalar_face_detection=1.5
        )
        return output_files

def process_video_auto(video_file, progress=gr.Progress()):
    """Process video with automatic speaker detection and stream status updates"""
    global GLOBAL_LOG
    GLOBAL_LOG = LogCollector()

    old_stdout = sys.stdout
    sys.stdout = StdoutCapture(old_stdout)

    status_value = "⏳ Ready to process..."
    detected_info_output = gr.update(visible=False)
    face_gallery_output = gr.update(visible=False)
    output_video_output = gr.update(visible=False)
    video_dict_value = None
    annotated_frame_output = gr.update(visible=False)

    def snapshot():
        return (
            status_value,
            detected_info_output,
            face_gallery_output,
            output_video_output,
            video_dict_value,
            annotated_frame_output,
            GLOBAL_LOG.get_text()
        )

    try:
        if video_file is None:
            status_value = "⚠️ Please upload a video file"
            yield snapshot()
            return

        progress(0, desc="Starting processing...")
        status_value = "🔄 Starting processing..."
        GLOBAL_LOG.add("Starting video processing...")
        yield snapshot()

        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp(dir=TEMP_DIR)
            print(f"Created temporary directory: {temp_dir}")

            progress(0.1, desc="Detecting speakers in video...")
            status_value = "🔍 Detecting speakers in video..."
            print("Starting face detection in video...")
            yield snapshot()

            face_images, num_speakers, annotated_frame, error_msg = detect_and_extract_all_faces(video_file)
            print(f"Face detection completed. Found {num_speakers} speakers.")

            if error_msg:
                print(f"Error: {error_msg}")
                status_value = f"❌ {error_msg}"
                if annotated_frame is not None:
                    annotated_frame_output = gr.update(value=annotated_frame, visible=True)
                yield snapshot()
                return

            if num_speakers == 0:
                print("No speakers detected in the video.")
                status_value = "❌ No speakers detected in the video. Please ensure faces are visible in the first frame."
                if annotated_frame is not None:
                    annotated_frame_output = gr.update(value=annotated_frame, visible=True)
                yield snapshot()
                return

            face_gallery_images = [(img, f"Speaker {i+1}") for i, img in enumerate(face_images)]
            detected_info = f"🎯 Detected {num_speakers} speaker{'s' if num_speakers > 1 else ''} in the video"
            detected_info_output = gr.update(value=detected_info, visible=True)
            face_gallery_output = gr.update(value=face_gallery_images, visible=True)
            if annotated_frame is not None:
                annotated_frame_output = gr.update(value=annotated_frame, visible=True)

            progress(0.3, desc=f"Separating {num_speakers} speakers...")
            status_value = f"🎬 Separating {num_speakers} speakers..."
            print(f"Starting audio-visual separation for {num_speakers} speakers...")
            yield snapshot()

            try:
                print("Starting GPU-accelerated video processing...")
                with GPU_PROGRESS_LOCK:
                    GPU_PROGRESS_STATE["progress"] = 0.0
                    GPU_PROGRESS_STATE["status"] = "Processing on GPU..."

                progress(0.4, desc="Processing on GPU...")
                status_value = "Processing on GPU..."
                yield snapshot()

                gpu_result = {"output_files": None, "exception": None}

                def run_gpu_processing():
                    try:
                        gpu_result["output_files"] = process_video_gpu(
                            video_file=video_file,
                            temp_dir=temp_dir,
                            num_speakers=num_speakers
                        )
                    except Exception as exc:
                        gpu_result["exception"] = exc

                gpu_thread = threading.Thread(target=run_gpu_processing, daemon=True)
                gpu_thread.start()

                last_reported_progress = 0.4
                last_status_message = "Processing on GPU..."

                while gpu_thread.is_alive():
                    time.sleep(0.5)
                    with GPU_PROGRESS_LOCK:
                        gpu_status = GPU_PROGRESS_STATE.get("status", "Processing on GPU...")
                        gpu_progress_value = GPU_PROGRESS_STATE.get("progress", 0.0)

                    mapped_progress = 0.4 + 0.5 * gpu_progress_value
                    mapped_progress = min(mapped_progress, 0.89)

                    if (
                        mapped_progress > last_reported_progress + 0.01
                        or gpu_status != last_status_message
                    ):
                        progress(mapped_progress, desc=gpu_status)
                        last_reported_progress = mapped_progress
                        last_status_message = gpu_status
                        status_value = gpu_status
                        yield snapshot()

                gpu_thread.join()

                if gpu_result["exception"] is not None:
                    raise gpu_result["exception"]

                output_files = gpu_result["output_files"]

                progress(0.9, desc="Preparing results...")
                status_value = "📦 Preparing results..."
                print("Processing completed successfully!")
                print(f"Generated {num_speakers} output videos")
                yield snapshot()

                video_dict_value = {i: output_files[i] for i in range(num_speakers)}
                video_dict_value['temp_dir'] = temp_dir
                output_video_output = gr.update(value=output_files[0], visible=True)

                progress(1.0, desc="Complete!")
                status_value = f"✅ Successfully separated {num_speakers} speakers! Click on any face below to view their video."
                yield snapshot()

            except Exception as e:
                print(f"Processing failed: {str(e)}")
                import traceback
                traceback.print_exc()
                status_value = f"❌ Processing failed: {str(e)}"
                output_video_output = gr.update(visible=False)
                video_dict_value = None
                yield snapshot()
                return

        except Exception as e:
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass

            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            status_value = f"❌ Error: {str(e)}"
            detected_info_output = gr.update(visible=False)
            face_gallery_output = gr.update(visible=False)
            output_video_output = gr.update(visible=False)
            annotated_frame_output = gr.update(visible=False)
            video_dict_value = None
            yield snapshot()
            return
    finally:
        sys.stdout = old_stdout

def on_face_click(evt: gr.SelectData, video_dict):
    """Handle face gallery click events"""
    if video_dict is None or evt.index not in video_dict:
        return None
    
    return video_dict[evt.index]

# Create the Gradio interface
custom_css = """
.face-gallery {
    border-radius: 10px;
    overflow: hidden;
}
.face-gallery img {
    border-radius: 8px;
    transition: transform 0.2s ease-in-out;
}
.face-gallery img:hover {
    transform: scale(1.05);
    cursor: pointer;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
}
.detected-info {
    background-color: #f0f0f0;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}
"""

with gr.Blocks(
    title="Video Speaker Auto-Separation",
    theme=gr.themes.Soft(),
    css=custom_css
) as demo:
    gr.Markdown(
        """
        # 🎥 Dolphin: Efficient Audio-Visual Speech Separation with Discrete Lip Semantics and Hierarchical Top-Down Attention
        <p align="left">
        <img src="https://visitor-badge.laobi.icu/badge?page_id=JusperLee.Dolphin" alt="访客统计" /><img src="https://img.shields.io/github/stars/JusperLee/Dolphin?style=social" alt="GitHub stars" /><img alt="Static Badge" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" />
        </p>
        
        ### Automatically detect and separate ALL speakers in your video
        
        Simply upload a video and the system will:
        1. 🔍 Automatically detect all speakers in the video
        2. 🎭 Show you each detected speaker's face
        3. 🎬 Generate individual videos for each speaker with their isolated audio
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            video_input = gr.Video(
                label="📹 Upload Your Video",
                height=300,
                interactive=True
            )
            
            # Add example video section
            gr.Markdown("### 🎬 Try with Example Video")
            gr.Examples(
                examples=[["demo1/mix.mp4"]],
                inputs=video_input,
                label="Click to load example video",
                cache_examples=False
            )
            
            process_btn = gr.Button(
                "🚀 Auto-Detect and Process",
                variant="primary",
                size="lg"
            )
            
            status = gr.Textbox(
                label="Status",
                interactive=False,
                value="⏳ Ready to process..."
            )
            
            processing_log = gr.Textbox(
                label="📋 Processing Details",
                lines=10,
                max_lines=15,
                interactive=False,
                value=""
            )
        
        with gr.Column(scale=3):
            annotated_frame = gr.Image(
                label="📸 Detected Speakers in First Frame",
                visible=False,
                height=300
            )
            
            detected_info = gr.Markdown(
                visible=False,
                elem_classes="detected-info"
            )
            
            gr.Markdown("### 👇 Click on any face below to view that speaker's video")
            
            face_gallery = gr.Gallery(
                label="Detected Speaker Faces",
                show_label=False,
                columns=5,
                rows=1,
                height=200,
                visible=False,
                object_fit="contain",
                elem_classes="face-gallery"
            )
            
            output_video = gr.Video(
                label="🎬 Selected Speaker's Video",
                height=300,
                visible=False,
                autoplay=True
            )
    
    # Hidden state
    video_dict = gr.State()
    
    gr.Markdown(
        """
        ---
        ### 📖 How it works:
        
        1. **Upload** - Select any video file
        2. **Process** - Click the button to start automatic detection
        3. **Review** - See all detected speakers and their positions
        4. **Select** - Click on any face to watch that speaker's separated video
        
        ### 💡 Tips for best results:
        
        - ✅ Ensure all speakers' faces are visible in the first frame
        - ✅ Use videos with good lighting and clear face views
        - ✅ Works best with frontal or near-frontal face angles
        - ⏱️ Processing time depends on video length and number of speakers
        
        ### 🚀 Powered by:
        - RetinaFace for face detection
        - Dolphin model for audio-visual separation
        - GPU acceleration when available
        <footer style="display:none;">
            <a href='https://clustrmaps.com/site/1c828' title='Visit tracker'>
                <img src='//clustrmaps.com/map_v2.png?cl=080808&w=300&t=tt&d=XYmTC4S_SxuX7G06iJ16lU43VCNkCBFRLXMfEM5zvmo&co=ffffff&ct=808080'/>
            </a>
        </footer>
        """
    )
    
    # Event handlers
    outputs_list = [
        status,
        detected_info,
        face_gallery,
        output_video,
        video_dict,
        annotated_frame,
        processing_log
    ]
    
    process_btn.click(
        fn=process_video_auto,
        inputs=[video_input],
        outputs=outputs_list,
        show_progress=True
    )
    
    face_gallery.select(
        fn=on_face_click,
        inputs=[video_dict],
        outputs=output_video
    )

# Launch the demo - HF Space will handle this automatically
if __name__ == "__main__":
    import os
    demo.launch(server_name="0.0.0.0", server_port=7860)
