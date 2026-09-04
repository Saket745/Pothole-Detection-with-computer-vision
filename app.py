import os
import time
import threading
from pathlib import Path
from flask import Flask, render_template, Response, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import torch
import supervision as sv

# ==============================================================================
# YOLOv12 Backward-Compatibility Patch for Area Attention (AAttn)
# ==============================================================================
# In Ultralytics 8.3+, AAttn was refactored from single 'qkv' to 'qk' and 'v'.
# This monkey-patch handles checkpoints trained on either v1.0 or newer versions.
import ultralytics.nn.modules.block as ultralytics_block

def patched_aattn_forward(self, x):
    B, C, H, W = x.shape
    N = H * W
    if hasattr(self, 'qkv'):
        qkv = self.qkv(x)
        qk_feat = qkv[:, :2 * C, :, :]
        v = qkv[:, 2 * C:, :, :]
        pp = self.pe(v)
        qk = qk_feat.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)
    else:
        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x)
        pp = self.pe(v)
        v = v.flatten(2).transpose(1, 2)

    if self.area > 1:
        qk = qk.reshape(B * self.area, N // self.area, C * 2)
        v = v.reshape(B * self.area, N // self.area, C)
        B, N, _ = qk.shape
    q, k = qk.split([C, C], dim=2)

    q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
    k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
    v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)

    attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
    max_attn = attn.max(dim=-1, keepdim=True).values
    exp_attn = torch.exp(attn - max_attn)
    attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
    x = (v @ attn.transpose(-2, -1))

    x = x.permute(0, 3, 1, 2)

    if self.area > 1:
        x = x.reshape(B // self.area, N * self.area, C)
        B, N, _ = x.shape
    x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

    return self.proj(x + pp)

ultralytics_block.AAttn.forward = patched_aattn_forward

from ultralytics import YOLO

# ==============================================================================
# Configuration & State Management
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

app = Flask(__name__, template_folder="Templates")
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)
app.config['MAX_CONTENT_LENGTH'] = 250 * 1024 * 1024  # 250 MB max upload

class VideoDetectorStream:
    """Thread-safe real-time detector and stream provider."""
    
    def __init__(self, model_path="best.pt", default_source="demo.mp4"):
        print(f"[*] Loading YOLOv12 model from {model_path}...")
        self.model = YOLO(model_path)
        print("[+] Model loaded successfully!")

        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
            color=sv.Color.from_hex("#FF3366")
        )
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.6,
            text_thickness=1,
            text_color=sv.Color.WHITE,
            color=sv.Color.from_hex("#FF3366"),
            text_padding=6
        )

        self.source = default_source
        self.confidence = 0.35
        self.lock = threading.Lock()
        
        # Stats
        self.detection_count = 0
        self.fps = 0.0
        self.latency_ms = 0.0
        self.stream_id = 0

    def set_source(self, source_path):
        with self.lock:
            self.source = source_path
            self.stream_id += 1
            print(f"[*] Switched video source to: {self.source}")

    def set_confidence(self, conf):
        with self.lock:
            self.confidence = max(0.05, min(0.95, float(conf)))

    def get_stats(self):
        with self.lock:
            return {
                "detections": self.detection_count,
                "fps": round(self.fps, 1),
                "latency_ms": round(self.latency_ms, 1),
                "confidence": round(self.confidence, 2),
                "source": Path(self.source).name if isinstance(self.source, str) else f"Camera {self.source}"
            }

    def generate_frames(self):
        local_stream_id = self.stream_id
        current_source = self.source
        cap = cv2.VideoCapture(current_source)

        prev_time = time.time()
        fps_smoothing = 0.9

        while True:
            # Check if source switched
            if self.stream_id != local_stream_id:
                cap.release()
                local_stream_id = self.stream_id
                current_source = self.source
                cap = cv2.VideoCapture(current_source)

            success, frame = cap.read()
            if not success:
                # Loop video automatically
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.05)
                continue

            start_infer = time.time()

            # YOLOv12 inference with current confidence
            with self.lock:
                conf = self.confidence

            results = self.model(frame, conf=conf, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)

            # Annotate bounding boxes and labels
            annotated_frame = self.box_annotator.annotate(scene=frame, detections=detections)
            labels = [
                f"{self.model.names[cid]} {score:.2f}"
                for cid, score in zip(detections.class_id, detections.confidence)
            ]
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels
            )

            # Time & FPS calculations
            infer_time = (time.time() - start_infer) * 1000.0
            now = time.time()
            curr_fps = 1.0 / max(1e-5, (now - prev_time))
            prev_time = now

            # Update shared statistics
            with self.lock:
                self.detection_count = len(detections)
                self.latency_ms = infer_time
                self.fps = (self.fps * fps_smoothing) + (curr_fps * (1.0 - fps_smoothing))

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

# Global detector instance
detector = VideoDetectorStream(
    model_path=str(BASE_DIR / "best.pt"),
    default_source=str(BASE_DIR / "demo.mp4")
)

# ==============================================================================
# Flask Routes & APIs
# ==============================================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        detector.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/stats')
def api_stats():
    return jsonify(detector.get_stats())

@app.route('/api/set_confidence', methods=['POST'])
def set_confidence():
    data = request.get_json() or {}
    conf = data.get('confidence', 0.35)
    detector.set_confidence(conf)
    return jsonify({"status": "success", "confidence": detector.confidence})

@app.route('/api/select_video', methods=['POST'])
def select_video():
    data = request.get_json() or {}
    video_name = data.get('video_name')

    if not video_name:
        return jsonify({"status": "error", "message": "No video name specified"}), 400

    if video_name == "demo.mp4":
        target_path = BASE_DIR / "demo.mp4"
    else:
        target_path = UPLOAD_DIR / secure_filename(video_name)

    if not target_path.exists():
        return jsonify({"status": "error", "message": "File not found"}), 404

    detector.set_source(str(target_path))
    return jsonify({"status": "success", "active_video": video_name})

@app.route('/api/videos', methods=['GET'])
def list_videos():
    videos = ["demo.mp4"]
    if UPLOAD_DIR.exists():
        for file in UPLOAD_DIR.iterdir():
            if file.suffix.lower() in ALLOWED_EXTENSIONS:
                videos.append(file.name)
    return jsonify({"videos": videos, "active": Path(detector.source).name})

@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "Empty filename"}), 400

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return jsonify({"status": "error", "message": f"Unsupported format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    filename = secure_filename(file.filename)
    save_path = UPLOAD_DIR / filename
    file.save(str(save_path))

    # Automatically activate the uploaded video
    detector.set_source(str(save_path))
    return jsonify({
        "status": "success",
        "message": f"Uploaded {filename} successfully and switched feed",
        "filename": filename
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)