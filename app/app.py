import av
import sys
import time
import threading
import torch
import streamlit as st
import torchvision.transforms as T
from PIL import Image
from collections import deque
from streamlit_webrtc import webrtc_streamer, WebRtcMode

sys.path.append('..')
from src.utils.model import AgeNet

# --- configs ---
MODEL_PATH = "../data/model/AgeNet.pt"
IMG_SIZE   = 224
N_FRAMES   = 4
THRESHOLD  = 0.9339  # classification threshold
PREDICT_INTERVAL = 1.0      # seconds between 
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
 
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

@st.cache_resource
def load_model():
    model = AgeNet(seq_len=N_FRAMES, d_model=1280, nhead=8)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.backbone.freeze()
    model.use_precompute(False)   # raw image path: backbone -> transformer
    model.to(DEVICE)
    model.eval()
    return model
 
 
def predict(model, frames):
    clip = torch.stack([transform(f.convert("RGB")) for f in frames])  # (T, C, H, W)
    clip = clip.unsqueeze(0).to(DEVICE)                                # (1, T, C, H, W)
 
    with torch.inference_mode():
        person_logit, age_pred = model(clip)
        person_prob = torch.sigmoid(person_logit).item()
        age         = age_pred.item()
 
    return person_prob >= THRESHOLD, person_prob, age
 
 
class LiveFramePredictor:
    """
    Holds a rolling buffer of recent frames and the latest prediction.
    Runs inference on a background thread so the video stream never stalls
    waiting for the model.
    """
    def __init__(self, model):
        self.model    = model
        self.buffer   = deque(maxlen=N_FRAMES)
        self.lock     = threading.Lock()
        self.last_run = 0.0
        self.latest   = {"is_person": None, "prob": None, "age": None, "ms": None}
        self.busy     = False
 
    def on_frame(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_image()  # PIL Image, RGB
 
        with self.lock:
            self.buffer.append(img)
            ready_to_run = (
                len(self.buffer) == N_FRAMES
                and not self.busy
                and (time.time() - self.last_run) >= PREDICT_INTERVAL
            )
            frames_snapshot = list(self.buffer) if ready_to_run else None
 
        if frames_snapshot is not None:
            self.busy = True
            threading.Thread(target=self._run_inference, args=(frames_snapshot,), daemon=True).start()
 
        return frame  # pass the original frame through untouched for display
 
    def _run_inference(self, frames):
        t0 = time.time()
        try:
            is_person, prob, age = predict(self.model, frames)
            with self.lock:
                self.latest = {
                    "is_person": is_person,
                    "prob":      prob,
                    "age":       age,
                    "ms":        (time.time() - t0) * 1000,
                }
                self.last_run = time.time()
        finally:
            self.busy = False
 
    def get_latest(self):
        with self.lock:
            return dict(self.latest)
 
 
# ── UI ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AgeNet Live Demo", layout="centered")
st.title("Age estimation - Live webcam")
 
model_load_error = None
try:
    model = load_model()
except Exception as e:
    model_load_error = str(e)
 
if model_load_error:
    st.error(f"Could not load model from `{MODEL_PATH}`: {model_load_error}")
    st.info("Place your trained `AgeNet.pt` in the `model/` folder next to this app.")
    st.stop()
 
if "predictor" not in st.session_state:
    st.session_state.predictor = LiveFramePredictor(model)
 
predictor = st.session_state.predictor
 
ctx = webrtc_streamer(
    key="agenet-live",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=predictor.on_frame,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
 
st.subheader("Live prediction")
placeholder = st.empty()
 
if ctx.state.playing:
    result = predictor.get_latest()
 
    with placeholder.container():
        if result["is_person"] is None:
            st.info(f"Collecting frames... need {N_FRAMES} buffered frames before the first prediction.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Person detected", "Yes" if result["is_person"] else "No")
            m2.metric("Confidence", f"{result['prob'] * 100:.1f}%")
            m3.metric("Estimated age", f"{result['age']:.1f} yrs" if result["is_person"] else "—")
            st.caption(f"Last inference: {result['ms']:.0f} ms on {DEVICE.type.upper()}")
 
            if not result["is_person"]:
                st.warning("No person detected with sufficient confidence — age estimate is not meaningful.")
 
    time.sleep(0.5)
    st.rerun()
else:
    placeholder.info("Click **Start** above to begin the live webcam stream.")