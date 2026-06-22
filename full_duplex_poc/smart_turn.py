import logging
import numpy as np

logger = logging.getLogger(__name__)

class SilenceTurnDetector:
    """Fallback turn detector that always returns unfinished. 
    It relies on the TurnCollector's MAX_SILENCE_FALLBACK_MS to end the turn."""
    
    def predict(self, user_audio_buffer: np.ndarray, agent_audio_buffer=None) -> dict:
        return {
            "finished": False,
            "score": 0.0,
            "reason": "silence_fallback"
        }

class SmartTurnMonoONNX:
    """SmartTurn end-of-turn detector wrapping ONNX session and feature extractor."""

    def __init__(
        self,
        model_path: str = "/Utilisateurs/tnguye28/smart-turn/smart-turn-v3.1.onnx",
        threshold: float = 0.90,
        min_speech_ms: float = 500.0,
        fs: int = 16000,
    ) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.fs = fs
        self._session = None
        self._feature_extractor = None

    def _load(self) -> None:
        """Lazy-load SmartTurn ONNX session and feature extractor."""
        if self._session is not None:
            return

        logger.info("[SmartTurn] Loading SmartTurn model from %s...", self.model_path)
        try:
            import onnxruntime as ort
            from transformers import WhisperFeatureExtractor
            
            self._feature_extractor = WhisperFeatureExtractor(chunk_length=8)

            so = ort.SessionOptions()
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._session = ort.InferenceSession(self.model_path, sess_options=so)
            logger.info("✓ [SmartTurn] SmartTurn model ready")
        except ImportError as e:
            logger.error(f"[SmartTurn] Failed to load ONNX runtime or transformers: {e}")
            raise
        except Exception as e:
            logger.error(f"[SmartTurn] Failed to load ONNX model at {self.model_path}: {e}")
            raise

    def predict(self, user_audio_buffer: np.ndarray, agent_audio_buffer=None) -> dict:
        """Predict whether the current audio segment is a complete turn.
        
        Args:
            user_audio_buffer: Numpy array of 16kHz mono audio samples.
        """
        try:
            self._load()
        except Exception:
            # Fallback to silence
            return {"finished": False, "score": 0.0, "reason": "load_error"}

        sampling_rate = self.fs
        # SmartTurn operates on last 8 seconds of audio
        max_samples = 8 * sampling_rate
        if len(user_audio_buffer) > max_samples:
            user_audio_buffer = user_audio_buffer[-max_samples:]
        elif len(user_audio_buffer) < max_samples:
            # Pad with zeros at the beginning
            padding = max_samples - len(user_audio_buffer)
            user_audio_buffer = np.pad(user_audio_buffer, (padding, 0), mode='constant', constant_values=0)

        try:
            inputs = self._feature_extractor(
                user_audio_buffer,
                sampling_rate=sampling_rate,
                return_tensors="np",
                padding="max_length",
                max_length=max_samples,
                truncation=True,
                do_normalize=True,
            )

            input_features = inputs.input_features.squeeze(0).astype(np.float32)
            input_features = np.expand_dims(input_features, axis=0)  # Add batch dim

            outputs = self._session.run(None, {"input_features": input_features})
            probability = float(outputs[0][0].item())
            finished = probability >= self.threshold

            return {
                "finished": finished,
                "score": probability,
                "reason": "smartturn"
            }
        except Exception as e:
            logger.error(f"[SmartTurn] Inference error: {e}")
            return {"finished": False, "score": 0.0, "reason": "inference_error"}

def get_turn_detector(model_type="smartturn"):
    if model_type == "smartturn":
        import os
        if os.path.exists("/Utilisateurs/tnguye28/smart-turn/smart-turn-v3.1.onnx"):
            try:
                import onnxruntime
                return SmartTurnMonoONNX()
            except ImportError:
                logger.warning("[SmartTurn] onnxruntime not found. Falling back to SilenceTurnDetector.")
                return SilenceTurnDetector()
        else:
            logger.warning("[SmartTurn] ONNX model not found. Falling back to SilenceTurnDetector.")
            return SilenceTurnDetector()
    
    return SilenceTurnDetector()
