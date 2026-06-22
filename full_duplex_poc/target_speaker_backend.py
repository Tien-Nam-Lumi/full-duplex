import os
import logging
import numpy as np
import torch
import threading
from .audio_utils import load_audio_mono_16k

logger = logging.getLogger(__name__)

# Hardcode the repo and checkpoint for speechbrain_ft based on Pvad
SB_FT_REPO_ID = "Nampfiev1995/pvad-speechbrain-ft"
SB_FT_CHECKPOINT = "best_checkpoint_rec98.pt"

_SPEECHBRAIN_FT_MODEL = None
_model_lock = threading.Lock()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def resolve_ft_checkpoint() -> str:
    """Download fine-tuned checkpoint from HF Hub."""
    logger.info(f"Resolving fine-tuned checkpoint '{SB_FT_CHECKPOINT}' from HF repo '{SB_FT_REPO_ID}'...")
    try:
        from huggingface_hub import hf_hub_download
        checkpoint_path = hf_hub_download(repo_id=SB_FT_REPO_ID, filename=SB_FT_CHECKPOINT)
        logger.info(f"Resolved fine-tuned checkpoint to: {checkpoint_path}")
        return checkpoint_path
    except Exception as e:
        logger.error(f"Failed to download fine-tuned checkpoint: {e}")
        raise RuntimeError(f"Failed to resolve fine-tuned checkpoint. Error: {e}")

def _load_speechbrain_ft():
    """Lazy-load SpeechBrain fine-tuned model on device."""
    global _SPEECHBRAIN_FT_MODEL
    import torch
    from speechbrain.inference.speaker import SpeakerRecognition
    
    savedir = os.path.join(os.path.expanduser("~"), ".cache", "speechbrain")
    
    if _SPEECHBRAIN_FT_MODEL is None:
        with _model_lock:
            if _SPEECHBRAIN_FT_MODEL is None:
                logger.info("Loading SpeechBrain ECAPA-TDNN (Fine-tuned)...")
                model = SpeakerRecognition.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=savedir,
                    run_opts={"device": DEVICE},
                )
                checkpoint_path = resolve_ft_checkpoint()
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                model.mods.embedding_model.load_state_dict(state_dict, strict=True)
                logger.info("✓ Fine-tuned ECAPA-TDNN model loaded")
                _SPEECHBRAIN_FT_MODEL = model
    return _SPEECHBRAIN_FT_MODEL

def extract_embedding(audio: np.ndarray) -> np.ndarray | None:
    """Extract a 1D speaker embedding for speechbrain_ft.
    
    audio is assumed to be mono float32 16kHz.
    Returns L2-normalized 1D numpy array.
    """
    model = _load_speechbrain_ft()
    audio_t = torch.from_numpy(audio).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb_t = model.encode_batch(audio_t)  # [1, 1, 192]
        emb = emb_t.squeeze(0).squeeze(0).cpu().numpy()
        
    # Standard L2 normalization
    norm = np.linalg.norm(emb)
    if norm > 1e-8:
        emb = emb / norm
    return emb

def enroll_speaker(audio_inputs: list) -> tuple[str, np.ndarray | None]:
    """Enroll a target speaker from up to 3 audio clips, returning (html_msg, embedding)."""
    embeddings = []
    total_duration = 0.0
    
    for inp in audio_inputs:
        if inp is None:
            continue
        try:
            audio = load_audio_mono_16k(inp)
            duration = len(audio) / 16000.0
            if duration < 0.5:
                logger.warning("Enrollment clip too short (< 0.5s), skipping")
                continue
                
            total_duration += duration
            emb = extract_embedding(audio)
            if emb is not None:
                embeddings.append(emb)
        except Exception as e:
            logger.error(f"Failed to process enrollment clip: {e}")
            
    if not embeddings:
        return (
            "❌ Please record or upload at least one valid audio clip (>0.5s)",
            None
        )
        
    # Average all embeddings
    avg_emb = np.mean(embeddings, axis=0).astype(np.float32)
    norm = np.linalg.norm(avg_emb)
    if norm > 1e-8:
        avg_emb = avg_emb / norm
        
    dim = len(avg_emb)
    n_clips = len(embeddings)
    
    return (
        f"✅ Enrolled with {n_clips} clip(s)! Total duration: {total_duration:.1f}s",
        avg_emb
    )
