import io
import os
import json
import logging
from typing import Any, Dict, List
import numpy as np
from PIL import Image

# Import the Google GenAI SDK
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

def _convert_frames_to_pil_bytes(frames: List[np.ndarray], max_frames: int = 10) -> List[bytes]:
    """
    Sub-samples frames evenly across the video sequence and encodes them to JPEG bytes.
    """
    if not frames:
        return []

    # Evenly sample frames up to max_frames to reduce token payload and API latency
    total_frames = len(frames)
    indices = np.linspace(0, total_frames - 1, min(total_frames, max_frames), dtype=int)
    
    encoded_bytes = []
    for idx in indices:
        frame = frames[idx]
        if frame is None or frame.size == 0:
            continue
            
        # Convert BGR (OpenCV format) to RGB for PIL
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = frame[:, :, ::-1]
        else:
            rgb_frame = frame

        img = Image.fromarray(rgb_frame)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=80)
        encoded_bytes.append(buffer.getvalue())

    return encoded_bytes


def analyze_with_gemini(
    frames: List[np.ndarray],
    mode: str,
) -> Dict[str, Any]:
    """
    Fallback analysis service using Gemini multimodal vision capability.
    Degrades to a standard fallback response if Gemini API is unconfigured or fails.
    """
    frame_count = len(frames)
    api_key = os.environ.get("GEMINI_API_KEY")

    # Guard clause: If no API key is available, return local fallback message immediately
    if not api_key or not frames:
        return _build_local_fallback_response(
            mode=mode,
            frame_count=frame_count,
            reason="Gemini API key not configured or no frames provided."
        )

    try:
        # Initialize Gemini Client
        client = genai.Client(api_key=api_key)

        # Downsample and convert numpy array frames into JPEG byte buffers
        image_bytes_list = _convert_frames_to_pil_bytes(frames, max_frames=12)

        if not image_bytes_list:
            return _build_local_fallback_response(
                mode=mode,
                frame_count=frame_count,
                reason="Failed to convert video frames to valid images."
            )

        # Construct multimodal contents list
        contents = []
        for img_bytes in image_bytes_list:
            contents.append(
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
            )

        prompt = f"""
        You are an expert sports biomechanics and movement safety assistant analyzing a sequence of video frames.
        The movement mode is: '{mode}'.

        Analyze the movement performed across these frames:
        1. Identify any form errors, suboptimal movement patterns, or potential biomechanical issues.
        2. Provide clear, actionable feedback for improvement.

        Return your output strictly following this JSON structure:
        - "issues": List of objects, each containing:
            - "joint": string name of joint or body region (e.g., "knee", "spine", "shoulder")
            - "issue": short description of error
            - "value": optional observed metric or descriptive string
            - "target": target recommendation or optimal range
        - "feedback": Concise overview paragraph explaining the form analysis and recommendations.
        """
        contents.append(prompt)

        # Define JSON output schema
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "issues": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "joint": {"type": "STRING"},
                            "issue": {"type": "STRING"},
                            "value": {"type": "STRING"},
                            "target": {"type": "STRING"}
                        },
                        "required": ["joint", "issue"]
                    }
                },
                "feedback": {"type": "STRING"}
            },
            "required": ["issues", "feedback"]
        }

        # Request Gemini generation
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.2,
            )
        )

        parsed_data = json.loads(response.text)

        return {
            "mode": mode,
            "used_fallback": True,
            "issues": parsed_data.get("issues", []),
            "feedback": parsed_data.get("feedback", "No detailed feedback generated."),
            "diagnostics": {
                "frames_available": frame_count,
                "frames_analyzed": len(image_bytes_list),
                "fallback_provider": "gemini-3.5-flash",
            },
        }

    except Exception as exc:
        logger.warning(f"Gemini API analysis failed: {exc}. Falling back to standard message.")
        return _build_local_fallback_response(
            mode=mode,
            frame_count=frame_count,
            reason=f"Gemini service execution error: {str(exc)}"
        )


def _build_local_fallback_response(mode: str, frame_count: int, reason: str = "") -> Dict[str, Any]:
    """
    Standard deterministic fallback response when automated AI vision inference is unavailable.
    """
    return {
        "mode": mode,
        "used_fallback": True,
        "issues": [],
        "feedback": (
            "The pose could not be detected reliably in enough video frames. "
            "Please upload a clear, full-body video with a steady camera, good lighting, "
            "and the entire movement visible."
        ),
        "diagnostics": {
            "frames_available": frame_count,
            "fallback_provider": "local_message",
            "detail": reason,
        },
    }
