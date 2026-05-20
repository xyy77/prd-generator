import json

from src.config import settings
from src.utils.vision_client import VisionClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


def image_analyst_node(state: dict, reference_context: str = "") -> dict:
    image_paths = state.get("image_paths", [])
    if not image_paths:
        logger.info("No image paths in state, skipping image analysis")
        return {"current_stage": "image_analysis_skipped"}

    if settings.vision_provider == "none":
        logger.info("Vision provider is 'none', skipping image analysis")
        return {"current_stage": "image_analysis_skipped"}

    product_idea = state.get("product_idea", "")

    try:
        vision = VisionClient()
        result = vision.analyze_images(image_paths, product_idea)
        logger.info("Image analysis complete, %d images analyzed", len(result.get("images", [])))
        return {
            "image_analysis": result,
            "current_stage": "image_analysis_complete",
        }
    except Exception as e:
        logger.error("Image analysis failed: %s", e)
        return {
            "image_analysis": {},
            "current_stage": "image_analysis_failed",
            "error_message": str(e),
        }
