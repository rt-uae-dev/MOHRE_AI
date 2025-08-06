"""Utility to compress images to meet size constraints."""

from PIL import Image
import os


def compress_image_to_jpg(image_path: str, output_path: str, max_kb: int = 250) -> str:
    """Compress an image to JPEG format under a size threshold.

    Args:
        image_path: Path to the source image to compress.
        output_path: Location where the compressed JPEG will be saved.
        max_kb: Target maximum size in kilobytes for the compressed file.

    Returns:
        Path to the compressed JPEG file.

    Raises:
        FileNotFoundError: If ``image_path`` does not exist.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img = Image.open(image_path).convert("RGB")

    # Use modern resampling filter
    resample = Image.Resampling.LANCZOS

    quality = 95
    img.save(output_path, "JPEG", quality=quality)

    # Try reducing quality with a higher minimum threshold for document readability
    min_quality = 35  # Increased from 10 to 35 for better document quality
    while os.path.getsize(output_path) > max_kb * 1024 and quality > min_quality:
        quality -= 5
        img.save(output_path, "JPEG", quality=quality)

    # Resize if still too large, but be more conservative
    if os.path.getsize(output_path) > max_kb * 1024:
        width, height = img.size
        min_width = 600  # Increased from 300 to 600 to maintain readability
        while os.path.getsize(output_path) > max_kb * 1024 and width > min_width:
            width = int(width * 0.9)
            height = int(height * 0.9)
            resized = img.resize((width, height), resample)
            resized.save(output_path, "JPEG", quality=quality)

    # Final check
    final_kb = os.path.getsize(output_path) / 1024
    if final_kb > max_kb:
        print(f"⚠️ Warning: Could not compress below {max_kb}KB. Final size: {final_kb:.2f}KB")

    print(f"✅ Compressed {os.path.basename(image_path)} to {final_kb:.1f}KB (quality={quality})")
    return output_path
