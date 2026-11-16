#!/usr/bin/env python3
"""
SENTINEL Phase 1 - Video Frame Extraction Demo

This script demonstrates the video input module by:
1. Loading a video file (or using mock adapter)
2. Extracting frames using OpenCV
3. Preprocessing frames (resize, normalize)
4. Displaying frame information and quality metrics
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.video import create_video_adapter, VideoStreamProcessor
from app.config import settings
from loguru import logger


async def process_video(source: str, frame_skip: int = 1, max_frames: int = 10, display: bool = False):
    """Process video and extract frames."""
    
    logger.info(f"Creating video adapter for: {source}")
    adapter = create_video_adapter(source, frame_skip=frame_skip)
    
    connected = await adapter.connect()
    if not connected:
        logger.error(f"Failed to connect to video source: {source}")
        return False
    
    try:
        info = await adapter.get_info()
        logger.info(f"Video Info: {info.width}x{info.height} @ {info.fps:.2f}fps, "
                   f"{info.frame_count} frames, {info.duration:.2f}s duration, codec: {info.codec}")
        
        processor = VideoStreamProcessor(adapter, frame_skip=frame_skip, enable_quality_check=True)
        
        frame_count = 0
        async for frame_data in processor.process_stream():
            frame = frame_data["frame"]
            processed = frame_data["processed"]
            quality = frame_data["quality"]
            
            logger.info(
                f"Frame #{frame.frame_number}: "
                f"{frame.width}x{frame.height} @ {frame.timestamp:.3f}s | "
                f"Processed: {processed.processed.shape} | "
                f"Quality: {quality['quality_score']:.2f} "
                f"({'PASS' if quality['passed'] else 'FAIL'})"
            )
            
            if display:
                import cv2
                # Show original frame
                cv2.imshow("Original Frame", frame.frame)
                # Show processed (denormalized for display)
                display_frame = (processed.processed[0].transpose(1, 2, 0) * 255).astype('uint8')
                cv2.imshow("Processed Frame", display_frame)
                
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    break
            
            frame_count += 1
            if frame_count >= max_frames:
                break
        
        stats = processor.get_stats()
        logger.info(f"Processing complete. Stats: {stats}")
        
    finally:
        await adapter.disconnect()
        if display:
            import cv2
            cv2.destroyAllWindows()
    
    return True


async def main():
    parser = argparse.ArgumentParser(description="SENTINEL Phase 1 - Video Frame Extraction")
    parser.add_argument("source", nargs="?", default="mock", 
                       help="Video source: file path, rtsp:// URL, or 'mock' for testing")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=10, help="Maximum frames to process")
    parser.add_argument("--display", action="store_true", help="Display frames using OpenCV")
    parser.add_argument("--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)")
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    
    logger.info("=" * 60)
    logger.info("SENTINEL Phase 1 - Video Frame Extraction")
    logger.info("=" * 60)
    
    success = await process_video(
        source=args.source,
        frame_skip=args.frame_skip,
        max_frames=args.max_frames,
        display=args.display
    )
    
    if success:
        logger.info("Phase 1 video processing completed successfully!")
        return 0
    else:
        logger.error("Phase 1 video processing failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)