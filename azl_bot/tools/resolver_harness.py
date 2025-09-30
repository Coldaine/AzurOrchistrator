"""Offline resolver harness for testing and calibration without emulator."""

import argparse
import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

import cv2
import numpy as np
from loguru import logger

from azl_bot.core.capture import Frame
from azl_bot.core.configs import load_config, create_default_config
from azl_bot.core.llm_client import Target
from azl_bot.core.ocr import OCRClient
from azl_bot.core.resolver import Resolver, Candidate


@dataclass
class ResolveResult:
    """Result of a resolve operation."""
    image_path: str
    target_kind: str
    target_value: str
    candidates: List[Dict[str, Any]]
    top_candidate: Optional[Dict[str, Any]]
    resolve_time_ms: float
    method_times: Dict[str, float]


def load_image_as_frame(image_path: Path) -> Frame:
    """Load an image file as a Frame object.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Frame object with the image
    """
    # Load image
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    h, w = image_bgr.shape[:2]
    
    # Create a minimal Frame object (no letterbox detection for offline use)
    # Assume the entire image is the active area
    frame = Frame(
        png_bytes=b"",  # Not needed for offline use
        image_bgr=image_bgr,
        full_w=w,
        full_h=h,
        active_rect=(0, 0, w, h),
        ts=time.time()
    )
    
    return frame


def draw_candidates_overlay(
    image: np.ndarray, 
    candidates: List[Candidate], 
    top_candidate: Optional[Candidate],
    frame_w: int,
    frame_h: int
) -> np.ndarray:
    """Draw candidate points on image with labels and confidence.
    
    Args:
        image: Input image (BGR)
        candidates: List of candidate points
        top_candidate: The selected top candidate
        frame_w: Frame width for coordinate conversion
        frame_h: Frame height for coordinate conversion
        
    Returns:
        Image with overlays drawn
    """
    overlay = image.copy()
    
    # Color scheme
    colors = {
        "ocr": (0, 255, 0),       # Green
        "template": (255, 0, 0),   # Blue
        "orb": (0, 165, 255),      # Orange
        "region_hint": (128, 128, 128),  # Gray
        "llm_arbitration": (255, 0, 255),  # Magenta
        "direct": (255, 255, 0)    # Cyan
    }
    
    # Draw all candidates
    for i, candidate in enumerate(candidates):
        x_norm, y_norm = candidate.point
        x_px = int(x_norm * frame_w)
        y_px = int(y_norm * frame_h)
        
        color = colors.get(candidate.method, (200, 200, 200))
        
        # Draw circle
        is_top = top_candidate and candidate == top_candidate
        radius = 12 if is_top else 8
        thickness = 3 if is_top else 2
        cv2.circle(overlay, (x_px, y_px), radius, color, thickness)
        
        # Draw label
        label = f"{candidate.method[:3]} {candidate.confidence:.2f}"
        label_pos = (x_px + 15, y_px - 10)
        cv2.putText(overlay, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, color, 1, cv2.LINE_AA)
    
    # Draw crosshair on top candidate
    if top_candidate:
        x_norm, y_norm = top_candidate.point
        x_px = int(x_norm * frame_w)
        y_px = int(y_norm * frame_h)
        color = colors.get(top_candidate.method, (255, 255, 255))
        
        # Draw crosshair
        cv2.line(overlay, (x_px - 20, y_px), (x_px + 20, y_px), color, 2)
        cv2.line(overlay, (x_px, y_px - 20), (x_px, y_px + 20), color, 2)
    
    return overlay


def resolve_offline(
    resolver: Resolver,
    frame: Frame,
    target: Target
) -> ResolveResult:
    """Run resolver on a frame and collect detailed timing.
    
    Args:
        resolver: Resolver instance
        frame: Frame to process
        target: Target to resolve
        
    Returns:
        Detailed resolve result
    """
    start_time = time.time()
    method_times = {}
    
    # Get all candidates with timing
    candidates_list = []
    
    # Try text detection
    if target.kind == "text":
        text_start = time.time()
        text_candidates = resolver._detect_by_text(target, frame)
        method_times["text"] = (time.time() - text_start) * 1000
        candidates_list.extend(text_candidates)
    
    # Try template detection
    if target.kind == "icon":
        template_start = time.time()
        template_candidates = resolver._detect_by_template(target, frame)
        method_times["template"] = (time.time() - template_start) * 1000
        candidates_list.extend(template_candidates)
    
    # Try region detection
    if target.kind == "region":
        region_start = time.time()
        region_candidates = resolver._detect_by_region(target, frame)
        method_times["region"] = (time.time() - region_start) * 1000
        candidates_list.extend(region_candidates)
    
    # Get top candidate
    if candidates_list:
        candidates_list.sort(key=lambda c: c.confidence, reverse=True)
        top_candidate = candidates_list[0]
    else:
        top_candidate = None
    
    total_time = (time.time() - start_time) * 1000
    
    # Convert candidates to dicts
    candidates_dicts = [
        {
            "method": c.method,
            "x_norm": c.point[0],
            "y_norm": c.point[1],
            "confidence": c.confidence
        }
        for c in candidates_list
    ]
    
    top_dict = None
    if top_candidate:
        top_dict = {
            "method": top_candidate.method,
            "x_norm": top_candidate.point[0],
            "y_norm": top_candidate.point[1],
            "confidence": top_candidate.confidence
        }
    
    return ResolveResult(
        image_path="",
        target_kind=target.kind,
        target_value=target.value,
        candidates=candidates_dicts,
        top_candidate=top_dict,
        resolve_time_ms=total_time,
        method_times=method_times
    )


def process_directory(
    input_path: Path,
    target: Optional[Target],
    resolver: Resolver,
    save_overlays: bool,
    out_dir: Optional[Path]
) -> List[ResolveResult]:
    """Process all images in a directory.
    
    Args:
        input_path: Directory containing images
        target: Target to resolve (None to skip resolution)
        resolver: Resolver instance
        save_overlays: Whether to save overlay images
        out_dir: Output directory for overlays
        
    Returns:
        List of resolve results
    """
    results = []
    
    # Find all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.rglob(f"*{ext}"))
    
    logger.info(f"Found {len(image_files)} images in {input_path}")
    
    for img_path in sorted(image_files):
        logger.info(f"Processing: {img_path.name}")
        
        try:
            frame = load_image_as_frame(img_path)
            
            if target:
                result = resolve_offline(resolver, frame, target)
                result.image_path = str(img_path)
                results.append(result)
                
                logger.info(f"  Resolved in {result.resolve_time_ms:.1f}ms")
                if result.top_candidate:
                    top = result.top_candidate
                    logger.info(f"  Top: {top['method']} ({top['x_norm']:.3f}, {top['y_norm']:.3f}) conf={top['confidence']:.2f}")
                
                # Save overlay if requested
                if save_overlays and out_dir and result.candidates:
                    # Reconstruct Candidate objects for drawing
                    candidates = [
                        Candidate(
                            point=(c["x_norm"], c["y_norm"]),
                            confidence=c["confidence"],
                            method=c["method"]
                        )
                        for c in result.candidates
                    ]
                    
                    top_cand = None
                    if result.top_candidate:
                        top_cand = Candidate(
                            point=(result.top_candidate["x_norm"], result.top_candidate["y_norm"]),
                            confidence=result.top_candidate["confidence"],
                            method=result.top_candidate["method"]
                        )
                    
                    overlay = draw_candidates_overlay(
                        frame.image_bgr, candidates, top_cand,
                        frame.full_w, frame.full_h
                    )
                    
                    out_path = out_dir / f"{img_path.stem}_overlay.png"
                    cv2.imwrite(str(out_path), overlay)
                    logger.debug(f"  Saved overlay: {out_path}")
            else:
                # Just load and optionally save without resolution
                if save_overlays and out_dir:
                    out_path = out_dir / f"{img_path.stem}_copy.png"
                    cv2.imwrite(str(out_path), frame.image_bgr)
                    
        except Exception as e:
            logger.error(f"  Failed to process {img_path}: {e}")
    
    return results


def save_results_csv(results: List[ResolveResult], output_path: Path):
    """Save results to CSV file.
    
    Args:
        results: List of resolve results
        output_path: Path to output CSV file
    """
    with open(output_path, 'w', newline='') as f:
        fieldnames = [
            'image_path', 'target_kind', 'target_value',
            'top_method', 'top_x_norm', 'top_y_norm', 'top_confidence',
            'resolve_time_ms', 'num_candidates'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row = {
                'image_path': result.image_path,
                'target_kind': result.target_kind,
                'target_value': result.target_value,
                'top_method': result.top_candidate['method'] if result.top_candidate else '',
                'top_x_norm': result.top_candidate['x_norm'] if result.top_candidate else '',
                'top_y_norm': result.top_candidate['y_norm'] if result.top_candidate else '',
                'top_confidence': result.top_candidate['confidence'] if result.top_candidate else '',
                'resolve_time_ms': result.resolve_time_ms,
                'num_candidates': len(result.candidates)
            }
            writer.writerow(row)
    
    logger.info(f"Saved CSV results to {output_path}")


def save_results_json(results: List[ResolveResult], output_path: Path):
    """Save results to JSON file.
    
    Args:
        results: List of resolve results
        output_path: Path to output JSON file
    """
    results_dicts = [asdict(result) for result in results]
    
    with open(output_path, 'w') as f:
        json.dump(results_dicts, f, indent=2)
    
    logger.info(f"Saved JSON results to {output_path}")


def print_summary(results: List[ResolveResult]):
    """Print summary statistics.
    
    Args:
        results: List of resolve results
    """
    if not results:
        logger.info("No results to summarize")
        return
    
    total_time = sum(r.resolve_time_ms for r in results)
    avg_time = total_time / len(results)
    
    # Collect method times
    method_times_agg = {}
    for result in results:
        for method, time_ms in result.method_times.items():
            if method not in method_times_agg:
                method_times_agg[method] = []
            method_times_agg[method].append(time_ms)
    
    # Success rate
    success_count = sum(1 for r in results if r.top_candidate)
    success_rate = success_count / len(results) * 100
    
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Total images processed: {len(results)}")
    logger.info(f"Success rate: {success_rate:.1f}% ({success_count}/{len(results)})")
    logger.info(f"Average resolve time: {avg_time:.1f}ms")
    logger.info(f"Total time: {total_time:.1f}ms")
    logger.info("\nMethod timings (average):")
    for method, times in sorted(method_times_agg.items()):
        avg_method_time = sum(times) / len(times)
        logger.info(f"  {method}: {avg_method_time:.1f}ms")
    logger.info("="*60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Offline resolver harness for testing and calibration"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input image file or directory"
    )
    parser.add_argument(
        "--target", "-t",
        type=str,
        help="Target to resolve in format 'kind:value' (e.g., 'text:Commissions', 'icon:battle_icon')"
    )
    parser.add_argument(
        "--save-overlays",
        action="store_true",
        help="Save overlay images with detection visualizations"
    )
    parser.add_argument(
        "--out", "-o",
        type=Path,
        default=Path("./harness_output"),
        help="Output directory for results and overlays"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to config file (default: use AZL_CONFIG env or defaults)"
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json", "both"],
        default="both",
        help="Output format for results"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger.info("Offline Resolver Harness")
    logger.info(f"Input: {args.input}")
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()
    
    # Initialize components
    logger.info("Initializing OCR and Resolver...")
    ocr = OCRClient(
        backend=config.resolver.ocr,
        config=config.resolver.thresholds.model_dump()
    )
    
    templates_dir = str(Path("./config/templates").absolute())
    resolver = Resolver(
        config=config.resolver.model_dump(),
        ocr_client=ocr,
        templates_dir=templates_dir,
        llm=None  # No LLM for offline harness
    )
    
    # Parse target if provided
    target = None
    if args.target:
        parts = args.target.split(":", 1)
        if len(parts) != 2:
            logger.error("Target must be in format 'kind:value'")
            return
        
        kind, value = parts
        if kind not in ["text", "icon", "region", "point", "bbox"]:
            logger.error(f"Unknown target kind: {kind}")
            return
        
        target = Target(kind=kind, value=value)
        logger.info(f"Target: {kind}='{value}'")
    
    # Create output directory
    args.out.mkdir(parents=True, exist_ok=True)
    
    # Process input
    if args.input.is_file():
        logger.info("Processing single image")
        frame = load_image_as_frame(args.input)
        
        if target:
            result = resolve_offline(resolver, frame, target)
            result.image_path = str(args.input)
            results = [result]
            
            # Save overlay if requested
            if args.save_overlays and result.candidates:
                candidates = [
                    Candidate(
                        point=(c["x_norm"], c["y_norm"]),
                        confidence=c["confidence"],
                        method=c["method"]
                    )
                    for c in result.candidates
                ]
                
                top_cand = None
                if result.top_candidate:
                    top_cand = Candidate(
                        point=(result.top_candidate["x_norm"], result.top_candidate["y_norm"]),
                        confidence=result.top_candidate["confidence"],
                        method=result.top_candidate["method"]
                    )
                
                overlay = draw_candidates_overlay(
                    frame.image_bgr, candidates, top_cand,
                    frame.full_w, frame.full_h
                )
                
                out_path = args.out / f"{args.input.stem}_overlay.png"
                cv2.imwrite(str(out_path), overlay)
                logger.info(f"Saved overlay: {out_path}")
        else:
            results = []
            
    elif args.input.is_dir():
        logger.info("Processing directory")
        results = process_directory(
            args.input, target, resolver,
            args.save_overlays, args.out if args.save_overlays else None
        )
    else:
        logger.error(f"Input not found: {args.input}")
        return
    
    # Save results
    if results:
        if args.format in ["csv", "both"]:
            save_results_csv(results, args.out / "results.csv")
        
        if args.format in ["json", "both"]:
            save_results_json(results, args.out / "results.json")
        
        print_summary(results)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
