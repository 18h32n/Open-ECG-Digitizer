"""
Synthetic ECG Image Generation for Training Data Augmentation.

This module generates realistic ECG images from time-series signals,
applying various degradations to match competition data characteristics.
"""

import numpy as np
import torch
import cv2
from typing import Tuple, Optional, Dict, Any
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random


class ECGImageGenerator:
    """Generate realistic ECG images from time-series signals."""

    def __init__(
        self,
        image_size: Tuple[int, int] = (2200, 1700),
        grid_spacing_mm: float = 5.0,
        mm_per_pixel: float = 0.1,
        mv_per_mm: float = 10.0,
        sampling_rate: int = 500,
        paper_speed: float = 25.0,  # mm/s
    ):
        """Initialize ECG image generator.

        Args:
            image_size: Output image size (width, height) in pixels
            grid_spacing_mm: Grid spacing in millimeters (major gridlines)
            mm_per_pixel: Resolution in mm/pixel
            mv_per_mm: Vertical calibration (typically 10 mm/mV)
            sampling_rate: Signal sampling rate in Hz
            paper_speed: Paper speed in mm/s (typically 25 or 50)
        """
        self.image_size = image_size
        self.grid_spacing_mm = grid_spacing_mm
        self.mm_per_pixel = mm_per_pixel
        self.mv_per_mm = mv_per_mm
        self.sampling_rate = sampling_rate
        self.paper_speed = paper_speed

        # Calculate derived parameters
        self.grid_spacing_px = int(grid_spacing_mm / mm_per_pixel)
        self.minor_grid_spacing_px = self.grid_spacing_px // 5

    def generate_grid(self, width: int, height: int) -> np.ndarray:
        """Generate ECG grid pattern.

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Grid image as numpy array (H, W, 3) in RGB
        """
        # Create white background
        grid = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Draw minor grid lines (1mm)
        minor_color = (255, 200, 200)  # Light red
        for x in range(0, width, self.minor_grid_spacing_px):
            grid[:, x, :] = minor_color
        for y in range(0, height, self.minor_grid_spacing_px):
            grid[y, :, :] = minor_color

        # Draw major grid lines (5mm)
        major_color = (255, 150, 150)  # Darker red
        for x in range(0, width, self.grid_spacing_px):
            grid[:, x, :] = major_color
        for y in range(0, height, self.grid_spacing_px):
            grid[y, :, :] = major_color

        return grid

    def signal_to_pixels(self, signal: np.ndarray, baseline_y: int) -> np.ndarray:
        """Convert signal values to pixel y-coordinates.

        Args:
            signal: Signal in mV
            baseline_y: Baseline y-coordinate in pixels

        Returns:
            Y-coordinates for each sample
        """
        # Convert mV to mm, then mm to pixels
        mm_displacement = signal * self.mv_per_mm
        px_displacement = mm_displacement / self.mm_per_pixel

        # Invert Y-axis (image coordinates increase downward)
        y_coords = baseline_y - px_displacement

        return y_coords.astype(int)

    def draw_signal(
        self,
        grid: np.ndarray,
        signal: np.ndarray,
        start_x: int,
        start_y: int,
        signal_color: Tuple[int, int, int] = (0, 0, 0),
        line_thickness: int = 2,
    ) -> np.ndarray:
        """Draw ECG signal on grid.

        Args:
            grid: Background grid image
            signal: Signal array in mV
            start_x: Starting x position in pixels
            start_y: Starting y position (baseline) in pixels
            signal_color: RGB color for signal
            line_thickness: Thickness of signal line

        Returns:
            Image with signal drawn
        """
        img = grid.copy()

        # Calculate x positions based on paper speed
        # samples_per_mm = sampling_rate / paper_speed
        # px_per_sample = mm_per_pixel * paper_speed / sampling_rate
        px_per_sample = self.mm_per_pixel * self.paper_speed / self.sampling_rate

        # Convert signal to pixel coordinates
        y_coords = self.signal_to_pixels(signal, start_y)
        x_coords = start_x + np.arange(len(signal)) * px_per_sample

        # Draw line segments
        for i in range(len(signal) - 1):
            x1, y1 = int(x_coords[i]), int(y_coords[i])
            x2, y2 = int(x_coords[i + 1]), int(y_coords[i + 1])

            # Clip to image bounds
            if 0 <= x1 < img.shape[1] and 0 <= y1 < img.shape[0]:
                if 0 <= x2 < img.shape[1] and 0 <= y2 < img.shape[0]:
                    cv2.line(img, (x1, y1), (x2, y2), signal_color, line_thickness)

        return img

    def generate_standard_12lead_layout(
        self,
        signals: Dict[str, np.ndarray],
        layout: str = "3x4+1R",
    ) -> np.ndarray:
        """Generate standard 12-lead ECG layout.

        Args:
            signals: Dictionary mapping lead names to signal arrays (in mV)
            layout: Layout type ("3x4+1R", "3x4+3R", "6x2", etc.)

        Returns:
            ECG image as numpy array
        """
        width, height = self.image_size
        grid = self.generate_grid(width, height)

        # Define layout: 3x4 grid with rhythm strip
        if layout == "3x4+1R":
            # 3 rows of 4 leads each
            rows = 3
            cols = 4
            lead_order = [
                ["I", "aVR", "V1", "V4"],
                ["II", "aVL", "V2", "V5"],
                ["III", "aVF", "V3", "V6"],
            ]
            rhythm_lead = "II"

            # Calculate dimensions
            margin = 100  # pixels
            col_width = (width - 2 * margin) // cols
            row_height = (height - 2 * margin - 200) // (rows + 1)  # +1 for rhythm

            # Draw grid leads (2.5 seconds each)
            for row in range(rows):
                for col in range(cols):
                    lead_name = lead_order[row][col]
                    if lead_name in signals:
                        signal = signals[lead_name]

                        # Truncate to 2.5 seconds
                        n_samples = int(2.5 * self.sampling_rate)
                        signal = signal[:n_samples]

                        # Calculate position
                        x = margin + col * col_width + 20
                        y = margin + row * row_height + row_height // 2

                        # Draw signal
                        grid = self.draw_signal(grid, signal, x, y)

            # Draw rhythm strip (10 seconds)
            if rhythm_lead in signals:
                signal = signals[rhythm_lead]
                n_samples = int(10.0 * self.sampling_rate)
                signal = signal[:n_samples]

                x = margin + 20
                y = margin + rows * row_height + row_height // 2 + 100

                grid = self.draw_signal(grid, signal, x, y)

        return grid

    def apply_degradations(
        self,
        image: np.ndarray,
        degradation_type: str = "clean",
        severity: float = 0.5,
    ) -> np.ndarray:
        """Apply realistic degradations to ECG image.

        Args:
            image: Clean ECG image
            degradation_type: Type of degradation to apply
            severity: Degradation severity (0 = none, 1 = maximum)

        Returns:
            Degraded image
        """
        img = image.copy()

        if degradation_type == "clean":
            return img

        elif degradation_type == "scanned_color":
            # Add scanning artifacts
            # Slight blur
            kernel_size = int(3 + severity * 2)
            if kernel_size % 2 == 0:
                kernel_size += 1
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

            # Add noise
            noise = np.random.normal(0, severity * 10, img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)

            # JPEG compression artifacts
            quality = int(95 - severity * 30)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        elif degradation_type == "scanned_bw":
            # Convert to grayscale then back to RGB
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Add threshold variation
            threshold = int(200 - severity * 50)
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

            img = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

        elif degradation_type == "photo":
            # Perspective distortion
            h, w = img.shape[:2]

            # Random perspective transform
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

            # Add random offset to corners
            max_offset = int(severity * 100)
            pts2 = np.float32([
                [random.randint(0, max_offset), random.randint(0, max_offset)],
                [w - random.randint(0, max_offset), random.randint(0, max_offset)],
                [random.randint(0, max_offset), h - random.randint(0, max_offset)],
                [w - random.randint(0, max_offset), h - random.randint(0, max_offset)],
            ])

            M = cv2.getPerspectiveTransform(pts1, pts2)
            img = cv2.warpPerspective(img, M, (w, h), borderValue=(255, 255, 255))

            # Add blur (out of focus)
            kernel_size = int(3 + severity * 4)
            if kernel_size % 2 == 0:
                kernel_size += 1
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

        elif degradation_type == "stained":
            # Add stains (coffee, water marks)
            n_stains = int(severity * 10)

            for _ in range(n_stains):
                # Random stain position and size
                cx = random.randint(0, img.shape[1])
                cy = random.randint(0, img.shape[0])
                radius = random.randint(50, 200)

                # Create circular stain
                y, x = np.ogrid[:img.shape[0], :img.shape[1]]
                mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2

                # Random brown color
                stain_color = np.array([
                    random.randint(150, 200),
                    random.randint(120, 160),
                    random.randint(80, 120)
                ])

                # Apply stain with alpha blending
                alpha = random.uniform(0.3, 0.7) * severity
                img[mask] = (1 - alpha) * img[mask] + alpha * stain_color

        elif degradation_type == "mold":
            # Add mold patterns (darker, irregular shapes)
            n_mold_spots = int(severity * 15)

            for _ in range(n_mold_spots):
                cx = random.randint(0, img.shape[1])
                cy = random.randint(0, img.shape[0])
                radius = random.randint(30, 150)

                # Irregular shape using random noise
                y, x = np.ogrid[:img.shape[0], :img.shape[1]]
                base_mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2

                # Add irregularity
                noise_mask = np.random.rand(img.shape[0], img.shape[1]) > 0.3
                mask = base_mask & noise_mask

                # Dark green/black color
                mold_color = np.array([
                    random.randint(100, 140),
                    random.randint(120, 150),
                    random.randint(80, 110)
                ])

                alpha = random.uniform(0.4, 0.8) * severity
                img[mask] = (1 - alpha) * img[mask] + alpha * mold_color

        elif degradation_type == "damaged":
            # Torn edges, creases
            # Add crease lines
            n_creases = int(severity * 5)

            for _ in range(n_creases):
                # Random line
                x1 = random.randint(0, img.shape[1])
                y1 = random.randint(0, img.shape[0])
                x2 = random.randint(0, img.shape[1])
                y2 = random.randint(0, img.shape[0])

                # Draw thick dark line
                thickness = random.randint(2, 8)
                cv2.line(img, (x1, y1), (x2, y2), (180, 180, 180), thickness)

            # Add torn edge effect by masking random edge regions
            if random.random() < severity:
                edge = random.choice(['top', 'bottom', 'left', 'right'])
                tear_depth = int(severity * 100)

                if edge == 'top':
                    img[:tear_depth, :] = 255
                elif edge == 'bottom':
                    img[-tear_depth:, :] = 255
                elif edge == 'left':
                    img[:, :tear_depth] = 255
                elif edge == 'right':
                    img[:, -tear_depth:] = 255

        return img.astype(np.uint8)

    def generate_training_pair(
        self,
        signals: Dict[str, np.ndarray],
        degradation_type: Optional[str] = None,
        severity: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Generate a training pair (image, signals).

        Args:
            signals: Dictionary of 12-lead signals in mV
            degradation_type: Type of degradation (random if None)
            severity: Degradation severity (random if None)

        Returns:
            Tuple of (degraded_image, signals_dict)
        """
        # Generate clean image
        clean_image = self.generate_standard_12lead_layout(signals)

        # Random degradation if not specified
        if degradation_type is None:
            degradation_types = [
                "clean",
                "scanned_color",
                "scanned_bw",
                "photo",
                "stained",
                "mold",
                "damaged",
            ]
            degradation_type = random.choice(degradation_types)

        if severity is None:
            severity = random.uniform(0.3, 0.9)

        # Apply degradation
        degraded_image = self.apply_degradations(clean_image, degradation_type, severity)

        return degraded_image, signals


def generate_synthetic_dataset(
    signal_database_path: str,
    output_path: str,
    n_samples: int = 1000,
    image_size: Tuple[int, int] = (2200, 1700),
):
    """Generate synthetic ECG dataset from signal database.

    Args:
        signal_database_path: Path to PhysioNet or similar database
        output_path: Output directory for generated images
        n_samples: Number of samples to generate
        image_size: Output image size
    """
    import os
    from tqdm import tqdm

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Initialize generator
    generator = ECGImageGenerator(image_size=image_size)

    print(f"Generating {n_samples} synthetic ECG images...")

    for i in tqdm(range(n_samples)):
        # TODO: Load real signals from database
        # For now, generate random signals for demonstration
        signals = {}
        for lead in ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]:
            # Generate 10 seconds of synthetic signal
            n_samples_signal = 10 * 500  # 10 seconds at 500 Hz
            # Simple synthetic ECG (replace with real data)
            t = np.linspace(0, 10, n_samples_signal)
            signal = 0.5 * np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(n_samples_signal)
            signals[lead] = signal

        # Generate training pair
        image, signals_dict = generator.generate_training_pair(signals)

        # Save image
        image_path = os.path.join(output_path, f"ecg_{i:06d}.png")
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Save signals as CSV
        signal_path = os.path.join(output_path, f"ecg_{i:06d}.csv")
        # TODO: Save signals to CSV

    print(f"✅ Generated {n_samples} samples in {output_path}")


if __name__ == "__main__":
    print("Synthetic ECG Image Generator ready!")
    print("\nExample usage:")
    print("""
    from src.strategies.synthetic_data_generator import ECGImageGenerator

    # Create generator
    generator = ECGImageGenerator(image_size=(2200, 1700))

    # Generate signals (example - replace with real data)
    signals = {
        'I': np.random.randn(5000) * 0.5,
        'II': np.random.randn(5000) * 0.5,
        # ... other leads
    }

    # Generate image
    image, signals = generator.generate_training_pair(signals, degradation_type='photo')

    # Save
    cv2.imwrite('synthetic_ecg.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    """)
