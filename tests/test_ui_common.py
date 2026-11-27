"""
SAFELABS Visual Test Suite Common UI Engine
Provides dark utilitarian rendering routines, charts, gauges, and window loop management.
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Dict, Any, Optional

# Color palette: Utilitarian matte zinc palette (BGR format for OpenCV)
BG = (11, 9, 9)            # #09090b
SURFACE = (21, 18, 18)     # #121215
SURFACE_CARD = (28, 25, 25)# #19191c
BORDER = (42, 39, 39)      # #27272a
BORDER_LIGHT = (60, 56, 56)# #38383c
TEXT = (245, 244, 244)     # #f4f4f5
MUTED = (170, 161, 161)    # #a1a1aa
GREEN = (94, 197, 34)      # #22c55e
RED = (68, 68, 239)        # #ef4444
AMBER = (11, 158, 245)     # #f59e0b
BLUE = (248, 189, 56)      # #38bdf8
PURPLE = (235, 120, 168)   # #a855f7
WHITE = (255, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

def create_base_canvas(width: int = 1120, height: int = 680) -> np.ndarray:
    """Creates a base dark canvas with uniform background."""
    canvas = np.full((height, width, 3), BG, dtype=np.uint8)
    return canvas

def draw_header(canvas: np.ndarray, title: str, category: str, test_id: str, status: str = "RUNNING"):
    """Draws utilitarian header banner with test metadata."""
    h, w = canvas.shape[:2]
    # Header bar background
    cv2.rectangle(canvas, (0, 0), (w, 64), SURFACE, -1)
    cv2.line(canvas, (0, 64), (w, 64), BORDER, 1)

    # Category badge
    cv2.rectangle(canvas, (24, 16), (24 + 140, 48), SURFACE_CARD, -1)
    cv2.rectangle(canvas, (24, 16), (24 + 140, 48), BORDER_LIGHT, 1)
    cv2.putText(canvas, category.upper(), (34, 37), FONT, 0.42, MUTED, 1, cv2.LINE_AA)

    # Title
    cv2.putText(canvas, title, (180, 40), FONT_BOLD, 0.65, TEXT, 1, cv2.LINE_AA)

    # Test ID & timestamp
    cv2.putText(canvas, f"ID: {test_id} | TIME: {time.strftime('%H:%M:%S')}", (w - 380, 40), FONT, 0.44, MUTED, 1, cv2.LINE_AA)

    # Status Pill
    pill_w = 110
    pill_x = w - 134
    pill_color = GREEN if status == "PASSED" else (AMBER if status == "RUNNING" else RED)
    bg_tint = (30, 50, 30) if status == "PASSED" else ((20, 40, 60) if status == "RUNNING" else (30, 20, 60))
    cv2.rectangle(canvas, (pill_x, 18), (pill_x + pill_w, 46), bg_tint, -1)
    cv2.rectangle(canvas, (pill_x, 18), (pill_x + pill_w, 46), pill_color, 1)
    cv2.putText(canvas, status, (pill_x + 16, 37), FONT_BOLD, 0.46, pill_color, 1, cv2.LINE_AA)

def draw_card(canvas: np.ndarray, x: int, y: int, w: int, h: int, title: Optional[str] = None):
    """Draws a standard zinc card with subtle border and optional title."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), SURFACE, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), BORDER, 1)
    if title:
        cv2.rectangle(canvas, (x, y), (x + w, y + 28), SURFACE_CARD, -1)
        cv2.line(canvas, (x, y + 28), (x + w, y + 28), BORDER, 1)
        cv2.putText(canvas, title.upper(), (x + 14, y + 19), FONT, 0.42, MUTED, 1, cv2.LINE_AA)

def draw_metric_pill(canvas: np.ndarray, x: int, y: int, label: str, value: str, status_color: Tuple[int, int, int] = TEXT):
    """Draws a compact metric key-value readout."""
    cv2.putText(canvas, label.upper(), (x, y), FONT, 0.38, MUTED, 1, cv2.LINE_AA)
    cv2.putText(canvas, value, (x, y + 20), FONT_BOLD, 0.58, status_color, 1, cv2.LINE_AA)

def draw_progress_bar(canvas: np.ndarray, x: int, y: int, w: int, h: int, fraction: float, label: str = ""):
    """Draws a horizontal progress bar."""
    fraction = max(0.0, min(1.0, fraction))
    cv2.rectangle(canvas, (x, y), (x + w, y + h), SURFACE_CARD, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), BORDER, 1)
    fill_w = int(w * fraction)
    if fill_w > 0:
        cv2.rectangle(canvas, (x + 1, y + 1), (x + fill_w - 1, y + h - 1), GREEN, -1)
    if label:
        txt = f"{label} ({int(fraction * 100)}%)"
        cv2.putText(canvas, txt, (x + w + 14, y + int(h * 0.75)), FONT, 0.42, MUTED, 1, cv2.LINE_AA)

def draw_line_chart(canvas: np.ndarray, x: int, y: int, w: int, h: int,
                    data_points: List[float], min_val: float, max_val: float,
                    title: str, unit: str = "ms", threshold: Optional[float] = None,
                    line_color: Tuple[int, int, int] = BLUE):
    """Renders a real-time line chart inside given bounding box."""
    draw_card(canvas, x, y, w, h, title)
    plot_x = x + 50
    plot_y = y + 36
    plot_w = w - 65
    plot_h = h - 52

    # Draw chart background and grid lines
    cv2.rectangle(canvas, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (16, 14, 14), -1)
    cv2.rectangle(canvas, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), BORDER, 1)

    # 3 horizontal gridlines
    for i in range(4):
        gy = plot_y + int(plot_h * (i / 3.0))
        val = max_val - (i / 3.0) * (max_val - min_val)
        cv2.line(canvas, (plot_x, gy), (plot_x + plot_w, gy), (26, 24, 24), 1)
        cv2.putText(canvas, f"{val:.1f}", (x + 6, gy + 4), FONT, 0.35, MUTED, 1, cv2.LINE_AA)

    # Threshold line if specified
    if threshold is not None and min_val <= threshold <= max_val:
        ty = plot_y + int(plot_h * (1.0 - (threshold - min_val) / (max_val - min_val + 1e-6)))
        cv2.line(canvas, (plot_x, ty), (plot_x + plot_w, ty), RED, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"LIMIT {threshold:.0f}{unit}", (plot_x + plot_w - 90, ty - 4), FONT, 0.36, RED, 1, cv2.LINE_AA)

    # Plot data points
    if len(data_points) >= 2:
        pts = []
        n = len(data_points)
        for idx, val in enumerate(data_points):
            px = plot_x + int(plot_w * (idx / max(1, n - 1)))
            norm = (val - min_val) / (max_val - min_val + 1e-6)
            norm = max(0.0, min(1.0, norm))
            py = plot_y + int(plot_h * (1.0 - norm))
            pts.append((px, py))

        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i + 1], line_color, 2, cv2.LINE_AA)
        cv2.circle(canvas, pts[-1], 3, WHITE, -1)

    # Current value display
    current_val = data_points[-1] if data_points else 0.0
    cv2.putText(canvas, f"{current_val:.2f} {unit}", (x + w - 100, y + 20), FONT_BOLD, 0.44, line_color, 1, cv2.LINE_AA)

def draw_bar_chart(canvas: np.ndarray, x: int, y: int, w: int, h: int,
                   labels: List[str], values: List[float], max_val: float,
                   unit: str = "ms", colors: Optional[List[Tuple[int, int, int]]] = None):
    """Draws horizontal bars for comparing components."""
    draw_card(canvas, x, y, w, h, "Stage Latency Decomposition")
    if not labels:
        return
    bar_y_start = y + 42
    bar_h = 20
    spacing = 34
    bar_max_w = w - 180

    for idx, (label, val) in enumerate(zip(labels, values)):
        curr_y = bar_y_start + idx * spacing
        color = colors[idx % len(colors)] if colors else BLUE
        cv2.putText(canvas, label, (x + 14, curr_y + 14), FONT, 0.42, TEXT, 1, cv2.LINE_AA)

        # Bar background
        cv2.rectangle(canvas, (x + 150, curr_y), (x + 150 + bar_max_w, curr_y + bar_h), SURFACE_CARD, -1)
        # Bar fill
        fill_w = int(bar_max_w * max(0.0, min(1.0, val / (max_val + 1e-6))))
        if fill_w > 0:
            cv2.rectangle(canvas, (x + 150, curr_y), (x + 150 + fill_w, curr_y + bar_h), color, -1)
        # Value text
        cv2.putText(canvas, f"{val:.2f} {unit}", (x + 158 + bar_max_w, curr_y + 15), FONT, 0.40, MUTED, 1, cv2.LINE_AA)

def draw_log_console(canvas: np.ndarray, x: int, y: int, w: int, h: int,
                     logs: List[str], title: str = "Live Execution Trace"):
    """Draws a scrolling terminal style log box."""
    draw_card(canvas, x, y, w, h, title)
    inner_x = x + 12
    inner_y = y + 44
    line_h = 18
    max_visible = max(1, (h - 50) // line_h)
    visible_logs = logs[-max_visible:]

    for idx, log in enumerate(visible_logs):
        log_color = MUTED
        if "[PASS]" in log or "PASS" in log:
            log_color = GREEN
        elif "[FAIL]" in log or "FAIL" in log or "ERROR" in log:
            log_color = RED
        elif "[WARN]" in log or "RECONNECT" in log:
            log_color = AMBER
        elif "[INFO]" in log or "-->" in log:
            log_color = BLUE
        cv2.putText(canvas, log, (inner_x, inner_y + idx * line_h), FONT, 0.38, log_color, 1, cv2.LINE_AA)

def draw_footer_prompt(canvas: np.ndarray, message: str, is_complete: bool = False):
    """Draws footer prompt line."""
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, h - 36), (w, h), SURFACE, -1)
    cv2.line(canvas, (0, h - 36), (w, h - 36), BORDER, 1)

    color = GREEN if is_complete else MUTED
    cv2.putText(canvas, message, (24, h - 13), FONT, 0.42, color, 1, cv2.LINE_AA)
    if is_complete:
        cv2.putText(canvas, "[ESC / ANY KEY TO EXIT]", (w - 220, h - 13), FONT_BOLD, 0.42, MUTED, 1, cv2.LINE_AA)

def render_and_wait(window_name: str, canvas: np.ndarray, delay_ms: int = 1) -> int:
    """Updates window display and returns key code."""
    cv2.imshow(window_name, canvas)
    return cv2.waitKey(delay_ms) & 0xFF

