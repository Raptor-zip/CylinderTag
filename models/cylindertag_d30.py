"""Flat, single-nozzle CylinderTag ID 0; dimensions in mm, bed at Z=0.

The continuous polygon construction follows CylinderTag_generator.m:draw.
Edit the named parameters and regenerate with the CAD skill's step launcher.
"""
from math import pi, sqrt
from pathlib import Path

from build123d import Align, Box, Color, Polygon, Pos, extrude

CORE_DIAMETER = 30.0
BASE_THICKNESS = 0.2
PATTERN_THICKNESS = 0.2
PATTERN_HEIGHT = 40.0
AXIAL_MARGIN = 2.0
MARKER_ID = 0
WHITE_RATIO = 0.2
# Approximate bending neutral plane at the middle of the white substrate.
# Raised pattern, adhesion, and real bending change this; measure after mounting.
STRIP_LENGTH = pi * (CORE_DIAMETER + BASE_THICKNESS)
STRIP_HEIGHT = PATTERN_HEIGHT + 2 * AXIAL_MARGIN
DICTIONARY = Path(__file__).resolve().parents[1] / "CTag_2f12c.marker"


def marker_codes():
    lines = DICTIONARY.read_text().splitlines()
    count, columns, features = map(int, lines[0].split())
    assert 0 <= MARKER_ID < count and features == 2
    codes = list(map(int, lines[MARKER_ID + 1].split()))
    assert len(codes) == columns and all(0 <= code < 64 for code in codes)
    return codes


def boundary(symbol):
    # Positive roots of MATLAB's [-1, H, (.11 - .2*decoder)*H^2].
    decoder = (1.47, 1.54, 1.61, 1.68, 1.68, 1.61, 1.54, 1.47)
    c = WHITE_RATIO / 2 + WHITE_RATIO**2 / 4 - .2 * decoder[symbol]
    root = sqrt(1 + 4 * c)
    return PATTERN_HEIGHT * (1 + (root if symbol >= 4 else -root)) / 2


def black_polygons():
    codes = marker_codes()
    pitch = STRIP_LENGTH / len(codes)
    bar_width = pitch / 1.5
    half_gap = PATTERN_HEIGHT * WHITE_RATIO / 2
    for column, code in enumerate(codes):
        x0, x1 = column * pitch, column * pitch + bar_width
        left, right = boundary(code // 8), boundary(code % 8)
        # Match the MATLAB image orientation: top edge is y=AXIAL_MARGIN.
        m, h = AXIAL_MARGIN, PATTERN_HEIGHT
        yield [(x0, m), (x1, m),
               (x1, m + right - half_gap), (x0, m + left - half_gap)]
        yield [(x0, m + left + half_gap), (x1, m + right + half_gap),
               (x1, m + h), (x0, m + h)]


def gen_step():
    base = Box(STRIP_LENGTH, STRIP_HEIGHT, BASE_THICKNESS,
               align=(Align.MIN, Align.MIN, Align.MIN))
    bars = [Pos(0, 0, BASE_THICKNESS) * extrude(Polygon(*pts, align=None),
            amount=PATTERN_THICKNESS) for pts in black_polygons()]
    tag = base.fuse(*bars).clean()
    tag.label = "CylinderTag ID 0 - D30 flat wrap - white base and raised black pattern"
    tag.color = Color("white")
    assert tag.is_valid and len(tag.solids()) == 1
    bbox = tag.bounding_box().size
    for actual, expected in zip(bbox, (STRIP_LENGTH, STRIP_HEIGHT,
                                      BASE_THICKNESS + PATTERN_THICKNESS)):
        assert abs(actual - expected) < 1e-6, (actual, expected)
    # Verify volume analytically from all 24 quadrilateral profiles.
    area = 0.0
    for pts in black_polygons():
        area += abs(sum(a[0]*b[1] - b[0]*a[1]
                        for a, b in zip(pts, pts[1:] + pts[:1]))) / 2
    expected_volume = STRIP_LENGTH*STRIP_HEIGHT*BASE_THICKNESS + area*PATTERN_THICKNESS
    assert abs(tag.volume - expected_volume) < 1e-5
    return tag


if __name__ == "__main__":
    # Optional true-color, flat reference for comparing a sliced print.
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {STRIP_LENGTH} {STRIP_HEIGHT}" width="950" height="440">',
           f'<rect width="{STRIP_LENGTH}" height="{STRIP_HEIGHT}" fill="white"/>']
    svg += ['<polygon fill="black" points="' + ' '.join(f'{x},{y}' for x,y in pts) + '"/>'
            for pts in black_polygons()]
    Path(__file__).with_suffix('.svg').write_text('\n'.join(svg + ['</svg>']))
