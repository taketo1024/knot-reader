"""Stage 6: verification overlay.

Draws the algorithm's interpretation back on the image so it can be checked:
  - reconstructed under-passages (bridges) in orange  -> shows over/under
  - crossings circled in red
  - edge labels (bottommost=1, CCW) in blue at edge midpoints
"""

import cv2
import numpy as np


def draw_overlay(gray, info):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    P, order, edge_label, m = info['P'], info['order'], info['edge_label'], info['m']

    # under-passages (bridges) — these are the strands the algorithm puts UNDER
    for b0, b1, _, _ in info['bridges']:
        cv2.line(vis, tuple(b0.astype(int)), tuple(b1.astype(int)), (0, 140, 255), 1)

    # crossings
    for c in info['crossings']:
        cv2.circle(vis, tuple(c['pt'].astype(int)), 12, (0, 0, 255), 2)

    # edge labels at edge midpoints
    for j in range(m):
        a = order[j][0]
        b = order[(j + 1) % m][0]
        mid = (a + (b if b >= a else b + len(P))) // 2 % len(P)
        px, py = P[mid]
        cv2.putText(vis, str(edge_label[j]), (int(px) + 3, int(py)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 0, 0), 1, cv2.LINE_AA)
    return vis
