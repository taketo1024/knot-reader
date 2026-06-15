"""Stages 2-5: a cropped knot diagram -> PD code.

Pipeline:
  skeletonize -> ordered arcs (the solid/over strands, broken at under-passages)
  -> bridge each broken end to its continuation by velocity continuity
  -> crossings = bridge x arc intersections (bridge under, arc over)
  -> traverse the single closed curve -> edges (bottommost = 1, CCW)
  -> PD code X[a,b,c,d] in KnotTheory convention.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import cv2
import numpy as np
from skimage.morphology import skeletonize


# ---------- skeleton -> ordered arcs ----------

def _neighbors(sk, y, x):
    H, W = sk.shape
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W and sk[yy, xx]:
                out.append((yy, xx))
    return out


def skeleton_arcs(binary, min_area=400):
    """Drop small components (dashes), skeletonize, order each into an (x,y) polyline."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    strand = np.zeros_like(binary)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            strand[lab == i] = 1
    sk = skeletonize(strand > 0).astype(np.uint8)

    K = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
    deg = cv2.filter2D(sk, -1, K, borderType=cv2.BORDER_CONSTANT)
    ncc, sklab = cv2.connectedComponents(sk, 8)

    arcs = []
    for c in range(1, ncc):
        ys, xs = np.where(sklab == c)
        comp = set(zip(ys.tolist(), xs.tolist()))
        deg1 = [p for p in comp if deg[p] == 1]
        start = deg1[0] if deg1 else next(iter(comp))
        path, prev, cur, seen = [start], None, start, {start}
        while True:
            nbs = [q for q in _neighbors(sk, *cur) if q in comp and q != prev and q not in seen]
            if not nbs:
                break
            prev, cur = cur, nbs[0]
            path.append(cur)
            seen.add(cur)
        if len(path) >= 3:
            arcs.append(np.array([[x, y] for y, x in path], float))
    return arcs


# ---------- bridges (velocity-continuity pairing) ----------

def _end_velocity(poly, at_start, win=22):
    pts = poly[:win] if at_start else poly[-win:][::-1]
    d = pts[0] - pts.mean(0)
    nrm = np.linalg.norm(d)
    return d / nrm if nrm else d


def pair_bridges(arcs, max_gap=150.0, min_cos=0.5):
    """Pair arc endpoints whose velocities continue into each other across a gap."""
    ends = []  # (pt, vel, arc_idx, which-end)
    for ai, poly in enumerate(arcs):
        ends.append((poly[0], _end_velocity(poly, True), ai, 0))
        ends.append((poly[-1], _end_velocity(poly, False), ai, 1))

    def score(a, b):
        d = b[0] - a[0]
        dist = np.linalg.norm(d)
        if dist < 5 or dist > max_gap:
            return -1
        u = d / dist
        if a[1] @ u < min_cos or b[1] @ (-u) < min_cos:
            return -1
        return float(a[1] @ u + b[1] @ (-u))

    cand = []
    for i in range(len(ends)):
        for j in range(i + 1, len(ends)):
            s = score(ends[i], ends[j])
            if s > 0:
                cand.append((s, i, j))
    cand.sort(reverse=True)

    used, bridges = set(), []
    for _, i, j in cand:
        if i in used or j in used:
            continue
        used |= {i, j}
        bridges.append((ends[i][0], ends[j][0],
                        (ends[i][2], ends[i][3]), (ends[j][2], ends[j][3])))
    return bridges


# ---------- crossings = bridge x arc intersections ----------

def _seg_int(p1, p2, p3, p4):
    r, s = p2 - p1, p4 - p3
    rxs = r[0] * s[1] - r[1] * s[0]
    if abs(rxs) < 1e-9:
        return None
    qp = p3 - p1
    t = (qp[0] * s[1] - qp[1] * s[0]) / rxs
    u = (qp[0] * r[1] - qp[1] * r[0]) / rxs
    if 0 <= t <= 1 and 0 <= u <= 1:
        return t, u, p1 + t * r
    return None


def find_crossings(arcs, bridges, excl=12.0, merge=15.0):
    """Each bridge-arc intersection is a crossing (bridge=under, arc=over).

    A straight bridge can clip the same arc polyline at a shared vertex and be
    counted twice; intersections of the *same* bridge+arc within `merge` px are
    one crossing, so we keep only the first of each such cluster.
    """
    crossings = []
    for bi, (b0, b1, (aA, _), (aB, _)) in enumerate(bridges):
        for ai, poly in enumerate(arcs):
            if ai == aA or ai == aB:
                continue
            hits = []
            for k in range(len(poly) - 1):
                r = _seg_int(b0, b1, poly[k], poly[k + 1])
                if r is None:
                    continue
                t, u, ip = r
                if np.linalg.norm(ip - b0) < excl or np.linalg.norm(ip - b1) < excl:
                    continue
                if any(np.linalg.norm(ip - h['pt']) < merge for h in hits):
                    continue
                hits.append(dict(pt=ip, bridge=bi, tb=t, arc=ai, sa=k + u))
            crossings.extend(hits)
    return crossings


# ---------- traversal: closed curve -> ordered encounters + edges ----------

def traverse(arcs, bridges):
    """Walk the alternating arc/bridge cycle; return ordered steps."""
    adj = {}
    for bi, (_, _, nA, nB) in enumerate(bridges):
        adj[nA] = (nB, bi, 0)
        adj[nB] = (nA, bi, 1)

    steps, start, cur, guard = [], (0, 0), (0, 0), 0
    while True:
        a, e = cur
        steps.append(('arc', a, e == 0))
        partner, bi, side = adj[(a, 1 - e)]
        steps.append(('bridge', bi, side == 0))
        cur = partner
        guard += 1
        if cur == start or guard > 5 * len(arcs):
            break
    return steps


def build_encounters(arcs, bridges, crossings, steps):
    """Concatenate the curve in traversal order; return master points + ordered encounters."""
    by_arc, by_bridge = defaultdict(list), defaultdict(list)
    for ci, c in enumerate(crossings):
        by_arc[c['arc']].append(ci)
        by_bridge[c['bridge']].append(ci)

    P, enc = [], []  # enc: (pos_in_P, crossing_idx, role)
    for kind, idx, asc in steps:
        base = len(P)
        if kind == 'arc':
            poly = arcs[idx] if asc else arcs[idx][::-1]
            P.extend(map(tuple, poly))
            for ci in sorted(by_arc[idx], key=lambda c: crossings[c]['sa'], reverse=not asc):
                sa = crossings[ci]['sa']
                pos = sa if asc else (len(arcs[idx]) - 1 - sa)
                enc.append((base + int(round(pos)), ci, 'over'))
        else:
            b0, b1, _, _ = bridges[idx]
            seg = np.linspace(b0 if asc else b1, b1 if asc else b0, 8)
            P.extend(map(tuple, seg))
            for ci in sorted(by_bridge[idx], key=lambda c: crossings[c]['tb'], reverse=not asc):
                tb = crossings[ci]['tb']
                tt = tb if asc else 1 - tb
                enc.append((base + int(round(tt * 7)), ci, 'under'))

    enc.sort(key=lambda e: e[0])
    return np.array(P, float), enc


def label_edges(P, enc):
    """Edges sit between consecutive encounters; number them bottommost=1, CCW."""
    m = len(enc)
    pos = [e[0] for e in enc]

    # orient CCW on the page: shoelace with y flipped (y-up); positive => CCW
    x, y = P[:, 0], -P[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    order = enc if area > 0 else enc[::-1]

    # start at the edge containing the bottommost (max-y) point
    bottom = int(np.argmax(P[:, 1]))
    start = 0
    for j in range(m):
        a, b = order[j][0], order[(j + 1) % m][0]
        if (a <= bottom <= b) or (b < a and (bottom >= a or bottom <= b)):
            start = j
            break

    edge_label = {(start + k) % m: k + 1 for k in range(m)}
    return order, edge_label, m


# ---------- PD assembly (KnotTheory convention) ----------

def _arc_tangent(poly, sa):
    k = int(np.clip(sa, 1, len(poly) - 2))
    t = poly[k + 1] - poly[k - 1]
    nrm = np.linalg.norm(t)
    return t / nrm if nrm else t


def assemble_pd(arcs, bridges, crossings, steps, order, edge_label, m):
    """X[a,b,c,d]: a=incoming under, b,c,d counterclockwise (under-strand a->c)."""
    arc_fwd, bri_asc = {}, {}
    for kind, idx, d in steps:
        (arc_fwd if kind == 'arc' else bri_asc)[idx] = d

    for c in crossings:
        b0, b1, _, _ = bridges[c['bridge']]
        vb = b1 - b0
        vb = vb / np.linalg.norm(vb)
        c['v_under'] = vb if bri_asc[c['bridge']] else -vb
        va = _arc_tangent(arcs[c['arc']], c['sa'])
        c['v_over'] = va if arc_fwd[c['arc']] else -va

    enc_pos = {}
    for j, (_, ci, role) in enumerate(order):
        enc_pos.setdefault(ci, {})[role] = j

    e_in = lambda j: edge_label[(j - 1) % m]
    e_out = lambda j: edge_label[j % m]
    ang = lambda v: np.arctan2(-v[1], v[0])  # y-up

    PD = []
    for ci, c in enumerate(crossings):
        ju, jo = enc_pos[ci]['under'], enc_pos[ci]['over']
        a, cc = e_in(ju), e_out(ju)
        stubs = [(a, -c['v_under']), (cc, c['v_under']),
                 (e_in(jo), -c['v_over']), (e_out(jo), c['v_over'])]
        a0 = ang(stubs[0][1])
        stubs.sort(key=lambda s: (ang(s[1]) - a0) % (2 * np.pi))
        PD.append([s[0] for s in stubs])
    return PD


def validate_pd(PD):
    """Every edge label must appear exactly twice."""
    counts = Counter(e for q in PD for e in q)
    return all(v == 2 for v in counts.values()) and len(counts) == 2 * len(PD)


def diagram_to_pd(binary):
    """Full stages 2-5 for one cropped diagram (binary, 255=ink) -> PD code."""
    arcs = skeleton_arcs(binary)
    bridges = pair_bridges(arcs)
    crossings = find_crossings(arcs, bridges)
    steps = traverse(arcs, bridges)
    P, enc = build_encounters(arcs, bridges, crossings, steps)
    order, edge_label, m = label_edges(P, enc)
    PD = assemble_pd(arcs, bridges, crossings, steps, order, edge_label, m)
    return PD, dict(arcs=arcs, bridges=bridges, crossings=crossings, P=P,
                    order=order, edge_label=edge_label, m=m)
