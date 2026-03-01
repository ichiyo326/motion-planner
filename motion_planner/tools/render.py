#!/usr/bin/env python3
"""
render.py — Visualise a planned trajectory as an animated GIF.

Usage:
    python tools/render.py --csv out/trajectory.csv \
                           --scene scene.json \
                           --output out/plan.gif \
                           --fps 15
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.animation as animation

# ─── DH parameters (must match fk.cpp) ──────────────────────────────────────
DH = [
    (0.000,  0.333,  0.0       ),
    (0.000,  0.000, -math.pi/2 ),
    (0.000,  0.316,  math.pi/2 ),
    (0.0825, 0.000,  math.pi/2 ),
    (-0.0825,0.384, -math.pi/2 ),
    (0.000,  0.000,  math.pi/2 ),
    (0.088,  0.000,  math.pi/2 ),
]

def dh_transform(a, d, alpha, theta):
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    R = np.array([
        [ct, -st*ca,  st*sa],
        [st,  ct*ca, -ct*sa],
        [0,   sa,     ca   ],
    ])
    t = np.array([a*ct, a*st, d])
    return R, t

def compute_fk(q):
    """Return 8 joint positions (world frame)."""
    positions = [np.zeros(3)]
    orientations = [np.eye(3)]
    for i, (a, d, alpha) in enumerate(DH):
        R_local, t_local = dh_transform(a, d, alpha, q[i])
        R_world = orientations[-1] @ R_local
        p_world = positions[-1] + orientations[-1] @ t_local
        orientations.append(R_world)
        positions.append(p_world)
    return positions  # list of 8 np.array(3)

# ─── Geometry helpers ────────────────────────────────────────────────────────

def draw_capsule(ax, p, q, r, color, alpha=0.6):
    """Draw a capsule as a cylinder + two sphere endpoints."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    axis = q - p
    length = np.linalg.norm(axis)
    if length < 1e-6:
        u = np.array([r, 0, 0])
        ax.scatter(*p, s=50, c=[color], alpha=alpha)
        return

    axis_n = axis / length
    # Find a perpendicular
    perp = np.array([1, 0, 0]) if abs(axis_n[0]) < 0.9 else np.array([0, 1, 0])
    perp = np.cross(axis_n, perp)
    perp /= np.linalg.norm(perp)
    perp2 = np.cross(axis_n, perp)

    N = 12
    theta = np.linspace(0, 2*math.pi, N, endpoint=False)
    ring = np.array([math.cos(t)*perp + math.sin(t)*perp2 for t in theta])

    # Cylinder faces
    for i in range(N):
        j = (i+1) % N
        verts = [
            p + r*ring[i], p + r*ring[j],
            q + r*ring[j], q + r*ring[i],
        ]
        poly = Poly3DCollection([verts], alpha=alpha)
        poly.set_facecolor(color)
        poly.set_edgecolor('none')
        ax.add_collection3d(poly)

    # Caps (simple circle patches approximation)
    for centre in [p, q]:
        cap_verts = [centre + r*ring[i] for i in range(N)]
        cap_poly = Poly3DCollection([cap_verts], alpha=alpha)
        cap_poly.set_facecolor(color)
        cap_poly.set_edgecolor('none')
        ax.add_collection3d(cap_poly)

def draw_sphere(ax, centre, radius, color, alpha=0.5):
    u = np.linspace(0, 2*math.pi, 16)
    v = np.linspace(0, math.pi, 10)
    x = centre[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = centre[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = centre[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

def draw_box(ax, pos, quat_wxyz, half_extents, color, alpha=0.4):
    """Draw an OBB as a wireframe box."""
    w, x, y, z = quat_wxyz
    # Rotation matrix from quaternion (wxyz active)
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    h = half_extents
    corners_local = np.array([
        [-h[0], -h[1], -h[2]], [ h[0], -h[1], -h[2]],
        [ h[0],  h[1], -h[2]], [-h[0],  h[1], -h[2]],
        [-h[0], -h[1],  h[2]], [ h[0], -h[1],  h[2]],
        [ h[0],  h[1],  h[2]], [-h[0],  h[1],  h[2]],
    ])
    corners = (R @ corners_local.T).T + np.array(pos)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        ax.plot3D(*zip(corners[a], corners[b]), color=color, alpha=0.8, lw=1.5)

def draw_capsule_obstacle(ax, pos, quat_wxyz, radius, half_length, color, alpha=0.5):
    w, x, y, z = quat_wxyz
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    axis = R @ np.array([0, 0, 1])
    p = np.array(pos) - half_length * axis
    q = np.array(pos) + half_length * axis
    draw_capsule(ax, p, q, radius, color, alpha)

# ─── Colour from min_sd ──────────────────────────────────────────────────────

def sd_color(sd, d_safe):
    if sd < 0:       return 'red'
    if sd < d_safe:  return 'orange'
    return 'limegreen'

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',    required=True)
    parser.add_argument('--scene',  required=True)
    parser.add_argument('--output', default='out/plan.gif')
    parser.add_argument('--fps',    type=int, default=10)
    parser.add_argument('--dpi',    type=int, default=90)
    args = parser.parse_args()

    # Load
    waypoints = np.loadtxt(args.csv, delimiter=',', skiprows=1)
    if waypoints.ndim == 1:
        waypoints = waypoints[np.newaxis, :]

    with open(args.scene) as f:
        scene = json.load(f)

    obstacles = scene.get('obstacles', [])
    link_caps = scene['robot']['link_capsules']
    d_safe    = scene['planning']['d_safe']

    # Compute FK for all waypoints
    all_fk = [compute_fk(q) for q in waypoints]

    # Compute min_sd (approximate — just use min joint-to-obstacle)
    # For rendering purposes we use a simplified sd estimate
    def approx_min_sd(fk_positions, caps, obs_list):
        min_sd = 1e9
        for lc in caps:
            P = np.array(fk_positions[lc['joint_i']])
            Q = np.array(fk_positions[lc['joint_j']])
            r = lc['radius']
            for obs in obs_list:
                if obs['type'] == 'sphere':
                    c = np.array(obs['pos'])
                    # dist from sphere centre to segment
                    seg = Q - P
                    t = np.clip(np.dot(c - P, seg) / (np.dot(seg,seg)+1e-12), 0, 1)
                    closest = P + t*seg
                    dist = np.linalg.norm(c - closest)
                    sd = dist - r - obs['radius']
                    if sd < min_sd: min_sd = sd
        return min_sd

    all_min_sd = [approx_min_sd(fk, link_caps, obstacles) for fk in all_fk]

    # Animation
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        ax.cla()
        ax.set_xlim(-0.5, 1.0); ax.set_ylim(-0.8, 0.8); ax.set_zlim(0, 1.2)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')

        fk  = all_fk[frame_idx]
        msd = all_min_sd[frame_idx]

        # ── Draw obstacles ───────────────────────────────────────────────
        for obs in obstacles:
            if obs['type'] == 'sphere':
                draw_sphere(ax, obs['pos'], obs['radius'], color='steelblue')
            elif obs['type'] == 'box':
                draw_box(ax, obs['pos'], obs.get('quat',[1,0,0,0]),
                         obs['half_extents'], color='slategray')
            elif obs['type'] == 'capsule':
                draw_capsule_obstacle(ax, obs['pos'], obs.get('quat',[1,0,0,0]),
                                      obs['radius'], obs['half_length'],
                                      color='steelblue')

        # ── Draw robot capsules ──────────────────────────────────────────
        rob_color = sd_color(msd, d_safe)
        for lc in link_caps:
            P = np.array(fk[lc['joint_i']])
            Q = np.array(fk[lc['joint_j']])
            draw_capsule(ax, P, Q, lc['radius'], rob_color, alpha=0.75)
            ax.plot3D(*zip(P, Q), color='black', lw=1.5, alpha=0.9)

        # ── Joint dots ───────────────────────────────────────────────────
        pts = np.array(fk)
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c='k', s=20, zorder=5)

        # ── Status text ──────────────────────────────────────────────────
        status = ("COLLISION" if msd < 0
                  else "NEAR" if msd < d_safe
                  else "SAFE")
        ax.set_title(
            f"Frame {frame_idx+1}/{len(waypoints)} | "
            f"min_sd={msd:.4f} m | {status}",
            fontsize=10
        )

        # Legend
        patches = [
            mpatches.Patch(color='limegreen', label='safe (sd ≥ d_safe)'),
            mpatches.Patch(color='orange',    label='near (0 ≤ sd < d_safe)'),
            mpatches.Patch(color='red',       label='collision (sd < 0)'),
            mpatches.Patch(color='steelblue', label='obstacle'),
        ]
        ax.legend(handles=patches, loc='upper right', fontsize=7)

    out_dir = Path(args.output).parent
    frames_dir = out_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"[render] Generating {len(waypoints)} frames...")
    frame_paths = []
    for i in range(len(waypoints)):
        update(i)
        fp = frames_dir / f"frame_{i:04d}.png"
        fig.savefig(fp, dpi=args.dpi, bbox_inches='tight')
        frame_paths.append(fp)
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(waypoints)}")

    plt.close(fig)

    # Assemble GIF
    try:
        from PIL import Image
        imgs = [Image.open(fp) for fp in frame_paths]
        ms = int(1000 / args.fps)
        imgs[0].save(args.output, save_all=True, append_images=imgs[1:],
                     loop=0, duration=ms)
        print(f"[render] GIF saved → {args.output}")
    except ImportError:
        print("[render] Pillow not found — trying ffmpeg fallback...")
        import subprocess
        pattern = str(frames_dir / 'frame_%04d.png')
        subprocess.run([
            'ffmpeg', '-y', '-framerate', str(args.fps),
            '-i', pattern,
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            args.output
        ], check=True)
        print(f"[render] GIF saved → {args.output}")


if __name__ == '__main__':
    main()
