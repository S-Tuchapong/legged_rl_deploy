#!/usr/bin/env python3
"""Single source of truth for the external-payload conditions (series A to E).

The payload is a rigid body welded into the Go2 trunk by
`unitree_robots/go2/payload.xml`, which `go2.xml` includes inside
`<body name="base_link">`.  Every condition is expressed with an explicit
`<inertial>` so that mass, centre of mass, and inertia are three genuinely
independent knobs; the geoms are visual only and never touch the physics.

Usage:
    ./payload_conditions.py list                 # table of every condition
    ./payload_conditions.py show A1              # detail plus the emitted MJCF
    ./payload_conditions.py apply A1             # write payload.xml
    ./payload_conditions.py apply S0             # restore the no-payload baseline
    ./payload_conditions.py manifest -o out.json # machine-readable dump
"""

from __future__ import annotations

import argparse
import json
import sys
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

# scripts/ -> legged_rl_deploy/ -> src/
SRC_DIR = Path(__file__).resolve().parents[2]
GO2_XML = SRC_DIR / "unitree_mujoco" / "unitree_robots" / "go2" / "go2.xml"
PAYLOAD_XML = SRC_DIR / "unitree_mujoco" / "unitree_robots" / "go2" / "payload.xml"
CONTACTS_XML = (SRC_DIR / "unitree_mujoco" / "unitree_robots" / "go2"
                / "payload_contacts.xml")

# Every depth-camera ray leaves the lens travelling forward (dx > 0.264) and
# downward (elevation <= -8.46 deg) from x = +0.32715, so a payload whose
# forward-most point stays behind this plane can never enter the depth image.
CAMERA_SAFE_X_MAX = 0.30
# depth_camera.cc force-disables geomgroup 1 and 3 for the sensor render, and
# mjv_defaultOption() enables groups 0-2, so group 1 is seen by the interactive
# viewer and never by the depth camera.
VIEWER_ONLY_GROUP = 1
# MuJoCo refuses an include file that contains no elements, so every condition
# (the baseline included) emits this massless marker.  Site group 4 is off in
# mjv_defaultOption() and depth_camera.cc zeroes every site group, so it is
# invisible in the viewer and in the depth image alike.

DECK_Z = 0.09            # payload deck height above the base_link origin
BRICK = (0.16, 0.10, 0.06)   # full sizes of the reference compact brick
CUBE = 0.06                  # full size of one dumbbell end mass


def mount_site() -> str:
    """MuJoCo refuses an include file that contains no elements, so every
    condition (the baseline included) emits this massless marker.  Site group 4
    is off in mjv_defaultOption() and depth_camera.cc zeroes every site group,
    so it is invisible in the viewer and in the depth image alike."""
    return f'  <site name="payload_mount" pos="0 0 {DECK_Z:g}" size="0.005" group="4"/>'


def box_diaginertia(mass: float, full: tuple[float, float, float]) -> np.ndarray:
    lx, ly, lz = full
    return np.array([
        mass / 12.0 * (ly * ly + lz * lz),
        mass / 12.0 * (lx * lx + lz * lz),
        mass / 12.0 * (lx * lx + ly * ly),
    ])


PAYLOAD_RGBA = "0.85 0.15 0.15 0.85"
BASKET_RGBA = "0.10 0.85 0.90 1"

# Sliding friction between the load and its basket.  This is the knob that
# decides whether series G differs from series F at all: a load only shifts once
# the trunk's lateral acceleration passes mu*g, so mu = 0.5 (slip above 26.6 deg
# of tilt, 4.9 m/s^2) never breaks loose during a 0.6 m/s beam walk, while
# mu = 0.1 (5.7 deg, 0.98 m/s^2) slides throughout.  Measured, not assumed.
CARGO_FRICTION_GRIPPY = 0.5
CARGO_FRICTION_SLIPPERY = 0.1


@dataclass(frozen=True)
class Geom:
    """A visual-only marker, positioned relative to the payload centre of mass.

    Either a box (`full_size` at `pos`) or, when `fromto` is given, a capsule
    edge of radius `radius` between two points.  Edges are what draw a wire
    basket: MuJoCo has no frustum primitive, and the real article is a frame
    rather than a solid anyway."""
    name: str
    full_size: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fromto: tuple[float, ...] | None = None
    radius: float = 0.004
    rgba: str = PAYLOAD_RGBA
    euler: tuple[float, float, float] | None = None
    collide: bool = False

    def x_max(self) -> float:
        if self.fromto is not None:
            return max(self.fromto[0], self.fromto[3]) + self.radius
        return self.pos[0] + self.full_size[0] / 2.0


@dataclass(frozen=True)
class Payload:
    mass: float
    com: tuple[float, float, float]          # relative to the base_link origin
    diaginertia: tuple[float, float, float]  # about the payload's own centre of mass
    geoms: tuple[Geom, ...]

    def validate(self) -> list[str]:
        warnings: list[str] = []
        if self.mass <= 0.0:
            raise ValueError("payload mass must be positive")
        a, b, c = self.diaginertia
        for x, y, z, label in ((a, b, c, "Ixx+Iyy>=Izz"),
                               (b, c, a, "Iyy+Izz>=Ixx"),
                               (c, a, b, "Izz+Ixx>=Iyy")):
            if x + y < z:
                raise ValueError(
                    f"diaginertia {self.diaginertia} violates {label}; MuJoCo "
                    "rejects inertias that no rigid body can have"
                )
        x_max = max(self.com[0] + g.x_max() for g in self.geoms)
        if x_max > CAMERA_SAFE_X_MAX:
            warnings.append(
                f"geom reaches x={x_max:.3f} m, past the {CAMERA_SAFE_X_MAX} m "
                f"camera-safe plane; harmless only because group "
                f"{VIEWER_ONLY_GROUP} is excluded from the depth render"
            )
        return warnings


@dataclass(frozen=True)
class Cargo:
    """An object that rides in the basket without being fixed to it.

    MuJoCo rejects <freejoint> on a nested body ("free joint can only be used on
    top level"), and a second top-level free joint would trip the episode
    manager's "exactly one free joint" check.  Three slides plus three hinges
    relative to base_link give the same six degrees of freedom, and because they
    are relative coordinates mj_resetData drops the object back into the basket
    at every episode reset, following whatever offset the reset applied."""
    mass: float
    size: tuple[float, float, float]
    rest_com: tuple[float, float, float]   # relative to the base_link origin
    rgba: str = PAYLOAD_RGBA
    friction: float = CARGO_FRICTION_GRIPPY
    # Travel limits, as a hard backstop behind the wall contacts.  A joint range
    # is a position-level constraint the solver enforces every step, so unlike a
    # wall it cannot be tunnelled through at any speed.  Set at the top rim, so
    # the walls still do all the ordinary work and the limit only catches an
    # escape that contact missed.
    travel: tuple[tuple[float, float], ...] = ()

    def diaginertia(self) -> np.ndarray:
        return box_diaginertia(self.mass, self.size)


@dataclass(frozen=True)
class Condition:
    cid: str
    series: str
    title: str
    tests: str
    payload: Payload | None = None
    cargo: Cargo | None = None
    alias_of: str | None = None
    in_default_sweep: bool = True


def brick_payload(mass: float, com, full=BRICK) -> Payload:
    return Payload(
        mass=mass,
        com=tuple(float(v) for v in com),
        diaginertia=tuple(box_diaginertia(mass, full)),
        geoms=(Geom("payload_geom", full),),
    )


def dumbbell_payload(mass: float, separation: float, axis: int, com=(0.0, 0.0, DECK_Z)) -> Payload:
    """Two equal end masses on a massless boom: mass and CoM stay fixed while
    the inertia about the two axes perpendicular to `axis` grows as m*d^2/4."""
    half = mass / 2.0
    own = box_diaginertia(half, (CUBE, CUBE, CUBE)) * 2.0   # both cubes, own spin
    spread = 2.0 * half * (separation / 2.0) ** 2
    inertia = own.copy()
    for k in range(3):
        if k != axis:
            inertia[k] += spread
    offset = np.zeros(3)
    offset[axis] = separation / 2.0
    return Payload(
        mass=mass,
        com=tuple(float(v) for v in com),
        diaginertia=tuple(inertia),
        geoms=(
            Geom("payload_geom_pos", (CUBE, CUBE, CUBE), tuple(+offset)),
            Geom("payload_geom_neg", (CUBE, CUBE, CUBE), tuple(-offset)),
        ),
    )


# The wire basket measured off the real fixture: a rectangular frustum whose
# bottom rests on the trunk.  MuJoCo has no frustum primitive, and a wire basket
# is a frame rather than a solid, so it is drawn as its 12 edges and its inertia
# comes from mass spread along those edges.
BASKET_HEIGHT = 0.15
BASKET_TOP = (0.36, 0.275)
BASKET_BOTTOM = (0.305, 0.235)
BASKET_Z_BOTTOM = 0.06          # trunk collision box tops out at z = 0.057
# MuJoCo has no continuous collision detection, so a wall is only seen if the
# cargo lands inside it on some timestep.  Measured against the 1 ms timestep, a
# 4 mm wall is passed through at 3 m/s, 10 mm at 6 m/s and 20 mm at 10 m/s.
# 16 mm covers the speeds a walking robot produces; the joint ranges below are
# the guarantee for everything faster.
BASKET_WALL_HALF_THICKNESS = 0.008
CARGO_SIZE = (0.25, 0.18, 0.10)          # snug: it nearly fills the basket floor
CARGO_SIZE_LOOSE = (0.15, 0.12, 0.10)    # room to slide before it hits a wall
# The fixture is invisible to automatic collision detection (contype 0) and its
# only contacts come from the explicit pairs written to payload_contacts.xml.
# That is not a shortcut: the basket is welded into base_link, which makes it the
# cargo's parent weld body, and MuJoCo's filterparent flag drops body-versus-
# parent contacts, so automatic detection produces nothing here whatever the
# contype bits say.  Going through pairs also guarantees the overhanging basket
# can never catch on the beam, the floor or the legs.


def frustum_edges(height=BASKET_HEIGHT, top=BASKET_TOP, bottom=BASKET_BOTTOM,
                  z_bottom=BASKET_Z_BOTTOM):
    """The 12 edges of a rectangular frustum, in base_link coordinates."""
    tx, ty = top[0] / 2.0, top[1] / 2.0
    bx, by = bottom[0] / 2.0, bottom[1] / 2.0
    zt = z_bottom + height
    lower = [(+bx, +by, z_bottom), (-bx, +by, z_bottom),
             (-bx, -by, z_bottom), (+bx, -by, z_bottom)]
    upper = [(+tx, +ty, zt), (-tx, +ty, zt), (-tx, -ty, zt), (+tx, -ty, zt)]
    edges = []
    for i in range(4):
        edges.append((lower[i], lower[(i + 1) % 4]))   # bottom rim
        edges.append((upper[i], upper[(i + 1) % 4]))   # top rim
        edges.append((lower[i], upper[i]))             # slanted post
    return edges


def wire_properties(edges, mass, segments=400):
    """Mass, centre of mass and inertia of `mass` spread uniformly along edges."""
    points, weights = [], []
    for start, end in edges:
        a, b = np.array(start), np.array(end)
        length = np.linalg.norm(b - a)
        for k in range(segments):
            points.append(a + (b - a) * (k + 0.5) / segments)
            weights.append(length / segments)
    points = np.array(points)
    weights = np.array(weights) * (mass / np.sum(weights))
    com = (weights[:, None] * points).sum(axis=0) / mass
    inertia = np.zeros((3, 3))
    for m, point in zip(weights, points):
        d = point - com
        inertia += m * (d @ d * np.eye(3) - np.outer(d, d))
    return mass, com, inertia


def merge_parts(parts):
    """Combine (mass, com, inertia-about-own-com) triples into one rigid body.
    The basket is symmetric about both vertical planes and the cargo sits on its
    axis, so the result is diagonal; anything else would need <fullinertia>."""
    mass = sum(m for m, _, _ in parts)
    com = sum(m * np.asarray(c) for m, c, _ in parts) / mass
    inertia = np.zeros((3, 3))
    for m, c, own in parts:
        d = np.asarray(c) - com
        inertia += own + m * (d @ d * np.eye(3) - np.outer(d, d))
    off_diagonal = np.max(np.abs(inertia - np.diag(np.diag(inertia))))
    if off_diagonal > 1e-9 * max(1.0, np.max(np.abs(inertia))):
        raise ValueError(
            f"inertia is not diagonal (largest off-diagonal {off_diagonal:.3g}); "
            "diaginertia cannot represent it")
    return mass, com, np.diag(inertia)


def basket_wall_geoms(height=BASKET_HEIGHT, top=BASKET_TOP, bottom=BASKET_BOTTOM,
                      z_bottom=BASKET_Z_BOTTOM) -> list[Geom]:
    """Floor plus four slanted panels, thin boxes that actually contain a load.

    A panel is drawn at the top rim's width, so its lower corners overhang the
    frustum slightly; that only matters against the cargo, which never reaches
    them."""
    t = BASKET_WALL_HALF_THICKNESS
    bx, by = bottom[0] / 2.0, bottom[1] / 2.0
    tx, ty = top[0] / 2.0, top[1] / 2.0
    mid_z = z_bottom + height / 2.0
    geoms = [Geom("basket_floor", (bottom[0], bottom[1], 2 * t),
                  (0.0, 0.0, z_bottom - t), rgba=BASKET_RGBA, collide=True)]

    # Panels normal to x: local z runs up the slope, local x is the thickness.
    slope_x = math.hypot(tx - bx, height)
    tilt_x = math.atan2(tx - bx, height)
    for sign in (+1.0, -1.0):
        # Offset outward along the panel normal so the inner face stays on the
        # measured frustum: thickening the wall must not shrink the basket.
        nx, nz = math.cos(tilt_x), -math.sin(tilt_x)
        geoms.append(Geom(
            f"basket_wall_x{'p' if sign > 0 else 'm'}",
            (2 * t, top[1], slope_x),
            (sign * ((bx + tx) / 2.0 + t * nx), 0.0, mid_z + sign * sign * t * nz),
            euler=(0.0, sign * tilt_x, 0.0), rgba=BASKET_RGBA, collide=True))

    # Panels normal to y: local x is the length, local y the thickness.
    slope_y = math.hypot(ty - by, height)
    tilt_y = math.atan2(ty - by, height)
    for sign in (+1.0, -1.0):
        ny, nz = math.cos(tilt_y), -math.sin(tilt_y)
        geoms.append(Geom(
            f"basket_wall_y{'p' if sign > 0 else 'm'}",
            (top[0], 2 * t, slope_y),
            (0.0, sign * ((by + ty) / 2.0 + t * ny), mid_z + t * nz),
            euler=(-sign * tilt_y, 0.0, 0.0), rgba=BASKET_RGBA, collide=True))
    return geoms


def cargo_at_rest(mass: float, size=CARGO_SIZE,
                  friction: float = CARGO_FRICTION_GRIPPY) -> Cargo:
    """A load sitting on the basket floor, centred, free to move within the
    basket's own inner volume."""
    reach_x = max(0.0, BASKET_TOP[0] / 2.0 - size[0] / 2.0)
    reach_y = max(0.0, BASKET_TOP[1] / 2.0 - size[1] / 2.0)
    lift = max(0.0, BASKET_HEIGHT - size[2])
    return Cargo(mass=mass, size=size, friction=friction,
                 rest_com=(0.0, 0.0, BASKET_Z_BOTTOM + size[2] / 2.0),
                 travel=((-reach_x, reach_x), (-reach_y, reach_y), (-0.01, lift)))


def walled_basket_payload(wire_mass: float) -> Payload:
    """The basket as a container: wire outline for the eye, thin panels for the
    physics.  Its own mass and inertia still come from the wire frame."""
    edges = frustum_edges()
    mass, com, inertia = wire_properties(edges, wire_mass)
    geoms = [Geom(f"basket_edge_{i}", fromto=tuple(a) + tuple(b), radius=0.004,
                  rgba=BASKET_RGBA)
             for i, (a, b) in enumerate(edges)]
    geoms += basket_wall_geoms()
    shifted = []
    for geom in geoms:
        if geom.fromto is not None:
            ft = np.array(geom.fromto).reshape(2, 3) - com
            shifted.append(replace(geom, fromto=tuple(ft.reshape(-1))))
        else:
            shifted.append(replace(geom, pos=tuple(np.array(geom.pos) - com)))
    return Payload(mass=float(mass), com=tuple(float(v) for v in com),
                   diaginertia=tuple(float(v) for v in np.diag(inertia)),
                   geoms=tuple(shifted))


def basket_payload(wire_mass: float, cargo_mass: float = 0.0,
                   cargo_size=CARGO_SIZE) -> Payload:
    edges = frustum_edges()
    parts = [wire_properties(edges, wire_mass)]
    geoms = [Geom(f"basket_edge_{i}", fromto=tuple(start) + tuple(end),
                  radius=0.004, rgba=BASKET_RGBA)
             for i, (start, end) in enumerate(edges)]
    if cargo_mass > 0.0:
        # Cargo rests on the basket floor, so its centre sits half its own
        # height above the bottom rim.
        cargo_com = np.array([0.0, 0.0, BASKET_Z_BOTTOM + cargo_size[2] / 2.0])
        parts.append((cargo_mass, cargo_com,
                      np.diag(box_diaginertia(cargo_mass, cargo_size))))
        geoms.append(Geom("basket_cargo", cargo_size, tuple(cargo_com)))

    mass, com, diaginertia = merge_parts(parts)
    # Geoms are written relative to the payload centre of mass.
    shifted = []
    for geom in geoms:
        if geom.fromto is not None:
            ft = np.array(geom.fromto).reshape(2, 3) - com
            shifted.append(replace(geom, fromto=tuple(ft.reshape(-1))))
        else:
            shifted.append(replace(geom, pos=tuple(np.array(geom.pos) - com)))
    return Payload(mass=float(mass), com=tuple(float(v) for v in com),
                   diaginertia=tuple(float(v) for v in diaginertia),
                   geoms=tuple(shifted))


AXIS_NAME = {0: "x (pitch+yaw)", 1: "y (roll+yaw)", 2: "z (roll+pitch)"}


def build_conditions() -> dict[str, Condition]:
    items: list[Condition] = [
        Condition("S0", "S", "no payload",
                  "control group; rerun whenever anything else changes",
                  payload=None),
    ]

    # A: mass only.  Compact brick, centred, inertia contribution negligible.
    for mass in (1.0, 2.5, 5.0, 10.0): #1.0, 2.0, 3.0, 5.0
        items.append(Condition(
            f"A{mass:g}", "A", f"{mass:g} kg compact brick on the back",
            "load capacity against mass alone; inertia barely moves",
            brick_payload(mass, (0.0, 0.0, DECK_Z))))

    # B: roll inertia at a fixed 2 kg.  Identical mass and CoM to A2 throughout.
    for sep in (0.0, 0.2, 0.4, 0.6, 0.8):
        items.append(Condition(
            f"B{sep:g}", "B", f"2 kg lateral dumbbell, d={sep:g} m",
            "roll inertia isolated from mass; mass and CoM are constant across B",
            dumbbell_payload(2.0, sep, axis=1)))

    # C: which axis the policy actually cares about.
    items.append(Condition(
        "Cx", "C", f"2 kg dumbbell along {AXIS_NAME[0]}, d=0.6 m",
        "same added moment as B0.6 but on pitch/yaw instead of roll",
        dumbbell_payload(2.0, 0.6, axis=0)))
    items.append(Condition(
        "Cy", "C", "alias of B0.6 (lateral dumbbell, d=0.6 m)",
        "kept for symmetry of the C comparison; do not run it twice",
        dumbbell_payload(2.0, 0.6, axis=1), alias_of="B0.6", in_default_sweep=False))
    plate = (1.0, 1.0, 0.02)
    items.append(Condition(
        "Cp", "C", "2 kg 1x1 m plate",
        "the originally proposed payload; loads all three axes at once so it "
        "confirms rather than isolates",
        brick_payload(2.0, (0.0, 0.0, DECK_Z), plate)))

    # D: centre-of-mass offset, mass and shape held at the A2 brick.
    for x in (-0.15, 0.15):
        items.append(Condition(
            f"Dx{'m' if x < 0 else 'p'}", "D", f"2 kg brick at x={x:+.2f} m",
            "constant pitch moment; hardest on the on/off ramp of the beam",
            brick_payload(2.0, (x, 0.0, DECK_Z))))
    for y in (0.05, 0.10): 
        items.append(Condition(
            f"Dy{int(y * 100)}", "D", f"2 kg brick at y={y:+.2f} m",
            "constant roll moment: expected to be the most damaging per mm on a "
            "0.10 m beam",
            brick_payload(2.0, (0.0, y, DECK_Z))))
    for z in (0.20, 0.35):
        items.append(Condition(
            f"Dz{int(z * 100)}", "D", f"2 kg brick at z={z:.2f} m",
            "raised centre of mass; shrinks the inverted-pendulum margin",
            brick_payload(2.0, (0.0, 0.0, z))))

    # E: a plausible deployment payload combining every effect at once.
    items.append(Condition(
        "E1", "E", "3 kg box, off-centre and high",
        "realistic backpack / spare battery; the deployment-facing summary point",
        brick_payload(3.0, (-0.10, 0.05, 0.15), (0.30, 0.25, 0.20))))

    # F: the physical wire basket, measured off the real fixture.
    items.append(Condition(
        "F0", "F", "empty wire basket, 0.5 kg",
        "geometry first: does the real basket clear the camera and the legs, and "
        "what does its bare frame alone cost",
        basket_payload(0.5)))
    items.append(Condition(
        "F2", "F", "wire basket carrying 2 kg, welded in place",
        "the fixture as it will be used; cargo sits low and inside the frame",
        basket_payload(0.5, cargo_mass=2.0)))

    # G: the same basket, but the load is free to move inside it.  G2 is F2 with
    # exactly one thing changed, which is what makes the pair worth running.
    for mass in (1.0, 2.0, 4.0):
        items.append(Condition(
            f"G{mass:g}", "G", f"basket with {mass:g} kg loose on its floor",
            "shifting cargo: the load slides and tips against the basket walls "
            "instead of riding rigidly with the trunk",
            walled_basket_payload(0.5), cargo=cargo_at_rest(mass)))
    items.append(Condition(
        "G2f", "G", "basket with a small 2 kg load, room to slide",
        "same mass as G2 with twice the travel before it hits a wall, which "
        "separates how far cargo moves from how much it weighs",
        walled_basket_payload(0.5), cargo=cargo_at_rest(2.0, CARGO_SIZE_LOOSE)))
    # Low-friction pair.  Measured over a full episode, a mu = 0.5 load slides
    # 1-3 mm in total and is indistinguishable from the welded F2; the same load
    # at mu = 0.1 slides 110 mm (snug) and 204 mm (small).  These are the two
    # conditions where series G actually differs from series F.
    items.append(Condition(
        "G2s", "G", "basket with a slippery 2 kg load",
        "cargo that really does shift: same geometry as G2, low friction",
        walled_basket_payload(0.5),
        cargo=cargo_at_rest(2.0, CARGO_SIZE, CARGO_FRICTION_SLIPPERY)))
    items.append(Condition(
        "G2fs", "G", "basket with a small slippery 2 kg load",
        "the most mobile load in the set: low friction and twice the travel",
        walled_basket_payload(0.5),
        cargo=cargo_at_rest(2.0, CARGO_SIZE_LOOSE, CARGO_FRICTION_SLIPPERY)))

    return {c.cid: c for c in items}


CONDITIONS = build_conditions()


# ---------------------------------------------------------------- robot model

def _vec(text, default=None):
    if text is None:
        return default
    return np.array([float(v) for v in text.split()])


def _quat2R(q):
    n = np.linalg.norm(q)
    if n == 0.0:
        return np.eye(3)
    w, x, y, z = q / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _axis2R(axis, theta):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


_STANCE = {"hip": 0.1, "thigh": 0.9, "calf": -1.8}   # the policy action offset
_HIP_SIGN = {"FL": +1, "RL": +1, "FR": -1, "RR": -1}
_CLASS_AXIS = {"abduction": np.array([1.0, 0.0, 0.0])}
_DEFAULT_AXIS = np.array([0.0, 1.0, 0.0])

_robot_cache: tuple[float, np.ndarray, np.ndarray] | None = None


def robot_properties() -> tuple[float, np.ndarray, np.ndarray]:
    """(mass, CoM, inertia about the CoM) of the bare robot at the nominal
    stance, in the base_link frame.  Any <include> is ignored, so this stays
    correct no matter which condition is currently applied."""
    global _robot_cache
    if _robot_cache is not None:
        return _robot_cache

    root = ET.parse(GO2_XML).getroot()
    parts: list[tuple[float, np.ndarray, np.ndarray]] = []

    def walk(body, p_parent, R_parent):
        p = p_parent + R_parent @ _vec(body.get("pos"), np.zeros(3))
        R = R_parent @ _quat2R(_vec(body.get("quat"), np.array([1.0, 0, 0, 0])))
        joint = body.find("joint")
        if joint is not None:
            name = joint.get("name", "")
            axis = _vec(joint.get("axis"),
                        _CLASS_AXIS.get(joint.get("class", ""), _DEFAULT_AXIS))
            leg, kind = name[:2], name.split("_")[1]
            angle = _STANCE[kind] * (_HIP_SIGN[leg] if kind == "hip" else 1.0)
            jpos = _vec(joint.get("pos"), np.zeros(3))
            Rj = _axis2R(axis, angle)
            p = p + R @ (jpos - Rj @ jpos)
            R = R @ Rj
        inertial = body.find("inertial")
        if inertial is not None:
            Rin = R @ _quat2R(_vec(inertial.get("quat"), np.array([1.0, 0, 0, 0])))
            parts.append((
                float(inertial.get("mass")),
                p + R @ _vec(inertial.get("pos")),
                Rin @ np.diag(_vec(inertial.get("diaginertia"))) @ Rin.T,
            ))
        for child in body.findall("body"):
            walk(child, p, R)

    base = root.find("worldbody").find("body")
    walk(base, np.zeros(3), np.eye(3))
    # The base body carries a world pos in the scene; re-reference to its origin.
    origin = _vec(base.get("pos"), np.zeros(3))

    mass = sum(m for m, _, _ in parts)
    com = sum(m * c for m, c, _ in parts) / mass - origin
    inertia = np.zeros((3, 3))
    for m, c, own in parts:
        d = (c - origin) - com
        inertia += own + m * (d @ d * np.eye(3) - np.outer(d, d))
    _robot_cache = (mass, com, inertia)
    return _robot_cache


def combined_properties(payload: Payload | None, cargo: "Cargo | None" = None):
    """Whole-robot mass, CoM and inertia, with loose cargo taken at its rest
    pose.  Once an episode runs the cargo moves, so these numbers describe the
    configuration the robot starts from, not a constant."""
    mass_r, com_r, inertia_r = robot_properties()
    parts = []
    if payload is not None:
        parts.append((payload.mass, np.array(payload.com), np.diag(payload.diaginertia)))
    if cargo is not None:
        parts.append((cargo.mass, np.array(cargo.rest_com), np.diag(cargo.diaginertia())))
    if not parts:
        return mass_r, com_r.copy(), inertia_r.copy()

    mass = mass_r + sum(m for m, _, _ in parts)
    com = (mass_r * com_r + sum(m * c for m, c, _ in parts)) / mass
    d = com_r - com
    inertia = inertia_r + mass_r * (d @ d * np.eye(3) - np.outer(d, d))
    for m, c, own in parts:
        d = c - com
        inertia += own + m * (d @ d * np.eye(3) - np.outer(d, d))
    return mass, com, inertia


def derived(condition: Condition) -> dict:
    mass_r, com_r, inertia_r = robot_properties()
    mass, com, inertia = combined_properties(condition.payload, condition.cargo)
    diag, diag_r = np.diag(inertia), np.diag(inertia_r)
    return {
        "total_mass_kg": float(mass),
        "mass_increase_pct": float(100.0 * (mass - mass_r) / mass_r),
        "com_base_frame_m": [float(v) for v in com],
        "com_shift_mm": [float(v) for v in (com - com_r) * 1000.0],
        "inertia_diag": [float(v) for v in diag],
        "inertia_ratio": [float(v) for v in diag / diag_r],
    }


# ------------------------------------------------------------------ rendering

def render(condition: Condition) -> str:
    head = [
        "<mujocoinclude>",
        "  <!-- " + "=" * 68,
        "    GENERATED by scripts/payload_conditions.py, do not edit by hand.",
        f"    CONDITION {condition.cid} (series {condition.series}): {condition.title}",
        f"    Tests: {condition.tests}",
        "",
        "    Included by go2.xml inside <body name=\"base_link\">, so the payload is",
        "    rigidly welded to the trunk: no joint, no relative motion, no sloshing.",
        f"    Geoms are visual only (group {VIEWER_ONLY_GROUP}, contype 0): the viewer",
        "    shows them, the depth camera never does, and they never collide.",
    ]
    if condition.payload is None:
        head += ["", "    Baseline: no payload body, only the massless mount marker.",
                 "  " + "=" * 68 + " -->", "", mount_site(), "</mujocoinclude>", ""]
        return "\n".join(head)

    payload = condition.payload
    props = derived(condition)
    mass_r, _, _ = robot_properties()
    head += [
        "",
        f"    mass        {payload.mass:g} kg  ({mass_r:.3f} -> "
        f"{props['total_mass_kg']:.3f} kg, {props['mass_increase_pct']:+.1f}%)",
        "    CoM         ({:+.4f}, {:+.4f}, {:+.4f}) m from the base_link origin".format(*payload.com),
        "    diaginertia {:.6g} {:.6g} {:.6g} kg m^2 about that CoM".format(*payload.diaginertia),
        "",
        "    whole-body CoM shift ({:+.1f}, {:+.1f}, {:+.1f}) mm".format(*props["com_shift_mm"]),
        "    whole-body I         [{:.4f}, {:.4f}, {:.4f}]".format(*props["inertia_diag"]),
        "    vs baseline          [x{:.2f}, x{:.2f}, x{:.2f}]  (roll, pitch, yaw)".format(*props["inertia_ratio"]),
    ]
    for warning in payload.validate():
        head += ["", f"    NOTE: {warning}"]
    head += ["  " + "=" * 68 + " -->", ""]

    body = [
        '  <body name="payload" pos="{:.6g} {:.6g} {:.6g}">'.format(*payload.com),
        '    <inertial pos="0 0 0" mass="{:.6g}"'.format(payload.mass),
        '      diaginertia="{:.8g} {:.8g} {:.8g}"/>'.format(*payload.diaginertia),
    ]
    for geom in payload.geoms:
        body += geom_lines(geom)
    body += ["  </body>"]
    body += cargo_lines(condition.cargo)
    body += [mount_site(), "</mujocoinclude>", ""]
    return "\n".join(head + body)


CARGO_AXES = (("tx", "slide", "1 0 0"), ("ty", "slide", "0 1 0"),
              ("tz", "slide", "0 0 1"), ("rx", "hinge", "1 0 0"),
              ("ry", "hinge", "0 1 0"), ("rz", "hinge", "0 0 1"))


def cargo_lines(cargo: "Cargo | None") -> list[str]:
    if cargo is None:
        return []
    lines = [
        "",
        '  <body name="cargo" pos="{:.6g} {:.6g} {:.6g}">'.format(*cargo.rest_com),
        '    <inertial pos="0 0 0" mass="{:.6g}"'.format(cargo.mass),
        '      diaginertia="{:.8g} {:.8g} {:.8g}"/>'.format(*cargo.diaginertia()),
    ]
    ranges = {"tx": 0, "ty": 1, "tz": 2}
    for name, kind, axis in CARGO_AXES:
        limit = ""
        if cargo.travel and name in ranges:
            lo, hi = cargo.travel[ranges[name]]
            limit = f' limited="true" range="{lo:.6g} {hi:.6g}"'
        lines.append(
            f'    <joint name="cargo_{name}" type="{kind}" axis="{axis}"{limit}'
            ' damping="0" armature="0" frictionloss="0"/>')
    for name, kind, axis in ():
        # damping, armature and frictionloss must all be stated: the go2 default
        # class this body inherits from sets 0.1 / 0.01 / 0.2, which would make a
        # passive object behave like a driven joint.
        lines.append(
            f'    <joint name="cargo_{name}" type="{kind}" axis="{axis}"'
            ' damping="0" armature="0" frictionloss="0"/>')
    lines += geom_lines(Geom("cargo_geom", cargo.size, (0.0, 0.0, 0.0),
                             rgba=cargo.rgba, collide=True))
    lines.append("  </body>")
    return lines


def geom_lines(geom: Geom) -> list[str]:
    if geom.fromto is not None:
        lines = [
            f'    <geom name="{geom.name}" type="capsule"',
            '      fromto="{:.6g} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g}"'.format(*geom.fromto),
            f'      size="{geom.radius:.6g}"',
        ]
    else:
        half = tuple(v / 2.0 for v in geom.full_size)
        lines = [
            f'    <geom name="{geom.name}" type="box"',
            '      size="{:.6g} {:.6g} {:.6g}" pos="{:.6g} {:.6g} {:.6g}"'.format(*half, *geom.pos),
        ]
    if geom.euler is not None:
        lines.append('      euler="{:.6g} {:.6g} {:.6g}"'.format(*geom.euler))
    lines.append(f'      group="{VIEWER_ONLY_GROUP}" contype="0" conaffinity="0"')
    lines.append(f'      rgba="{geom.rgba}"/>')
    return lines


def render_contacts(condition: Condition) -> str:
    """Model-level contact pairs, written to a second include file."""
    if condition.cargo is None or condition.payload is None:
        # MuJoCo refuses an include file with no elements, so emit an empty
        # contact section rather than nothing at all.
        return ("<mujocoinclude>\n"
                f"  <!-- condition {condition.cid} carries nothing loose -->\n"
                "  <contact/>\n</mujocoinclude>\n")
    walls = [g.name for g in condition.payload.geoms if g.name.startswith("basket_")
             and g.fromto is None]
    lines = ["<mujocoinclude>",
             f"  <!-- condition {condition.cid}: cargo against its container. -->",
             "  <contact>"]
    mu = condition.cargo.friction
    for wall in walls:
        lines.append(f'    <pair geom1="cargo_geom" geom2="{wall}"'
                     f' condim="3" friction="{mu:g} {mu:g} 0.005 0.0001 0.0001"/>')
    lines += ["  </contact>", "</mujocoinclude>", ""]
    return "\n".join(lines)


def apply(condition: Condition, path: Path = PAYLOAD_XML,
          contacts_path: Path = CONTACTS_XML) -> Path:
    if condition.payload is not None:
        condition.payload.validate()
    path.write_text(render(condition), encoding="utf-8")
    contacts_path.write_text(render_contacts(condition), encoding="utf-8")
    return path


# ------------------------------------------------------------------------ CLI

def resolve(cid: str) -> Condition:
    if cid not in CONDITIONS:
        raise SystemExit(
            f"error: unknown condition {cid!r}; known: {', '.join(CONDITIONS)}")
    return CONDITIONS[cid]


def default_sweep() -> list[Condition]:
    return [c for c in CONDITIONS.values() if c.in_default_sweep]


def cmd_list(_args) -> int:
    mass_r, _, inertia_r = robot_properties()
    print(f"bare robot: {mass_r:.3f} kg   I = "
          f"[{inertia_r[0,0]:.4f}, {inertia_r[1,1]:.4f}, {inertia_r[2,2]:.4f}] kg m^2\n")
    header = (f"{'id':6s} {'ser':4s} {'mass':>7s} {'dmass':>7s} "
              f"{'dCoM x/y/z (mm)':>22s}  {'I ratio roll/pitch/yaw':>24s}  title")
    print(header)
    print("-" * len(header))
    for condition in CONDITIONS.values():
        d = derived(condition)
        mark = "" if condition.in_default_sweep else "  [not in sweep]"
        print(f"{condition.cid:6s} {condition.series:4s} "
              f"{d['total_mass_kg']:6.2f}k {d['mass_increase_pct']:+6.1f}% "
              f"{d['com_shift_mm'][0]:+7.1f}{d['com_shift_mm'][1]:+7.1f}{d['com_shift_mm'][2]:+7.1f}  "
              f"x{d['inertia_ratio'][0]:5.2f} x{d['inertia_ratio'][1]:5.2f} x{d['inertia_ratio'][2]:5.2f}  "
              f"{condition.title}{mark}")
    return 0


def cmd_show(args) -> int:
    condition = resolve(args.condition)
    d = derived(condition)
    print(f"condition {condition.cid}  (series {condition.series})")
    print(f"  title : {condition.title}")
    print(f"  tests : {condition.tests}")
    if condition.alias_of:
        print(f"  alias : physically identical to {condition.alias_of}")
    print(f"  total mass    {d['total_mass_kg']:.4f} kg ({d['mass_increase_pct']:+.2f}%)")
    print("  CoM shift     {:+.2f} {:+.2f} {:+.2f} mm".format(*d["com_shift_mm"]))
    print("  inertia       {:.4f} {:.4f} {:.4f}".format(*d["inertia_diag"]))
    print("  vs baseline   x{:.3f} x{:.3f} x{:.3f}  (roll, pitch, yaw)".format(*d["inertia_ratio"]))
    if condition.payload:
        for warning in condition.payload.validate():
            print(f"  WARNING       {warning}")
    print()
    print(render(condition))
    return 0


def cmd_apply(args) -> int:
    condition = resolve(args.condition)
    path = Path(args.payload_xml) if args.payload_xml else PAYLOAD_XML
    apply(condition, path)
    d = derived(condition)
    print(f"[apply] {condition.cid}: {condition.title}")
    print(f"[apply] wrote {path}")
    print(f"[apply] wrote {CONTACTS_XML}")
    print(f"[apply] total {d['total_mass_kg']:.3f} kg, "
          "I ratio x{:.2f}/x{:.2f}/x{:.2f}".format(*d["inertia_ratio"]))
    print("[apply] restart unitree_mujoco for this to take effect")
    return 0


def cmd_manifest(args) -> int:
    mass_r, com_r, inertia_r = robot_properties()
    doc = {
        "bare_robot": {
            "mass_kg": float(mass_r),
            "com_base_frame_m": [float(v) for v in com_r],
            "inertia_diag": [float(v) for v in np.diag(inertia_r)],
        },
        "conditions": [
            {
                "id": c.cid, "series": c.series, "title": c.title, "tests": c.tests,
                "alias_of": c.alias_of, "in_default_sweep": c.in_default_sweep,
                "payload": None if c.payload is None else {
                    "mass_kg": c.payload.mass,
                    "com_m": list(c.payload.com),
                    "diaginertia": list(c.payload.diaginertia),
                },
                "derived": derived(c),
            }
            for c in CONDITIONS.values()
        ],
    }
    text = json.dumps(doc, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="table of every condition").set_defaults(func=cmd_list)
    p = sub.add_parser("show", help="detail plus the emitted MJCF")
    p.add_argument("condition")
    p.set_defaults(func=cmd_show)
    p = sub.add_parser("apply", help="write payload.xml for one condition")
    p.add_argument("condition")
    p.add_argument("--payload-xml", help=f"override the default {PAYLOAD_XML}")
    p.set_defaults(func=cmd_apply)
    p = sub.add_parser("manifest", help="machine-readable dump of every condition")
    p.add_argument("-o", "--out")
    p.set_defaults(func=cmd_manifest)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
