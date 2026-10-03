# 5-DOF Arm — LSPB Trajectory Planning & Control

Plans joint-space LSPB (trapezoidal velocity) trajectories to reach a
Cartesian target pose, quantizes them to the servos' 1-degree
resolution, and either plots the result for verification or streams it
to hardware over serial.

## Files

| File | Purpose |
|---|---|
| `lspb_joint_limits.py` | `ServoCalibration` (DH ↔ servo angle mapping) and `LSPBTrajectory` (the trapezoidal-velocity planner). Library module — no example/CLI code. |
| `lspb_symmetric_planning.py` | `SymmetricPlanningView` — optional wrapper that lets you plan/specify joint targets in a uniform `[-90, +90]` range even for joints whose true DH range is off-center. Library module. |
| `lspb_quantized_plot.py` | Solves IK for a Cartesian target, plans the move, plots ideal vs. 1°-quantized joint paths. **No hardware needed** — use this to sanity-check a move first. |
| `lspb_serial_send.py` | Same pipeline as above, then streams the quantized commands to the arm over serial at a fixed 40 ms interval. |

## Dependencies

```
pip install numpy scipy matplotlib pyserial
```

Plus the `arm5dof` package (forward/inverse kinematics), which must be
importable from these scripts.

## Kinematics: the `arm5dof` package

Forward and inverse kinematics are provided by `arm5dof`, not hand-rolled
DH code. `q = [0, 0, 0, 0, 0]` corresponds directly to hardware home
(servo `[90, 180, 180, 90, 0]`) — there's no separate "kin space" vs
"hardware space" offset to track.

```python
import numpy as np
from arm5dof import fk, ik, pitch_from_rotation, solve

T = fk(np.radians([10, -60, 80, -20, 45]))
p, R = T[:3, 3], T[:3, :3]

sols = ik(p, pitch_from_rotation(p, R), R)          # all valid solutions (rad)
q = solve(p, q_current=np.zeros(5), R=R)             # best constrained solution, or None
```

- **`fk(q_rad)`** — 4×4 end-effector transform for a 5-vector of joint
  angles (radians).
- **`pitch_from_rotation(p, R)`** — pitch angle needed by `ik()`.
- **`ik(p, pitch, R)`** — *every* valid joint-space solution for a given
  position/pitch/orientation. The arm can be redundant for some poses,
  so this can return more than one solution. Used in these scripts only
  as a diagnostic (to see how many branches reach a target).
- **`solve(p, q_current=..., R=R)`** — picks the best solution, biased
  toward `q_current` so the solver doesn't jump to an equally-valid but
  unexpected branch, and **already constrained to the arm's joint
  limits internally**. Returns `None` if no valid solution exists —
  both scripts here check for this and abort rather than sending a
  bogus move.

## Joint limits (degrees)

| Joint | Range |
|---|---|
| q1 | −90 to 90 |
| q2 | −180 to 0 |
| q3 | 0 to 180 |
| q4 | −90 to 90 |
| q5 | 0 to 180 |

These are enforced in two places: internally by `arm5dof.solve()`, and
again by `ServoCalibration`/`LSPBTrajectory` before planning (belt and
suspenders — if the two ever disagree, trust `arm5dof`'s internal
limits and update the calibration below to match).

## Servo calibration

```python
cal = ServoCalibration(offsets=[90, 180, 180, 90, 0],
                        directions=[1, 1, -1, 1, 1],
                        servo_min=0, servo_max=180)
```

- `offsets` — the servo command (deg) that corresponds to joint angle
  `q = 0` for each joint. `[90, 180, 180, 90, 0]` is hardware home.
- `directions` — `+1` if increasing `q` increases the servo command,
  `-1` if the servo is wired/mounted to move the opposite way.
  **q3 is flipped (`-1`) relative to the others — this has not been
  verified against physical hardware.**

### ⚠️ Verify direction signs before trusting any move

Command a small, safe positive-`q` step on each joint *alone* and
confirm it rotates the way `arm5dof.fk()` predicts. Flip the sign in
`directions` for any joint that moves the wrong way, and keep
`lspb_quantized_plot.py` and `lspb_serial_send.py` in sync.

## Usage

### 1. Check a move before running it on hardware

```
python lspb_quantized_plot.py
```

Prints the IK solution(s), the selected target, and move time, then
saves `lspb_quantized_path.png` comparing the ideal continuous path to
the 1°-quantized path for all 5 joints. No serial connection required.

### 2. Run the move on hardware

Edit `SERIAL_PORT` in `lspb_serial_send.py` to match your board, then:

```
python lspb_serial_send.py
```

If `pyserial` isn't installed, the port can't be opened, or
`SERIAL_ENABLED = False`, it falls back to print-only mode automatically
— safe to run without hardware connected.

### Changing the target

Both scripts build their Cartesian target from a known joint
configuration for convenience:

```python
q_default = np.radians([0, -90, 90, 0, 0])   # lspb_serial_send.py
T_target = fk(q_default)
p, R = T_target[:3, 3], T_target[:3, :3]
```

To target a specific Cartesian pose directly instead, build your own
`p` (3-vector) and `R` (3×3 rotation matrix) and skip the `fk()` call.

## Safety notes

- `solve()` returning `None` means the target is unreachable within
  joint limits — both scripts raise an error rather than sending
  anything in that case. Don't bypass this check.
- Always run `lspb_quantized_plot.py` on a new target before
  `lspb_serial_send.py`, and look at the printed IK solution count —
  more than one valid branch means the arm had a choice, and it's worth
  confirming `solve()` picked the one you expect.
- `directions=[1, 1, -1, 1, 1]` is unverified (see above) — a wrong
  sign means a joint moves opposite to plan, which can drive it into a
  hard stop.