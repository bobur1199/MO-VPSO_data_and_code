"""
hard_envs.py
============
Discriminating environments H6-H10 (Section 2.3).

DESIGN PRINCIPLES (fixed BEFORE any algorithm was run on them, so that the
set is not selected to favour any particular method):

  P1 Narrow passage   - gap clearance close to the robot diameter (3.0 units).
                        Clearance ratio r = gap / diameter is stated per env.
  P2 Concave trap     - a U-shaped obstacle whose opening faces the start, so
                        that a greedy path enters a dead end. This is the
                        classical local-minimum geometry for path planners.
  P3 High clutter     - many obstacles creating multiple competing homotopy
                        classes (25 circles vs 5-9 in O1-O5).
  P4 Representation   - a serpentine corridor requiring three direction
        stress         reversals, which stresses the 4-control-point encoding.

Each environment is built from ONE principle plus, in H10, a combination.
Difficulty calibration used the BASELINE algorithm (standard PSO), never the
proposed method, and the only admissible adjustment was to avoid a floor
(all algorithms near 0%) or a ceiling (all algorithms near 100%).
"""

HARD_ENVIRONMENTS = {
    '6': {   # P1 - single narrow gate
        'name': 'Offset narrow gate (ratio 1.33)',
        'principle': 'P1 narrow passage',
        'width': 100, 'height': 100, 'robot_radius': 1.5,
        'start': [10, 50], 'goal': [90, 50],
        'obstacles': [
            {'center': [50, 11.5], 'width': 6, 'height': 23.0},
            {'center': [50, 63.5], 'width': 6, 'height': 73.0},
        ],
    },
    '7': {   # P2 - concave U-trap opening toward the start
        'name': 'Concave U-trap',
        'principle': 'P2 concave trap',
        'width': 100, 'height': 100, 'robot_radius': 1.5,
        'start': [15, 50], 'goal': [85, 50],
        'obstacles': [
            {'center': [55, 72.5], 'width': 40, 'height': 6},
            {'center': [55, 27.5], 'width': 40, 'height': 6},
            {'center': [73, 50.0], 'width': 6,  'height': 45},
        ],
    },
    '8': {   # P3 - dense clutter, 25 circular obstacles
        'name': 'Dense clutter (25 obstacles)',
        'principle': 'P3 high clutter',
        'width': 100, 'height': 100, 'robot_radius': 1.5,
        'start': [5, 5], 'goal': [95, 95],
        'obstacles': [
            {'center': [x, y], 'radius': 5.0}
            for x, y in [
                (22, 18), (38, 12), (55, 22), (72, 14), (88, 25),
                (15, 35), (32, 40), (48, 33), (65, 42), (82, 38),
                (25, 55), (42, 58), (58, 52), (75, 60), (90, 55),
                (18, 72), (35, 78), (52, 70), (68, 80), (85, 72),
                (28, 92), (45, 88), (62, 95), (78, 90), (95, 82),
            ]
        ],
    },
    '9': {   # P4 - serpentine corridor, three reversals
        'name': 'Serpentine corridor (3 reversals)',
        'principle': 'P4 representation stress',
        'width': 100, 'height': 100, 'robot_radius': 1.5,
        'start': [10, 90], 'goal': [90, 5],
        'obstacles': [
            {'center': [30, 70], 'width': 60, 'height': 5},
            {'center': [70, 45], 'width': 60, 'height': 5},
            {'center': [30, 20], 'width': 60, 'height': 5},
        ],
    },
    '10': {  # P1 + P2 - two offset narrow gates in series
        'name': 'Double offset gate (clearance ratio 1.67)',
        'principle': 'P1 combined, sequential',
        'width': 100, 'height': 100, 'robot_radius': 1.5,
        'start': [10, 50], 'goal': [90, 50],
        'obstacles': [
            {'center': [35, 35.0], 'width': 6, 'height': 70},
            {'center': [35, 87.5], 'width': 6, 'height': 25},
            {'center': [70, 12.5], 'width': 6, 'height': 25},
            {'center': [70, 65.0], 'width': 6, 'height': 70},
        ],
    },
}

# Clearance figures stated in Section 2.3
CLEARANCE = {
    '6':  'gap 4.0 units at y=25, offset 25 units from the start-goal line, ratio 1.33',
    '7':  'detour corridors 24.5 units, trap depth 38 units',
    '8':  'obstacle radius 5.0, mean free spacing approx. 7 units',
    '9':  'corridor width 22.5 units, three mandatory reversals',
    '10': 'two gaps of 5.0 units, offset by 45 units in y, ratio 1.67',
}
