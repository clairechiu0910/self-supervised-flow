TEXTURES = ['carpet', 'grid', 'leather', 'tile', 'wood', 
            '02', 
            'Class1', 'Class2', 'Class3', 'Class4', 'Class5', 
            'Class6', 'Class7', 'Class8', 'Class9', 'Class10']

# brightness, threshold pairs
BACKGROUND = {
    'bottle': (200, 60),
    'screw': (200, 60),
    'capsule': (200, 60),
    'zipper': (200, 60),
    'hazelnut': (20, 20),
    'pill': (20, 20),
    'toothbrush': (20, 20),
    'metal_nut': (20, 20),
    '01': (200, 60),
    '03': (200, 60)
}

# note: these are half-widths in [0, 0.5]
# ((h_min, h_max), (w_min, w_max))
WIDTH_BOUNDS_PCT = {
    'carpet': ((0.03, 0.4), (0.03, 0.4)),
    'grid': ((0.03, 0.4), (0.03, 0.4)),
    'leather': ((0.03, 0.4), (0.03, 0.4)),
    'tile': ((0.03, 0.4), (0.03, 0.4)),
    'wood': ((0.03, 0.4), (0.03, 0.4)),
    
    'bottle': ((0.03, 0.4), (0.03, 0.4)),
    'cable': ((0.05, 0.4), (0.05, 0.4)),
    'capsule': ((0.03, 0.15), (0.03, 0.4)),
    'hazelnut': ((0.03, 0.35), (0.03, 0.35)),
    'metal_nut': ((0.03, 0.4), (0.03, 0.4)),
    'pill': ((0.03, 0.2), (0.03, 0.4)),
    'screw': ((0.03, 0.12), (0.03, 0.12)),
    'toothbrush': ((0.03, 0.4), (0.03, 0.2)),
    'transistor': ((0.03, 0.4), (0.03, 0.4)),
    'zipper': ((0.03, 0.4), (0.03, 0.2)),
    
    '01': ((0.03, 0.4), (0.03, 0.4)),
    '02': ((0.03, 0.4), (0.03, 0.4)),
    '03': ((0.03, 0.4), (0.03, 0.4)),

    'Class1': ((0.03, 0.4), (0.03, 0.4)),
    'Class2': ((0.03, 0.4), (0.03, 0.4)),
    'Class3': ((0.03, 0.4), (0.03, 0.4)),
    'Class4': ((0.03, 0.4), (0.03, 0.4)),
    'Class5': ((0.03, 0.4), (0.03, 0.4)),
    'Class6': ((0.03, 0.4), (0.03, 0.4)),
    'Class7': ((0.03, 0.4), (0.03, 0.4)),
    'Class8': ((0.03, 0.4), (0.03, 0.4)),
    'Class9': ((0.03, 0.4), (0.03, 0.4)),
    'Class10': ((0.03, 0.4), (0.03, 0.4)),
}

MIN_OVERLAP_PCT = {
    'bottle': 0.25,
    'capsule': 0.25,
    'hazelnut': 0.25,
    'metal_nut': 0.25,
    'pill': 0.25,
    'screw': 0.25,
    'toothbrush': 0.25,
    'zipper': 0.25,
    '01': 0.25,
    '03': 0.25,
}

MIN_OBJECT_PCT = {
    'bottle': 0.7,
    'capsule': 0.7,
    'hazelnut': 0.7,
    'metal_nut': 0.5,
    'pill': 0.7,
    'screw': .5,
    'toothbrush': 0.25,
    'zipper': 0.7,
    '01': 0.7,
    '03': 0.7,
}

NUM_PATCHES = {
    'carpet': 4,
    'grid': 4,
    'leather': 4,
    'tile': 4,
    'wood': 4,
    
    'bottle': 3,
    'cable': 3,
    'capsule': 3,
    'hazelnut': 3,
    'metal_nut': 3,
    'pill': 3,
    'screw': 4,
    'toothbrush': 3,
    'transistor': 3,
    'zipper': 4,
    
    '01': 3,
    '02': 4,
    '03': 3,

    'Class1': 4,
    'Class2': 4,
    'Class3': 4,
    'Class4': 4,
    'Class5': 4,
    'Class6': 4,
    'Class7': 4,
    'Class8': 4,
    'Class9': 4,
    'Class10': 4,
}

# k, x0 pairs
INTENSITY_LOGISTIC_PARAMS = {
    'bottle': (1 / 12, 24),
    'cable': (1 / 12, 24),
    'capsule': (1 / 2, 4),
    'hazelnut': (1 / 12, 24),
    'metal_nut': (1 / 3, 7),
    'pill': (1 / 3, 7),
    'screw': (1, 3),
    'toothbrush': (1 / 6, 15),
    'transistor': (1 / 6, 15),
    'zipper': (1 / 6, 15),

    'carpet': (1 / 3, 7),
    'grid': (1 / 3, 7),
    'leather': (1 / 3, 7),
    'tile': (1 / 3, 7),
    'wood': (1 / 6, 15),
    
    '01': (1 / 12, 24),
    '02': (1 / 3, 7),
    '03': (1 / 12, 24),

    'Class1': (1 / 3, 7),
    'Class2': (1 / 3, 7),
    'Class3': (1 / 3, 7),
    'Class4': (1 / 3, 7),
    'Class5': (1 / 3, 7),
    'Class6': (1 / 3, 7),
    'Class7': (1 / 3, 7),
    'Class8': (1 / 3, 7),
    'Class9': (1 / 3, 7),
    'Class10': (1 / 3, 7),
}
