"""
Analyze CWT parameter choices for LISA vs LIGO.

This script compares the CWT parameters used for LIGO and LISA
to determine if we made intelligent adaptations or just blindly copied.
"""

import numpy as np

print("="*70)
print("CWT PARAMETER ANALYSIS: LIGO vs LISA")
print("="*70)

# LIGO Parameters (from legacy)
ligo = {
    'fmin': 20.0,          # Hz
    'fmax': 512.0,         # Hz
    'sampling_rate': 4096,  # Hz
    'duration': 1.0,        # seconds
    'target_height': 8,     # frequency bins
    'target_width': 4096,   # time bins
    'n_scales': 64,         # CWT scales (not explicitly shown in config)
}

# LISA Parameters (current)
lisa = {
    'fmin': 1e-4,          # Hz
    'fmax': 1e-1,          # Hz
    'sampling_rate': 1.0,   # Hz
    'duration': 3600,       # seconds (1 hour)
    'target_height': 64,    # frequency bins
    'target_width': 3600,   # time bins
    'n_scales': 64,         # CWT scales
}

print("\n" + "="*70)
print("RAW PARAMETERS")
print("="*70)

print("\nLIGO:")
for k, v in ligo.items():
    print(f"  {k:20s} = {v}")

print("\nLISA:")
for k, v in lisa.items():
    print(f"  {k:20s} = {v}")

print("\n" + "="*70)
print("DERIVED QUANTITIES")
print("="*70)

# Frequency band
ligo['freq_band'] = ligo['fmax'] - ligo['fmin']
lisa['freq_band'] = lisa['fmax'] - lisa['fmin']

# Nyquist frequency
ligo['nyquist'] = ligo['sampling_rate'] / 2
lisa['nyquist'] = lisa['sampling_rate'] / 2

# Number of time samples
ligo['n_time_samples'] = int(ligo['sampling_rate'] * ligo['duration'])
lisa['n_time_samples'] = int(lisa['sampling_rate'] * lisa['duration'])

# Frequency resolution of CWT
ligo['freq_per_bin'] = ligo['freq_band'] / ligo['target_height']
lisa['freq_per_bin'] = lisa['freq_band'] / lisa['target_height']

# Time resolution of CWT
ligo['time_per_bin'] = ligo['duration'] / ligo['target_width']
lisa['time_per_bin'] = lisa['duration'] / lisa['target_width']

# Octaves spanned
ligo['octaves'] = np.log2(ligo['fmax'] / ligo['fmin'])
lisa['octaves'] = np.log2(lisa['fmax'] / lisa['fmin'])

# Scales per octave
ligo['scales_per_octave'] = ligo['n_scales'] / ligo['octaves']
lisa['scales_per_octave'] = lisa['n_scales'] / lisa['octaves']

print("\nFrequency Band:")
print(f"  LIGO: {ligo['freq_band']:.1f} Hz ({ligo['fmin']:.1f} - {ligo['fmax']:.1f} Hz)")
print(f"  LISA: {lisa['freq_band']:.6f} Hz ({lisa['fmin']:.6f} - {lisa['fmax']:.6f} Hz)")
print(f"  Ratio (LIGO/LISA): {ligo['freq_band'] / lisa['freq_band']:.1f}x")

print("\nNyquist Frequency:")
print(f"  LIGO: {ligo['nyquist']:.1f} Hz")
print(f"  LISA: {lisa['nyquist']:.1f} Hz")

print("\nTime Samples:")
print(f"  LIGO: {ligo['n_time_samples']} samples")
print(f"  LISA: {lisa['n_time_samples']} samples")

print("\nFrequency Resolution (per CWT bin):")
print(f"  LIGO: {ligo['freq_per_bin']:.2f} Hz/bin")
print(f"  LISA: {lisa['freq_per_bin']:.6f} Hz/bin")
print(f"  Ratio (LIGO/LISA): {ligo['freq_per_bin'] / lisa['freq_per_bin']:.1f}x")

print("\nTime Resolution (per CWT bin):")
print(f"  LIGO: {ligo['time_per_bin']*1000:.3f} ms/bin")
print(f"  LISA: {lisa['time_per_bin']:.3f} s/bin = {lisa['time_per_bin']*1000:.1f} ms/bin")

print("\nOctaves Spanned:")
print(f"  LIGO: {ligo['octaves']:.2f} octaves")
print(f"  LISA: {lisa['octaves']:.2f} octaves")

print("\nScales per Octave:")
print(f"  LIGO: {ligo['scales_per_octave']:.2f} scales/octave")
print(f"  LISA: {lisa['scales_per_octave']:.2f} scales/octave")

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

print("\n1. FREQUENCY RESOLUTION:")
ratio = ligo['freq_per_bin'] / lisa['freq_per_bin']
print(f"   LIGO has {ratio:.0f}x COARSER frequency resolution")
print(f"   BUT LIGO also spans {ligo['freq_band']/lisa['freq_band']:.0f}x WIDER band")
print(f"   > Relative coverage: LIGO uses {ligo['target_height']} bins for {ligo['octaves']:.1f} octaves")
print(f"   > Relative coverage: LISA uses {lisa['target_height']} bins for {lisa['octaves']:.1f} octaves")

scales_per_octave_ratio = lisa['scales_per_octave'] / ligo['scales_per_octave']
print(f"   > LISA has {scales_per_octave_ratio:.1f}x MORE scales per octave")

if scales_per_octave_ratio > 1.5:
    print(f"   WARNING: LISA is OVER-RESOLVING in frequency!")
    print(f"       We may be wasting parameters on frequency resolution")
    print(f"       Consider reducing target_height to match LIGO's scales/octave")
    suggested_height = int(lisa['octaves'] * ligo['scales_per_octave'])
    print(f"       Suggested target_height: ~{suggested_height}")

print("\n2. TIME RESOLUTION:")
print(f"   LIGO: {ligo['time_per_bin']*1000:.3f} ms per bin")
print(f"   LISA: {lisa['time_per_bin']*1000:.1f} ms per bin")
ratio = lisa['time_per_bin'] / ligo['time_per_bin']
print(f"   > LISA has {ratio:.0f}x COARSER time resolution")

if ratio > 2:
    print(f"   NOTE: LISA time resolution is much coarser")
    print(f"       But this makes sense: LISA signals evolve much more slowly")
    print(f"       LIGO: chirps in ~seconds, LISA: chirps in ~hours")

print("\n3. DURATION:")
print(f"   LIGO: {ligo['duration']:.1f} seconds")
print(f"   LISA: {lisa['duration']:.1f} seconds = {lisa['duration']/3600:.1f} hour")
print(f"   > This is appropriate: LISA needs longer observations for mHz signals")

print("\n4. ASPECT RATIO:")
ligo_aspect = ligo['target_width'] / ligo['target_height']
lisa_aspect = lisa['target_width'] / lisa['target_height']
print(f"   LIGO: {ligo['target_width']} x {ligo['target_height']} = {ligo_aspect:.1f}:1 aspect ratio")
print(f"   LISA: {lisa['target_width']} x {lisa['target_height']} = {lisa_aspect:.1f}:1 aspect ratio")

if abs(ligo_aspect - lisa_aspect) / ligo_aspect > 0.5:
    print(f"   WARNING: Aspect ratios differ significantly!")
    print(f"       LISA is more square, LIGO is very wide")
    print(f"       This changes the inductive bias of the CNN!")

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

print("\n1. FREQUENCY BINS (target_height):")
print(f"   Current: {lisa['target_height']}")
suggested_height = max(8, int(lisa['octaves'] * ligo['scales_per_octave']))
print(f"   Suggested: {suggested_height} (to match LIGO's ~13 scales/octave)")
print(f"   Justification: Match the scales-per-octave from LIGO")

print("\n2. TIME BINS (target_width):")
print(f"   Current: {lisa['target_width']}")
print(f"   This seems reasonable for 1 hour at 1 Hz sampling")
print(f"   Could downsample more if memory is an issue")

print("\n3. N_SCALES:")
print(f"   Current: {lisa['n_scales']}")
suggested_scales = int(lisa['octaves'] * 10)  # 10 scales per octave is standard
print(f"   Suggested: {suggested_scales} (10 scales/octave is standard for CWT)")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

print("""
It appears we BLINDLY COPIED some parameters without careful consideration:

[X] BAD: We increased target_height from 8 to 64 (8x more) even though 
        LISA spans 5000x SMALLER frequency range. This gives us way too
        many frequency bins per octave!

[X] BAD: We kept n_scales=64 the same, even though LISA spans 10 octaves
        vs LIGO's 4.7 octaves. This actually REDUCES our scales/octave
        from 13.7 to 6.4!

[OK] GOOD: Duration increase (1s to 1hr) makes sense for LISA's slow evolution

[OK] GOOD: Time resolution decrease makes sense for slower signals

RECOMMENDATION:
We should re-run with more carefully chosen parameters that match LIGO's
scales-per-octave in frequency space while adapting time resolution 
appropriately for LISA's physics.
""")

