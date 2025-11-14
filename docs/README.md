# Documentation

This directory contains **public-facing documentation** for the LISA Manifold Learning project.

## For Users

Start here:
1. **[Quick Start](QUICK_START.md)** - Get up and running quickly
2. **[Project Overview](PROJECT_OVERVIEW.md)** - High-level overview of goals and approach

## Core Documentation

### Scientific Foundation
- **[Mathematical Framework](MATHEMATICAL_FRAMEWORK.md)** - Rigorous mathematical formalism using differential geometry
  - Manifold hypothesis and smooth manifold structure
  - Autoencoder as chart, latent space geometry
  - Tangent space estimation and off-manifold distance
  - The β coefficient and its physical interpretation

### Previous Work
- **[LIGO Experiment Summary](MANIFOLD_FINAL_SUMMARY.md)** - Summary of previous LIGO work
  - Result: β = 0 (geometry didn't help)
  - Lessons learned and motivation for LISA

## Testing & Validation

- **[Testing Guide](TESTING.md)** - How to run tests
- **[Validation](VALIDATION.md)** - Data validation procedures

## Project Structure

```
cwt-manifold-grav-wav/
├── src/                  # Source code
│   ├── data/            # LISA noise and waveform models
│   ├── preprocessing/   # CWT transform
│   ├── models/          # Autoencoder architectures
│   ├── geometry/        # Manifold geometry computation
│   └── evaluation/      # Manifold-based scoring
├── tests/               # Test suite (137 tests)
├── config/              # Configuration files
├── scripts/             # Utility scripts
└── docs/                # This directory
```

## Scientific Question

**Can manifold learning improve gravitational wave source separation in LISA data?**

- **Challenge**: LISA observes many overlapping signals (confusion noise)
- **Approach**: Train autoencoder on confusion background, use manifold geometry to detect resolvable sources
- **Key Metric**: β coefficient (manifold geometry weight)
  - β = 0: Geometry doesn't help (LIGO result)
  - β > 0: Geometry provides discriminative power (LISA hypothesis)

## Key Concepts

### The Manifold Hypothesis
Normal data (confusion background) lies on a low-dimensional manifold $\mathcal{M}$ embedded in high-dimensional data space. Anomalies (resolvable sources) lie **off** the manifold.

### Off-Manifold Distance
For a point $z$ in latent space, decompose relative to manifold:
$$z - \mu = v_{\parallel} + v_{\perp}$$

The **off-manifold distance** $\delta_{\perp} = \|v_{\perp}\|$ measures geometric deviation.

### Combined Anomaly Score
$$s(x) = \alpha \cdot \epsilon(x) + \beta \cdot \delta_{\perp}(\phi(x))$$

where:
- $\epsilon(x)$ = reconstruction error (standard autoencoder)
- $\delta_{\perp}$ = off-manifold distance (manifold geometry)
- **β = weighting coefficient (what we measure!)**

## Development

For internal development notes, see `dev_docs/` (gitignored, not tracked).

## References

- Lee, J.M. (2012). *Introduction to Smooth Manifolds*. Springer.
- Lee, J.M. (2018). *Introduction to Riemannian Manifolds*. Springer.
- Bengio, Y., et al. (2013). "Representation Learning." *IEEE TPAMI*.

## Contact & Contributing

This is a research project investigating manifold learning for LISA gravitational wave detection.

---

**Quick Links**:
- [Mathematical Framework](MATHEMATICAL_FRAMEWORK.md) - Full mathematical details
- [Quick Start](QUICK_START.md) - Get started quickly
- [Project Overview](PROJECT_OVERVIEW.md) - High-level overview

