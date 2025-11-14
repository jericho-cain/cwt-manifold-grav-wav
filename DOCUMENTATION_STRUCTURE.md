# Documentation Structure

## Overview

Documentation is organized into two directories:

1. **`docs/`** - Public documentation (tracked in git)
2. **`dev_docs/`** - Development notes (gitignored, local only)

## Public Documentation (`docs/`)

**Tracked in version control** - Polished, user-facing documentation

### Contents (7 files)

| File | Purpose |
|------|---------|
| `README.md` | Documentation index and guide |
| `MATHEMATICAL_FRAMEWORK.md` | **Rigorous mathematical formalism** (differential geometry) |
| `MANIFOLD_FINAL_SUMMARY.md` | Previous LIGO experiment results (β = 0) |
| `PROJECT_OVERVIEW.md` | High-level project overview |
| `QUICK_START.md` | Quick start guide |
| `TESTING.md` | Testing procedures |
| `VALIDATION.md` | Data validation procedures |

### Key Document: MATHEMATICAL_FRAMEWORK.md

The crown jewel of the public docs! Contains:
- ✅ Smooth manifold theory (charts, tangent bundles)
- ✅ Riemannian geometry (metrics, geodesics, curvature)
- ✅ Normal bundle and tubular neighborhoods
- ✅ Second fundamental form
- ✅ The β coefficient (mathematical definition)
- ✅ LISA-specific physics
- ✅ References to Lee, Wald, etc.

**Suitable for**:
- Researchers understanding the mathematical foundation
- Reviewers evaluating the approach
- Collaborators learning the methodology
- Future maintainers of the codebase

## Development Documentation (`dev_docs/`)

**Gitignored** - Internal notes, not version controlled

### Contents (14 files)

#### Migration & Planning (5 files)
- `AUTOENCODER_MIGRATION_PLAN.md` - Migration strategy
- `MIGRATION_STATUS.md` - Migration progress
- `CWT_MIGRATION_PLAN.md` - CWT adaptation plan
- `CWT_ADAPTATION_COMPLETE.md` - CWT implementation notes
- `CWT_SUMMARY.md` - CWT quick reference

#### Design Decisions (3 files)
- `MANIFOLD_APPROACH_DECISION.md` - Why AE + manifold approach
- `APPROACH_B_RATIONALE.md` - Why confusion noise focus
- `LISA_VS_LIGO.md` - LIGO vs LISA differences

#### Status & Progress (6 files)
- `README.md` - Dev docs guide
- `CURRENT_STATUS.md` - Current status
- `SESSION_SUMMARY.md` - Session notes
- `VALIDATOR_STATUS.md` - Validator status
- `TEST_SUITE_COMPLETE.md` - Test completion summary
- `NEXT_STEPS.md` - Timeline and next steps

### Why Gitignored?

These documents are:
- **Working notes** that change frequently
- **Developer-specific** (may reference local paths)
- **Potentially incomplete** or superseded
- **Redundant** once decisions are in code
- **Not needed** for external users

## Version Control Status

### Tracked (`.git`)
```
docs/
├── README.md                      ✅ Tracked
├── MATHEMATICAL_FRAMEWORK.md      ✅ Tracked
├── MANIFOLD_FINAL_SUMMARY.md      ✅ Tracked
├── PROJECT_OVERVIEW.md            ✅ Tracked
├── QUICK_START.md                 ✅ Tracked
├── TESTING.md                     ✅ Tracked
└── VALIDATION.md                  ✅ Tracked
```

### Gitignored (`.gitignore`)
```
dev_docs/                          ❌ Ignored
├── README.md                      ❌ Local only
├── AUTOENCODER_MIGRATION_PLAN.md  ❌ Local only
├── MIGRATION_STATUS.md            ❌ Local only
├── CWT_MIGRATION_PLAN.md          ❌ Local only
├── CWT_ADAPTATION_COMPLETE.md     ❌ Local only
├── CWT_SUMMARY.md                 ❌ Local only
├── MANIFOLD_APPROACH_DECISION.md  ❌ Local only
├── APPROACH_B_RATIONALE.md        ❌ Local only
├── LISA_VS_LIGO.md                ❌ Local only
├── CURRENT_STATUS.md              ❌ Local only
├── SESSION_SUMMARY.md             ❌ Local only
├── VALIDATOR_STATUS.md            ❌ Local only
├── TEST_SUITE_COMPLETE.md         ❌ Local only
└── NEXT_STEPS.md                  ❌ Local only
```

## Benefits of This Structure

### For Public Documentation (`docs/`)
✅ **Version controlled** - Track changes over time  
✅ **Collaborative** - Share with team/reviewers  
✅ **Polished** - User-ready content  
✅ **Stable** - Only update when needed  
✅ **Professional** - Suitable for publication/sharing  

### For Development Documentation (`dev_docs/`)
✅ **Flexible** - Update freely without commits  
✅ **Personal** - Keep your own working notes  
✅ **Temporary** - Delete when no longer needed  
✅ **Unpolished** - Working thoughts, incomplete ideas  
✅ **Clean git history** - No noise from dev notes  

## Usage Guidelines

### When to Add to `docs/`
- Documentation needed by users/collaborators
- Stable, polished content
- Mathematical/scientific foundations
- User guides and tutorials
- Testing/validation procedures

### When to Add to `dev_docs/`
- Development planning and decisions
- Migration notes and status updates
- Session summaries and working notes
- Internal design rationale
- Temporary implementation notes

### Workflow
1. **During development**: Add notes to `dev_docs/`
2. **When stable**: Extract key info → polished doc in `docs/`
3. **After implementation**: Keep or delete `dev_docs/` notes as needed

## Git Configuration

In `.gitignore`:
```gitignore
# Development documentation (internal notes, not for public repo)
dev_docs/
```

This ensures:
- `dev_docs/` is never committed
- Each developer can maintain their own notes
- Public docs remain clean and professional

## Examples

### Public Documentation Example
**`docs/MATHEMATICAL_FRAMEWORK.md`**:
- Rigorous, complete mathematical formalism
- Publication-quality LaTeX equations
- Proper citations and references
- Suitable for sharing with research community

### Development Documentation Example
**`dev_docs/MIGRATION_STATUS.md`**:
- "TODO: Fix imports in geometry module"
- "Session 3: Migrated CWT, need to test"
- "Note: β=0 for LIGO, expect β>0 for LISA?"
- Working notes, not polished

## Summary

| Aspect | `docs/` | `dev_docs/` |
|--------|---------|-------------|
| **Version Control** | ✅ Tracked | ❌ Gitignored |
| **Audience** | Public/Users | Developer only |
| **Quality** | Polished | Working notes |
| **Stability** | Stable | Evolving |
| **Collaboration** | Shared | Personal |
| **Examples** | Math framework, guides | Migration notes, TODOs |

---

**Result**: Clean separation between public-facing documentation and internal development notes, keeping the git repository professional while allowing flexible local note-taking.

