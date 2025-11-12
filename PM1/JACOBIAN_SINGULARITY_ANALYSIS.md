# Jacobian Singularity and Condition Number Analysis

## Summary

This document analyzes where the Jacobian becomes singular (or near-singular) and explains the condition number magnitudes observed in the bacterial evolution model.

## Key Findings

### Condition Number Magnitudes

From systematic analysis across parameter space (R, C) and along trajectories:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Typical range** | 10² - 10⁴ | Well to moderately conditioned |
| **Median (trajectory)** | ~2,000 | Moderate conditioning, Newton should work |
| **Maximum observed** | ~7×10⁴ | Poorly conditioned, may cause numerical issues |
| **Near-singular threshold** | >10¹⁰ | Effectively singular (not observed in typical ranges) |

### Where the Jacobian Becomes Singular

The Jacobian becomes **worst-conditioned** (closest to singular) at:

1. **Low Resource, No Antibiotic** (R ≈ 0.27, C = 0)
   - Condition number: ~7×10⁴
   - Min singular value: ~9.5×10⁻⁵
   - **Why**: Low resource → birth rates → 0 → system becomes insensitive to population changes

2. **Very Low Resource** (R → 0)
   - Condition numbers: 500-600
   - **Why**: Monod term R/(K+R) → 0, decoupling population dynamics

3. **Trajectory Analysis** (along solver path)
   - Min condition: ~100 (well-conditioned)
   - Median: ~2,000 (moderate)
   - Max: ~25,000 (poorly conditioned)
   - Worst point: R ≈ 0.09, C ≈ 0.43

### Condition Number Scale Interpretation

| Condition Number | Status | Numerical Impact | Biological Meaning |
|-----------------|--------|------------------|-------------------|
| **10¹ - 10²** | Excellent | No numerical issues | Healthy, stable dynamics |
| **10² - 10³** | Good | Minor numerical issues possible | Normal operating regime |
| **10³ - 10⁴** | Moderate | May need careful tolerances | Transitioning states |
| **10⁴ - 10⁶** | Poor | Newton may converge slowly | Near-extinction or extreme stress |
| **10⁶ - 10¹²** | Very Poor | Newton may fail, need regularization | Degenerate system |
| **>10¹²** | Singular | System is degenerate | Multiple solutions or no solution |

## Intuitive Explanations

### 1. Low Resource (R → 0)

**What happens:**
- Monod term: `R/(K+R) → 0` as R → 0
- Birth rates: `b = rmax × monod × hill → 0`
- The `∂f/∂n` block becomes nearly zero (no growth sensitivity)

**Why this causes singularity:**
- When birth rates are zero, small changes in populations don't affect the dynamics
- The system becomes "stiff" - insensitive to perturbations
- This reduces the effective rank of the Jacobian

**Biological intuition:**
- Near starvation, the system is in a "frozen" state
- Populations can't grow, so their exact values matter less
- The system is transitioning between survival and extinction

### 2. High Antibiotic (C >> IC50)

**What happens:**
- Hill term: `1/(1 + (C/IC50)^h) → 0` as C >> IC50
- Birth rates: `b → 0` (complete inhibition)
- Similar decoupling as low resource

**Why this causes singularity:**
- Complete growth inhibition makes population dynamics insensitive
- The system becomes deterministic (only death, no birth)
- Reduced sensitivity to initial conditions

**Biological intuition:**
- Under lethal antibiotic pressure, all populations decline
- Exact population values less important than survival threshold
- System is in "extinction mode"

### 3. Zero Populations (n → 0)

**What happens:**
- Coupling terms vanish: `n × db/dR → 0`, `n × db/dC → 0`
- Resource and antibiotic equations decouple from populations
- The Jacobian loses coupling between blocks

**Why this causes singularity:**
- When populations are zero, they can't affect resource/antibiotic dynamics
- The system effectively reduces dimension
- Information about population-resource coupling is lost

**Biological intuition:**
- At extinction, the system simplifies (no consumers)
- Resource and antibiotic evolve independently
- The model becomes less informative about population dynamics

### 4. Extreme Parameter Combinations

**What happens:**
- Multiple mechanisms act simultaneously (low R AND high C)
- Multiple blocks become near-zero
- Rank deficiency can occur

**Why this causes singularity:**
- Multiple sources of decoupling compound
- The Jacobian loses information about multiple directions
- System becomes underdetermined

**Biological intuition:**
- Extreme stress creates "brittle" states
- Multiple failure modes simultaneously
- System behavior becomes unpredictable

## Mathematical Structure

The Jacobian has block structure:

```
J = [∂f/∂n  |  ∂f/∂R  |  ∂f/∂C ]
    [∂R/∂n  |  ∂R/∂R  |  ∂R/∂C ]
    [∂C/∂n  |  ∂C/∂R  |  ∂C/∂C ]
```

**When singularity occurs:**
- `∂f/∂n` block → 0 (low birth rates)
- Coupling terms `∂f/∂R`, `∂f/∂C` → 0 (zero populations or zero birth rates)
- This creates rank deficiency

## Practical Implications

### For Newton's Method

1. **Well-conditioned (κ < 10³)**: Newton converges quickly and reliably
2. **Moderate (10³ < κ < 10⁴)**: Newton works but may need tighter tolerances
3. **Poor (10⁴ < κ < 10⁶)**: Newton may converge slowly or need line search
4. **Very poor (κ > 10⁶)**: Newton may fail; consider:
   - Regularization (add small diagonal)
   - Homotopy continuation
   - Alternative solvers (GMRES, etc.)

### For Numerical Stability

- **Condition number < 10⁸**: Standard double precision sufficient
- **Condition number > 10⁸**: May need:
  - Higher precision arithmetic
  - Regularization
  - Specialized algorithms

### For Biological Interpretation

High condition numbers indicate:
- **Brittle states**: Small perturbations cause large changes
- **Non-unique solutions**: Multiple states are nearly equivalent
- **Regime transitions**: System is between stable states (e.g., survival ↔ extinction)
- **Reduced predictability**: System behavior is sensitive to numerical errors

## Recommendations

1. **Monitor condition numbers** during simulations
2. **Use adaptive tolerances** based on condition number
3. **Apply regularization** when κ > 10⁶
4. **Interpret results carefully** when κ > 10⁴ (system may be in transition)
5. **Consider alternative formulations** for extreme parameter regimes

## References

- Analysis performed by `analyze_jacobian_singularity.py`
- Trajectory analysis from `jacobian_condition_checks.py`
- Typical condition numbers: 100-25,000 along trajectories
- Worst-conditioned regions: Low resource (R < 0.3), especially with C ≈ 0

