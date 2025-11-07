## Scalar toy: multiple roots and sensitivity to initialization

We solve f(x)=sin(x)=0 with Newton's method. The problem has infinitely many roots at x=kπ and illustrates:

- Convergence to different roots depending on the initial guess.
- Potential failure near points where the derivative is zero (e.g., x≈π/2, J≈0).

Run:

```bash
python PM4/apps/scalar_toy_multiple_roots/solve.py
```

Expected behavior:
- Start near 0 → converge to 0.
- Start near π → converge to π.
- Start near π/2 → line search struggles and convergence may fail or stall.



