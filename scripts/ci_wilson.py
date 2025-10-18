"""
Compute Wilson score interval (95% default) for accuracies.

Usage examples:
  # If you know k correct out of n
  python -m scripts.ci_wilson --k 838 --n 1000

  # If you know accuracy as a float and n
  python -m scripts.ci_wilson --p 0.838 --n 1000

  # Provide both exact_accuracy and token-level accuracy
  python -m scripts.ci_wilson --exact-p 0.838 --exact-n 1000 --token-p 0.994981 --token-n 900000

Notes:
 - This does not read any model outputs; it's a pure math helper you can run
   after eval. Provide either p (rate) with n, or k (count) with n.
 - Confidence defaults to 95% (z=1.959964). You can change via --confidence.
"""

import argparse
from typing import Tuple, Optional
import math


def z_from_confidence(conf: float) -> float:
    """Return z-score for two-sided confidence level conf (e.g., 0.95).

    Uses a hardcoded common value for 95% and 90%; falls back to 97.5% point of
    normal for others using inverse error function approximation.
    """
    if abs(conf - 0.95) < 1e-6:
        return 1.959964
    if abs(conf - 0.90) < 1e-6:
        return 1.644854
    # Approximate inverse CDF via erfcinv; conf two-sided -> tail = (1-conf)/2
    # z = sqrt(2) * erfcinv(2*tail)
    try:
        tail = (1.0 - conf) / 2.0
        # math.erfcinv is not in Python stdlib; implement via inverse erf approximation
        # Using Winitzki approximation for erfinv
        def erfinv(x: float) -> float:
            a = 0.147  # Winitzki constant
            sgn = 1 if x >= 0 else -1
            ln = math.log(1 - x*x)
            t = 2/(math.pi*a) + ln/2
            return sgn * math.sqrt( math.sqrt(t*t - ln/a) - t )

        # Convert erfcinv to erfinv: erfcinv(y) = erfinv(1 - y) / sqrt(2)
        # Here we need z for two-sided tail: z = sqrt(2) * erfcinv(2*tail)
        z = math.sqrt(2.0) * (erfinv(1 - 2*tail) / math.sqrt(2.0))
        return float(z)
    except Exception:
        return 1.959964


def wilson_ci(p: float, n: int, z: float = 1.959964) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion.

    Args:
        p: observed proportion in [0,1]
        n: number of trials (>0)
        z: z-score (1.959964 for 95% two-sided)
    Returns:
        (lo, hi): lower and upper bounds in [0,1]
    """
    if n <= 0:
        return float('nan'), float('nan')
    p = max(0.0, min(1.0, p))
    z2 = z*z
    denom = 1 + z2/n
    center = (p + z2/(2*n)) / denom
    margin = z * math.sqrt( (p*(1-p)/n) + (z2/(4*n*n)) ) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def parse_args():
    ap = argparse.ArgumentParser(description="Wilson CI calculator for accuracies")
    ap.add_argument("--confidence", type=float, default=0.95, help="Two-sided confidence level (default 0.95)")
    # General single metric
    ap.add_argument("--p", type=float, default=None, help="Observed accuracy proportion in [0,1]")
    ap.add_argument("--k", type=float, default=None, help="Number of correct samples (will be rounded)")
    ap.add_argument("--n", type=int, default=None, help="Number of samples")
    # Named pairs for convenience
    ap.add_argument("--exact-p", type=float, default=None, help="Exact accuracy proportion")
    ap.add_argument("--exact-k", type=float, default=None, help="Exact correct count")
    ap.add_argument("--exact-n", type=int, default=None, help="Exact total count")
    ap.add_argument("--token-p", type=float, default=None, help="Token-level accuracy proportion")
    ap.add_argument("--token-k", type=float, default=None, help="Token-level correct count")
    ap.add_argument("--token-n", type=int, default=None, help="Token-level total count")
    return ap.parse_args()


def _resolve_pair(p: Optional[float], k: Optional[float], n: Optional[int]) -> Optional[Tuple[float,int]]:
    if n is None:
        return None
    if p is not None:
        return float(p), int(n)
    if k is not None:
        return float(k)/int(n), int(n)
    return None


def main():
    args = parse_args()
    z = z_from_confidence(args.confidence)

    pairs = []
    # General metric first if provided
    gen = _resolve_pair(args.p, args.k, args.n)
    if gen is not None:
        pairs.append(("accuracy", gen[0], gen[1]))

    # Named metrics
    exact = _resolve_pair(args.exact_p, args.exact_k, args.exact_n)
    if exact is not None:
        pairs.append(("exact_accuracy", exact[0], exact[1]))
    token = _resolve_pair(args.token_p, args.token_k, args.token_n)
    if token is not None:
        pairs.append(("token_accuracy", token[0], token[1]))

    if not pairs:
        print("No inputs provided. Pass --p/--k and --n, or use --exact-* / --token-*.")
        return

    print(f"Wilson {int(args.confidence*100)}% confidence intervals:")
    for name, p, n in pairs:
        lo, hi = wilson_ci(p, n, z)
        print(f"- {name}: p={p:.6f}, n={n} -> CI: [{lo:.6f}, {hi:.6f}]  (half-width={(hi-lo)/2:.6f})")


if __name__ == "__main__":
    main()
