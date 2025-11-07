import numpy as np
import matplotlib.pyplot as plt


def plot_residual_histories(histories, labels, title=None, outfile=None, show=True):
    """
    Plot residual norms over iterations on a semilogy scale.
    histories: list of sequences of (iter_index, residual_norm)
    labels: list of legend labels (same length as histories)
    title: optional title string
    outfile: optional path to save PNG
    show: whether to call plt.show()
    """
    if len(histories) != len(labels):
        raise ValueError("histories and labels must have same length")
    plt.figure(figsize=(6, 4))
    for hist, lab in zip(histories, labels):
        iters = [k for (k, _) in hist]
        norms = [max(1e-16, float(r)) for (_, r) in hist]
        plt.semilogy(iters, norms, '-o', ms=3, label=lab)
    plt.xlabel('Iteration')
    plt.ylabel('Residual norm ||f||')
    if title:
        plt.title(title)
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=150)
    if show:
        plt.show()



