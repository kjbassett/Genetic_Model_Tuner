"""Charts drawn from a run's history.

Kept apart from model_tuner so that matplotlib is imported only when a chart is
actually drawn, and so a failure to draw one can never take down a run that has
already finished computing.
"""

import logging
import os

_log = logging.getLogger(__name__)

GENERATION_SCORES_FILE = "generation_scores.png"


def spread(n, width=0.6):
    """Evenly spaced x-offsets for n points sharing one generation.

    Deterministic rather than random: an elite carried forward plots at an
    identical score in consecutive generations, and random jitter would let it
    land back under itself. Spacing by position also makes an organism's index
    readable off the chart.

    Args:
        n: How many points share the x position.
        width: Total horizontal span to spread them across.

    Returns:
        A list of n offsets, centred on zero.
    """
    if n < 2:
        return [0.0] * n
    step = width / (n - 1)
    return [i * step - width / 2 for i in range(n)]


def plot_generation_scores(generation_history, folder,
                           filename=GENERATION_SCORES_FILE, goal="max",
                           title=None):
    """Scatter every organism's score against its generation.

    One point per organism per generation, plus a line through the best of each
    generation -- the question the chart answers is whether selection is
    improving anything, which the cloud alone does not show.

    Args:
        generation_history: One list of organism entries per generation, each
            entry carrying "generation", "organism_index" and "score".
        folder: Directory to write the image into.
        filename: Name of the image file.
        goal: "max" or "min", deciding which end of a generation is its best.
        title: Heading for the chart. Defaults to the folder's name.

    Returns:
        Path to the written image, or None if there was nothing to draw or
        matplotlib is not installed.
    """
    generations = [[e for e in gen if e.get("score") is not None]
                   for gen in generation_history]
    generations = [gen for gen in generations if gen]
    if not generations:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")  # No display on a training box.
        import matplotlib.pyplot as plt
    except ImportError:
        _log.warning("matplotlib is not installed; skipping %s", filename)
        return None

    pick = max if goal == "max" else min
    xs, ys, bests = [], [], []
    for gen in generations:
        n = gen[0]["generation"]
        points = [(n + offset, entry["score"])
                  for entry, offset in zip(gen, spread(len(gen)))]
        xs += [x for x, _ in points]
        ys += [y for _, y in points]
        # At the winner's own x, not the generation's tick: a marker floating
        # beside the dot it describes invites reading it as another organism.
        bests.append(pick(points, key=lambda point: point[1]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, s=36, alpha=0.75, edgecolors="none",
               label="organism", zorder=3)
    ax.plot([x for x, _ in bests], [y for _, y in bests],
            color="tab:red", marker="o", markersize=5, linewidth=1.5,
            label=f"best of generation ({goal})", zorder=4)
    ax.axhline(0, color="0.6", linewidth=0.8, zorder=1)

    ax.set_xlabel("generation")
    ax.set_ylabel("score")
    ax.set_title(title or os.path.basename(os.path.normpath(folder)) or "run")
    ax.set_xticks([gen[0]["generation"] for gen in generations])
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()

    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
