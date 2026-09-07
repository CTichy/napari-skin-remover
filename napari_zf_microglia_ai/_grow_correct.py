"""
_grow_correct.py -- auto-growing wrapper around Correct Label / Correct
Adjacent Labels: retries the same threshold-based correction with a
progressively larger padded working region whenever the result's own
footprint touches the edge of that region -- catching real signal that
too small a pad would otherwise cut off. Also auto-detects when growth
reveals a neighboring label and expands into a joint multi-label
correction so a neighbor's territory is never wrongly consumed -- this
never grows silently into someone else's cell.

2D mode reuses correct_label_group_2d() directly (a single-ID call
degenerates to plain single-label correction, so one function already
covers both the 1-label and N-label case uniformly).

3D mode reuses the same two-pass architecture auto_contrast_correct_stack()
already uses for a whole fish (Pass 1: correct_label_from_intensity_3d()
independently per label; Pass 2: touching_groups_for_stack() +
correct_label_group_2d() wherever the independently-corrected labels now
touch) -- just scoped to the growing local group instead of every label
in the fish, and re-run from the ORIGINAL labels each attempt rather than
compounding one attempt on top of the last.

Neighbor discovery is intentionally never allowed to cascade past labels
already confirmed to intersect the group in this same attempt -- Pass 2
only ever runs once a given attempt found zero new neighbors from Pass 1
AND from the touching-group analysis; if either surfaces one, the whole
attempt is redone from scratch with the bigger group instead of applying
Pass 2 to a group that might still be missing a member.
"""

from __future__ import annotations

import numpy as np

from ._labeling import (
    correct_label_from_intensity_3d,
    correct_label_group_2d,
    touching_groups_for_stack,
    remove_debris_for_label,
)


def grow_correct_label_2d(
    labels: np.ndarray,
    image: np.ndarray,
    label_ids: "int | list[int]",
    z: int,
    lo: float,
    initial_pad: int = 15,
    growth_step: int = 15,
    max_iterations: int = 5,
    sigma: float = 1.0,
    progress_cb=None,
) -> "tuple[np.ndarray, dict]":
    """
    Auto-grows Correct Label's 2D (single-slice) correction until the
    result no longer touches the edge of its own padded working
    region, expanding the padding by growth_step each time it does.
    If the growing region starts overlapping a different label, that
    label is folded into the correction (via correct_label_group_2d(),
    the joint marker-seeded watershed correction) instead of being
    encroached on by the target label's own growth.

    labels, image  : (Z, Y, X) volumes, same shape
    label_ids       : the label(s) to correct -- a single int (Correct
                      Label's own use) or a list of 2+ ints (Correct
                      Adjacent Labels' own use, seeding the group with
                      both labels from the start instead of just one)
    z               : slice index -- only this slice is touched
    lo              : one-sided intensity cutoff (signal = image >= lo)
    initial_pad, growth_step, max_iterations : padding starts at
                      initial_pad, grows by growth_step each attempt
                      that still touches the border, up to
                      max_iterations attempts total
    sigma           : passed through to correct_label_group_2d()'s
                      watershed smoothing (irrelevant for a 1-label group)
    progress_cb      : optional callable(str)

    Returns (new_labels, report). report:
        group            -- sorted list of label ids corrected together
        pad_used          -- the padding of the last attempt actually made
        n_iterations       -- how many attempts were made
        converged          -- True if the last attempt didn't touch the border
        group_grew         -- True if growth ever pulled in a neighbor
        info               -- the underlying correct_label_group_2d()'s own
                              info dict from the last attempt

    Raises ValueError only if even the first attempt (initial_pad,
    single label) fails outright -- same errors correct_label_group_2d()
    itself raises (label not found, threshold connects to nothing).
    """
    def _report(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    group = set(label_ids) if isinstance(label_ids, (list, tuple, set)) else {int(label_ids)}
    pad = int(initial_pad)
    group_grew = False
    last_new_labels = labels
    last_info = None
    converged = False
    used_pad = pad
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        used_pad = pad
        labels_z = labels[z]
        seed = np.isin(labels_z, sorted(group))
        ys, xs = np.nonzero(seed)
        y0 = max(int(ys.min()) - used_pad, 0)
        y1 = min(int(ys.max()) + used_pad + 1, labels_z.shape[0])
        x0 = max(int(xs.min()) - used_pad, 0)
        x1 = min(int(xs.max()) + used_pad + 1, labels_z.shape[1])
        foreign_ids = {
            int(v) for v in np.unique(labels_z[y0:y1, x0:x1]) if v > 0
        } - group
        if foreign_ids:
            group |= foreign_ids
            group_grew = True
            _report(f"Growing group to include neighbor(s) {sorted(foreign_ids)} -> {sorted(group)}")

        _report(f"Attempt {iteration}: pad={used_pad}px, group={sorted(group)}")
        new_labels, info = correct_label_group_2d(
            labels, image, sorted(group), z, lo, pad=used_pad, sigma=sigma,
        )
        last_new_labels, last_info = new_labels, info

        if not info["touched_border"]:
            converged = True
            break
        pad += growth_step

    report = {
        "group": sorted(group),
        "pad_used": used_pad,
        "n_iterations": iteration,
        "converged": converged,
        "group_grew": group_grew,
        "info": last_info,
    }
    return last_new_labels, report


def grow_correct_label_3d(
    labels: np.ndarray,
    image: np.ndarray,
    label_ids: "int | list[int]",
    lo: float,
    initial_pad: int = 15,
    growth_step: int = 15,
    max_iterations: int = 5,
    sigma: float = 1.0,
    min_volume: "int | None" = None,
    final_min_fraction: float = 0.618,
    progress_cb=None,
) -> "tuple[np.ndarray, dict]":
    """
    3D analogue of grow_correct_label_2d(). See the module docstring
    for the two-pass architecture (independent-per-label 3D walk, then
    joint re-derivation wherever the group ends up touching).

    Neighbor discovery here is reactive rather than proactive (unlike
    the 2D version): 3D's true Z-extent isn't known ahead of a walk the
    way a 2D slice's XY bbox is, so there is nothing to pre-scan before
    running Pass 1 at least once. A neighbor found via either Pass 1's
    own foreign_touching/foreign_nearby reports, or via a touching group
    (Pass 2 candidate) that includes a label outside the current group,
    causes the WHOLE attempt to be redone from the original labels with
    the group expanded -- Pass 2 never runs on a group that might still
    be missing a member.

    Returns (new_labels, report): group, pad_used, n_iterations,
    converged, group_grew, per_label_reports (each group member's own
    correct_label_from_intensity_3d() report from the final attempt,
    keyed by label id).

    Raises ValueError only if even the first attempt's single-label
    correction fails outright (same errors correct_label_from_intensity_3d()
    itself raises).
    """
    def _report(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    group = set(label_ids) if isinstance(label_ids, (list, tuple, set)) else {int(label_ids)}
    pad = int(initial_pad)
    group_grew = False
    last_new_labels = labels
    last_per_label_reports: "dict[int, dict]" = {}
    converged = False
    used_pad = pad
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        used_pad = pad
        working = labels
        per_label_reports: "dict[int, dict]" = {}
        new_neighbors: "set[int]" = set()

        _report(f"Attempt {iteration}: pad={used_pad}px, group={sorted(group)} -- Pass 1 (independent per label)...")
        for lid in sorted(group):
            working, rep = correct_label_from_intensity_3d(
                working, image, lid, lo, pad=used_pad,
                min_volume=min_volume, final_min_fraction=final_min_fraction,
            )
            per_label_reports[lid] = rep
            for ids in rep["foreign_touching"].values():
                new_neighbors.update(ids)
            for ids in rep["foreign_nearby"].values():
                new_neighbors.update(ids)
        new_neighbors -= group

        _report(f"Attempt {iteration}: Pass 2 (touching-group check)...")
        groups_by_z = touching_groups_for_stack(working)
        pass2_jobs: "list[tuple[int, list[int]]]" = []
        for z, tgroups in groups_by_z.items():
            for tgroup in tgroups:
                inter = set(tgroup) & group
                if not inter:
                    continue
                extra = set(tgroup) - group
                if extra:
                    new_neighbors.update(extra)
                    continue
                pass2_jobs.append((z, sorted(tgroup)))

        if new_neighbors:
            group |= new_neighbors
            group_grew = True
            _report(f"Growing group to include neighbor(s) {sorted(new_neighbors)} -> {sorted(group)}, redoing this attempt")
            last_new_labels = working
            last_per_label_reports = per_label_reports
            continue

        for z, tgroup in pass2_jobs:
            try:
                working, _info = correct_label_group_2d(
                    working, image, tgroup, z, lo, pad=used_pad, sigma=sigma,
                )
            except ValueError:
                pass  # same tolerant skip auto_contrast_correct_stack's own Pass 2 uses

        last_new_labels = working
        last_per_label_reports = per_label_reports

        any_touched_border = any(r["touched_border"] for r in per_label_reports.values())
        if not any_touched_border:
            converged = True
            break
        pad += growth_step

    report = {
        "group": sorted(group),
        "pad_used": used_pad,
        "n_iterations": iteration,
        "converged": converged,
        "group_grew": group_grew,
        "per_label_reports": last_per_label_reports,
    }

    # Final whole-group debris cleanup -- same golden-ratio safety net as
    # every other final-safety-net stage in this plugin (see
    # auto_contrast_correct_stack()'s own step 5). Pass 1's own per-label
    # cleanup already ran, but Pass 2's watershed cuts can still leave a
    # small disconnected sliver behind at a cut that Pass 1 never saw
    # (it ran before Pass 2 touched that boundary). Scoped to just this
    # group's own labels, not the whole fish.
    n_debris_removed_total = 0
    if min_volume is not None:
        threshold = final_min_fraction * min_volume
        for lid in sorted(group):
            last_new_labels, n_removed = remove_debris_for_label(last_new_labels, lid, threshold)
            n_debris_removed_total += n_removed
    report["n_debris_removed_px"] = n_debris_removed_total

    return last_new_labels, report


def format_grow_report(report: dict, mode: str) -> str:
    lines = []
    group = report["group"]
    lines.append(
        f"Auto-grow ({mode}): {report['n_iterations']} attempt(s), final pad={report['pad_used']}px, "
        f"group={group}{' (grew from neighbor discovery)' if report['group_grew'] else ''}"
    )
    if report["converged"]:
        lines.append("  Converged -- no part of the result touches the padded region's own edge.")
    else:
        lines.append(
            "  NOT converged -- signal still reaches the padded region's edge after "
            f"{report['n_iterations']} attempt(s). Real signal may extend further; "
            "consider a larger starting pad or more max iterations, or correct this cell by hand."
        )
    if "n_debris_removed_px" in report:
        lines.append(f"  Debris removed: {report['n_debris_removed_px']} px")
    return "\n".join(lines)
