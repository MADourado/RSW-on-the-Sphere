"""Wave-set-level wrapper around ``periods.novel_frequency_content``:
given ``run_dynamics.run_dynamics()``'s own result dict, find every
sub-triad unit that contains a given target mode (1 for a private mode,
2+ for a mode shared across triads -- member or sum role alike, no
special-casing needed) and run the novelty-detection comparison against
each.

Kept free of integration/caching concerns on purpose: this module never
integrates anything itself, it only consumes ``results`` -- the exact
shape ``run_dynamics()`` already returns (``{'full': {...},
'triad_<member1>_<member2>': {...}, ...}``, each with ``t``/``E``/``labels``).
"""
from rsw_sphere.utilities.periods import novel_frequency_content_multi


def novelty_combined_for_target(results: dict, target_label: str, **kwargs) -> dict:
    """Single combined novelty-detection result for ``target_label``
    against ALL of its containing sub-triads at once (see
    ``periods.novel_frequency_content_multi``) -- one result per target,
    not one per (target, sub-triad) pair.

    Returns
    -------
    dict
        ``novel_frequency_content_multi(...)``'s own result dict, plus
        ``sub_names`` (list of containing sub-unit names, in the same
        order as the ``subs`` sequence passed internally -- a plotting
        caller needs this to know which units to draw).
    """
    full = results["full"]
    if target_label not in full["labels"]:
        raise ValueError(f"{target_label!r} not found in the full wave set "
                          f"(available: {full['labels']})")
    j_full = full["labels"].index(target_label)
    t_full, E_full = full["t"], full["E"][:, j_full]

    sub_names = [name for name, unit in results.items()
                 if name != "full" and target_label in unit["labels"]]
    if not sub_names:
        raise ValueError(f"{target_label!r} not found in any sub-triad")
    subs = [(results[name]["t"], results[name]["E"][:, results[name]["labels"].index(target_label)])
            for name in sub_names]

    result = novel_frequency_content_multi(t_full, E_full, subs, **kwargs)
    result["sub_names"] = sub_names
    return result


def novelty_combined_for_all_targets(results: dict, **kwargs) -> dict:
    """``novelty_combined_for_target`` for every mode in the full wave set.

    Returns
    -------
    dict
        ``{target_label: result, ...}``.
    """
    return {label: novelty_combined_for_target(results, label, **kwargs)
            for label in results["full"]["labels"]}
