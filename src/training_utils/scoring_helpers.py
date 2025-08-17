@dataclass(frozen=True) #immutable, default methods initiated
class Scores:
    ctc: float
    wer: float

def _is_better(curr: float, best: float, mode: str) -> bool:
    """
    Return True iff `curr` is better than `best` for the given attack mode.
    For targeted: we want lower WER on perturbed (hit the target).
    For untargeted: we want higher CTC loss on perturbed (degrade ASR).
    """
    if mode == "targeted":
        return curr < best
    if mode == "untargeted":
        return curr > best
    raise ValueError(f"Unknown attack_mode: {mode!r}")

def _best_agg(values: list[float], mode: str) -> float:
    """Min for targeted, max for untargeted."""
    if not values:
        return float("inf") if mode == "targeted" else float("-inf")
    return (min if mode == "targeted" else max)(values)