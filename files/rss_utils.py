import hashlib
from typing import Iterable, List
import numpy as np

def _stable_seed_bytes(server_round: int, a: int, b: int, layer_idx: int) -> bytes:
    # Same seed for a pair like in olympia. 
    if a > b:
        a, b = b, a
    payload = f"{server_round}:{a}:{b}:{layer_idx}".encode()
    return hashlib.sha256(payload).digest()

def _seed_to_uint64(seed_bytes: bytes) -> int:
    return int.from_bytes(seed_bytes[:8], "little", signed=False)

def _pair_sign(i: int, j: int) -> float:
    # Anti-symmetric sign: + for lower ID, - for higher ID
    return +1.0 if i < j else -1.0

def add_pairwise_noise(
    weights: List[np.ndarray],
    cid: int,
    self_id: int,
    participants: Iterable[int],
    server_round: int,
    noise_std: float,
    scale: float, 
) -> List[np.ndarray]:
 
    parts = [int(p) for p in participants]
    if noise_std <= 0.0 or len(parts) <= 1:
        return [w.astype(w.dtype, copy=True) for w in weights]

    others = [j for j in parts if j != self_id]
    if not others:
        return [w.astype(w.dtype, copy=True) for w in weights]

    masked = [w.astype(w.dtype, copy=True) for w in weights]

    for layer_index, w in enumerate(masked):
        acc = np.zeros_like(w, dtype=np.float64)

        for j in others:
            seed = _seed_to_uint64(_stable_seed_bytes(server_round, self_id, j, layer_index))
            rng = np.random.default_rng(seed)
            r = rng.normal(0.0, noise_std, size=w.shape).astype(np.float64, copy=False)
            acc += _pair_sign(self_id, j) * (r.astype(np.float64) * scale)  # scale = 1/n_i

        # adding in float64, then casting back to original dtype. 
        masked[layer_index] = (w.astype(np.float64) + acc).astype(w.dtype, copy=False)

    return masked
