"""Bitshift trellis quantizer (QTIP) for super-entry GPTQ-2D.

A trellis state is an L-bit window of a bitstream.  Each step shifts K*V new
bits into the state and emits the V values ``table[state, :]``, so T values cost
T*K bits.  The Viterbi algorithm finds the bitstream whose values are closest
to the input in Euclidean distance.  Tail-biting makes the bitstream circular
with two passes, which need not find the closest circular bitstream.
The codes 1MAD, 3INST, and HYB turn states into roughly Gaussian values, as in
QTIP (Tseng et al., 2024).

``trellis_quantizer_super_entry_torch`` and ``trellis_quantizer_super_entry_triton``
round each super-entry, read row by row, as one sequence and write ``scale``
times its trellis values, from which the T*K-bit code can be recovered.  The
Triton version needs ``y_flat``, ``scale``, and ``table`` on one CUDA device.
It returns the same values, never synchronizes with the host, and so runs in
CUDA graphs.
"""

from collections.abc import Callable
import itertools
import unittest

import torch
import triton
import triton.language as tl


def one_mad(L: int = 16, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
    """1MAD code: one multiply-add step, then the sum of the four bytes, in float32, returned in ``dtype``."""
    x = (torch.arange(1 << L, dtype=torch.int64, device=device) * 34038481 + 76625530) & 0xFFFFFFFF  # [2^L]
    x = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)  # [2^L]
    divisor = x.new_tensor(147.800537109375, dtype=torch.float32)  # not a Python scalar, which CUDA divides by through its reciprocal
    return ((x - 510).float() / divisor)[:, None].to(dtype=dtype)  # [2^L, 1]


def three_inst(L: int = 16, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
    """3INST code: one multiply-add step, then two masked fp16 halves summed, returned in ``dtype`` (float32 by default)."""
    x = ((torch.arange(1 << L, dtype=torch.int64, device=device) * 89226354 + 64248484) & 0x8FFF8FFF) ^ 0x3B603B60  # [2^L]
    halves = torch.stack((x >> 16, x & 0xFFFF), dim=-1)  # [2^L, 2], fp16 bit patterns
    halves = torch.where(halves >= 1 << 15, halves - (1 << 16), halves).to(torch.int16).view(torch.float16)
    return (halves[:, 0] + halves[:, 1]).float()[:, None].to(dtype=dtype)  # [2^L, 1], summed in fp16


def _gaussian_codebook(
    size: int, seed: int = 0, samples: int = 1 << 16, iters: int = 25, device: torch.device | str | None = None,
) -> torch.Tensor:
    """k-means codebook of the 2-D standard Gaussian, rescaled as in QTIP, computed on ``device``.

    The seeded points come from the CPU on every device, and each cluster sums its points in their
    order, so the codebook is deterministic on CUDA too, where atomic adds would not be.
    """
    generator = torch.Generator().manual_seed(seed)
    data = torch.randn(samples, 2, generator=generator).to(device)  # [samples, 2]
    centers = torch.randn(size, 2, generator=generator).to(device)  # [size, 2]
    for _ in range(iters):
        nearest = torch.cdist(data, centers).argmin(dim=1)  # [samples]
        counts = nearest.bincount(minlength=size)  # [size]
        sums = torch.segment_reduce(data[nearest.argsort(stable=True), :], "sum", lengths=counts, axis=0)  # [size, 2]
        centers = torch.where(counts[:, None] > 0, sums / counts[:, None].clamp_min(1), centers)  # [size, 2]
    return centers / centers.std(correction=0) * (15 / 16) ** .5  # population std sqrt(15/16)


def hyb(
    L: int = 16, tlut: torch.Tensor | None = None, seed: int = 0,
    device: torch.device | str | None = None, dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """HYB code: hash each state to a row of a 2-D lookup table ``tlut`` and
    flip the sign of its first value by one more hash bit.  The default
    ``tlut`` is a seeded float32 k-means codebook with 2^9 rows, computed on
    ``device``; pass a trained one to reproduce a QTIP checkpoint.  The table
    is made where ``tlut`` is and returned in ``dtype``, by default
    ``tlut``'s."""
    tlut = _gaussian_codebook(1 << 9, seed=seed, device=device) if tlut is None else tlut  # [2^tlut_bits, 2]
    tlut_bits = tlut.size(0).bit_length() - 1
    x = torch.arange(1 << L, dtype=torch.int64, device=tlut.device)
    x = x * (x + 1)  # [2^L], the hash
    table = tlut[(x >> (15 - tlut_bits)) & ((1 << tlut_bits) - 1), :]  # [2^L, 2]
    table[:, 0] *= 1 - 2 * ((x >> 15) & 1)
    return table.to(dtype=dtype)


def _viterbi(
    x: torch.Tensor,
    table: torch.Tensor,
    K: int,
    init: torch.Tensor | None = None,
    final: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the states [..., T/V] of the trellis path closest to x [..., T].

    State s goes to (s << K*V | bits) mod 2^L, so the predecessors of state s
    are h * 2^(L-K*V) + (s >> K*V) for h < 2^(K*V).  The optional boolean
    masks ``init`` and ``final`` [..., 2^L] restrict the first and last states.
    Ties go to the first index.
    """
    num_states, V = table.shape
    shift = K * V
    table = table.to(x)  # [2^L, V]
    xs = x.unflatten(-1, (-1, V))  # [..., T/V, V]

    def local_cost(t: int) -> torch.Tensor:
        return (table - xs[..., t, None, :]).square().sum(dim=-1)  # [..., 2^L]

    cost = local_cost(0)  # [..., 2^L]
    if init is not None:
        cost = cost.masked_fill(~init, torch.inf)
    choices = []
    for t in range(1, xs.size(-2)):
        best, h = cost.unflatten(-1, (1 << shift, num_states >> shift)).min(dim=-2)  # [..., 2^(L-K*V)]
        choices.append(h.to(torch.uint8) if shift <= 8 else h)  # h < 2^(K*V): uint8 when it fits, an eighth of int64's memory
        cost = best.repeat_interleave(1 << shift, dim=-1) + local_cost(t)  # [..., 2^L]
    if final is not None:
        cost = cost.masked_fill(~final, torch.inf)
    states = [cost.argmin(dim=-1)]  # backtrack from the last state, [...] each
    for h in reversed(choices):
        low = states[-1] >> shift  # [...]
        states.append(low.add(h.gather(-1, low[..., None])[..., 0], alpha=num_states >> shift))  # h * 2^(L-K*V) + low in int64
    return torch.stack(states[::-1], dim=-1)  # [..., T/V]


def _trellis_quantize_torch(
    x: torch.Tensor,
    table: torch.Tensor,
    K: int,
    tail_biting: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the sequences x [..., T]; return the values [..., T] and states [..., T/V].

    Tail-biting makes each bitstream circular, exactly T*K bits: a first pass
    on x rotated by half a sequence picks the L-K*V bits that the last state
    shares with the first, which constrain the second pass.
    """
    num_states, V = table.shape
    shift = K * V
    init = final = None
    if tail_biting:
        half = x.size(-1) // V // 2
        overlap = _viterbi(x.roll(half * V, dims=-1), table, K)[..., half, None] >> shift  # [..., 1]
        state = torch.arange(num_states, device=x.device)  # [2^L]
        init = (state >> shift) == overlap  # [..., 2^L]
        final = (state % (num_states >> shift)) == overlap  # [..., 2^L]
    states = _viterbi(x, table, K, init, final)  # [..., T/V]
    return table.to(x)[states, :].flatten(start_dim=-2), states


@triton.jit
def _viterbi_step_kernel(
    x_ptr, table_ptr, prev_ptr, overlap_ptr, best_ptr, t, rows, steps, roll,
    L: tl.constexpr, KV: tl.constexpr, V: tl.constexpr, FIRST: tl.constexpr, MASKED: tl.constexpr,
    BLOCK_B: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_J: tl.constexpr, W: tl.constexpr,
):
    """One step for BLOCK_B rows b and BLOCK_J groups j: best[b, j] is the min over h of cost[h * 2^(L-KV) + j]
    at step t, read from x rotated by ``roll`` steps, where cost[s] = prev[b, s >> KV] + |table[s, :] - x[b, t, :]|^2,
    or the local cost alone at t = 0, masked by (s >> KV) == overlap[b] if MASKED.  With j = g * W + w and
    W = min(2^(L-KV), 2^KV), s >> KV does not depend on w, so each prev value is loaded once for W groups, and each
    table entry once for all BLOCK_B rows."""
    G: tl.constexpr = 1 << (L - KV)
    BLOCK_G: tl.constexpr = BLOCK_J // W
    b = (tl.program_id(1) * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)  # [BLOCK_B]
    in_rows = b < rows
    g = tl.program_id(0) * BLOCK_G + tl.arange(0, BLOCK_G)  # [BLOCK_G]
    j = g[:, None] * W + tl.arange(0, W)[None, :]  # [BLOCK_G, W]
    offset = (b * steps + (t + roll) % steps) * V  # [BLOCK_B]
    x0 = tl.load(x_ptr + offset, mask=in_rows, other=0.)[:, None, None, None]
    x1 = tl.load(x_ptr + offset + 1, mask=in_rows & (V == 2), other=0.)[:, None, None, None]
    overlap = tl.load(overlap_ptr + b, mask=in_rows & FIRST & MASKED, other=0)[:, None, None, None]
    best = tl.full([BLOCK_B, BLOCK_G, W], float("inf"), x_ptr.dtype.element_ty)
    for h0 in range(0, 1 << KV, BLOCK_H):
        h = h0 + tl.arange(0, BLOCK_H)  # [BLOCK_H]
        s = h[:, None, None] * G + j[None, :, :]  # [BLOCK_H, BLOCK_G, W], the predecessors
        d = tl.load(table_ptr + s * V)[None, :, :, :] - x0
        cost = d * d  # [BLOCK_B, BLOCK_H, BLOCK_G, W], summed as torch sums over V <= 2 values
        if V == 2:
            d = tl.load(table_ptr + s * V + 1)[None, :, :, :] - x1
            cost = cost + d * d
        if FIRST:
            if MASKED:
                cost = tl.where((s >> KV)[None, :, :, :] == overlap, cost, float("inf"))
        else:
            low = (h[:, None] * G + g[None, :] * W) >> KV  # [BLOCK_H, BLOCK_G], s >> KV for every w
            cost = tl.load(prev_ptr + b[:, None, None] * G + low[None, :, :], mask=in_rows[:, None, None], other=0.)[:, :, :, None] + cost
        best = tl.minimum(best, tl.min(cost, axis=1))
    tl.store(best_ptr + b[:, None, None] * G + j[None, :, :], best, mask=in_rows[:, None, None])


@triton.jit
def _viterbi_final_kernel(
    x_ptr, table_ptr, prev_ptr, overlap_ptr, state_ptr, steps, roll,
    L: tl.constexpr, KV: tl.constexpr, V: tl.constexpr, HAS_PREV: tl.constexpr, MASKED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """state[b]: the first argmin over s of the cost at the last step, which is also the first step without
    HAS_PREV, masked by (s & (2^(L-KV) - 1)) == overlap[b] if MASKED."""
    G: tl.constexpr = 1 << (L - KV)
    b = tl.program_id(0).to(tl.int64)
    offset = (b * steps + (steps - 1 + roll) % steps) * V
    x0 = tl.load(x_ptr + offset)
    x1 = tl.load(x_ptr + offset + 1, mask=V == 2, other=0.)
    overlap = tl.load(overlap_ptr + b, mask=MASKED, other=0)
    best = tl.full([BLOCK], float("inf"), x_ptr.dtype.element_ty)
    arg = tl.zeros([BLOCK], tl.int32)
    for s0 in range(0, 1 << L, BLOCK):
        s = s0 + tl.arange(0, BLOCK)  # [BLOCK]
        d = tl.load(table_ptr + s * V) - x0
        cost = d * d
        if V == 2:
            d = tl.load(table_ptr + s * V + 1) - x1
            cost = cost + d * d
        if HAS_PREV:
            cost = tl.load(prev_ptr + b * G + (s >> KV)) + cost
        elif MASKED:
            cost = tl.where((s >> KV) == overlap, cost, float("inf"))
        if MASKED:
            cost = tl.where((s & (G - 1)) == overlap, cost, float("inf"))
        better = cost < best  # each lane keeps its first minimum
        best = tl.where(better, cost, best)
        arg = tl.where(better, s, arg)
    smallest = tl.min(best, axis=0)
    tl.store(state_ptr + b, tl.min(tl.where(best == smallest, arg, 1 << L), axis=0).to(tl.int64))


@triton.jit
def _viterbi_backtrack_kernel(
    x_ptr, table_ptr, history_ptr, state_ptr, scale_ptr, out_ptr, states_ptr, overlap_ptr, rows, steps, stop, roll, stride,
    L: tl.constexpr, KV: tl.constexpr, V: tl.constexpr, MASKED: tl.constexpr, OVERLAP: tl.constexpr,
    STATES: tl.constexpr, BLOCK_B: tl.constexpr,
):
    """Walk back from state[b] at the last step to step ``stop``.  The state at step t - 1 is the first of the
    candidates h * 2^(L-KV) + (state >> KV) of least cost at step t - 1, computed as the step kernel computes it,
    from the group minima of step t - 2 in history[t - 2] (``stride`` elements apart).  Store overlap[b] =
    state >> KV at step ``stop``, or write out[b, t * V + v] = table[state, v] * scale[b], and states[b, t] if
    STATES, at every step."""
    G: tl.constexpr = 1 << (L - KV)
    b = (tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)  # [BLOCK_B]
    in_rows = b < rows
    state = tl.load(state_ptr + b, mask=in_rows, other=0)  # [BLOCK_B]
    scale = tl.load(scale_ptr + b, mask=in_rows & (not OVERLAP), other=1.)
    overlap = tl.load(overlap_ptr + b, mask=in_rows & MASKED, other=0)[:, None]
    h = tl.arange(0, 1 << KV)[None, :]  # [1, 2^KV]
    for i in range(steps - 1 - stop):
        t = steps - 1 - i  # the state is at step t
        if not OVERLAP:
            for v in tl.static_range(V):
                value = tl.load(table_ptr + state * V + v, mask=in_rows, other=0.)
                tl.store(out_ptr + (b * steps + t) * V + v, value * scale, mask=in_rows)
            if STATES:
                tl.store(states_ptr + b * steps + t, state, mask=in_rows)
        low = state >> KV  # [BLOCK_B]
        s = h * G + low[:, None]  # [BLOCK_B, 2^KV], the candidates at step t - 1
        offset = (b * steps + (t - 1 + roll) % steps) * V  # [BLOCK_B]
        d = tl.load(table_ptr + s * V, mask=in_rows[:, None], other=0.) - tl.load(x_ptr + offset, mask=in_rows, other=0.)[:, None]
        cost = d * d
        if V == 2:
            d = tl.load(table_ptr + s * V + 1, mask=in_rows[:, None], other=0.) - tl.load(x_ptr + offset + 1, mask=in_rows, other=0.)[:, None]
            cost = cost + d * d
        # The minima added at step t - 1 are those of step t - 2; at step 0 (t = 1) none, and 0 + cost == cost.
        cost = tl.load(history_ptr + (t - 2).to(tl.int64) * stride + b[:, None] * G + (s >> KV), mask=in_rows[:, None] & (t > 1), other=0.) + cost
        if MASKED:
            cost = tl.where((t > 1) | ((s >> KV) == overlap), cost, float("inf"))
        state = tl.argmin(cost, axis=1, tie_break_left=True).to(tl.int64) * G + low
    if OVERLAP:
        tl.store(overlap_ptr + b, state >> KV, mask=in_rows)
    else:
        for v in tl.static_range(V):
            value = tl.load(table_ptr + state * V + v, mask=in_rows, other=0.)
            tl.store(out_ptr + (b * steps + stop) * V + v, value * scale, mask=in_rows)
        if STATES:
            tl.store(states_ptr + b * steps + stop, state, mask=in_rows)


def _trellis_quantize_triton(
    x: torch.Tensor,
    table: torch.Tensor,
    K: int,
    tail_biting: bool,
    *,
    out: torch.Tensor,
    scale: torch.Tensor,
    states: torch.Tensor | None = None,
) -> None:
    """``_trellis_quantize_torch`` with Triton: write out [..., T] = scale[..., None] times
    the values, and the states [..., T/V] into ``states`` if given, both contiguous.

    A state's cost is the best cost of its group of predecessors plus its own
    local cost, so each step keeps only the 2^(L-K*V) group minima, recomputing
    the 2^L costs from the table.  The minima of every step are kept, and the
    walk back recomputes, at each step, the costs of the one group on the path
    and takes the first least, the predecessor that the search would have
    chosen.  The first pass of tail-biting reads x rotated by half a sequence
    through indices, and both passes stay on the device.
    """
    num_states, V = table.shape
    T, shift = x.size(-1), K * V
    assert V <= 2 and shift <= 8 and T % V == 0, "the same sums need V <= 2, and a walk back holds 2^(K*V) <= 256 candidates"
    assert x.is_cuda and table.device == x.device and scale.device == x.device, (
        "the Triton trellis quantizer requires Y, scale, and table on one CUDA device"
    )
    x = x.reshape(-1, T).contiguous()  # [B, T], one sequence per row
    rows = x.size(0)
    if rows == 0:
        return
    L, G, steps = num_states.bit_length() - 1, num_states >> shift, T // V
    out, scale = out.view(rows, T), scale.reshape(rows).contiguous()  # [B, T], [B]
    states = None if states is None else states.view(rows, steps)  # [B, T/V]
    table = table.to(x).contiguous()  # [2^L, V]
    # The scratch buffer is a prefix of an allocation sized for a power of two of rows.  GPTQ-2D's sweep calls this on
    # antidiagonals of every length.  Captured in a CUDA graph, a freed buffer stays reserved in the graph's memory
    # pool and serves only later requests that fit in it, so with exact sizes the growing antidiagonals would keep
    # reserving new buffers, while powers of two reserve less than twice the largest: 2 GiB at 8192 x 8192 with
    # 16 x 16 super-entries.
    capacity = 1 << (rows - 1).bit_length()  # rows rounded up to a power of two
    slots = max(steps - 1, 1)
    history = x.new_empty(slots * capacity * G)[:slots * rows * G].view(slots, rows, G)  # [T/V-1, B, 2^(L-K*V)], each step's group minima
    state = x.new_empty(rows, dtype=torch.int64)  # [B], the last state of a pass
    overlap = x.new_zeros(rows, dtype=torch.int64)  # [B], from the first pass of tail-biting
    W = min(G, 1 << shift)  # groups g * W + w add the same minima, see _viterbi_step_kernel
    block_b, block_h, block_j = min(8, capacity), min(4, 1 << shift), max(W, min(128, G))
    back_b = max(1, min(8, 4096 >> shift))  # rows per walk back, holding 2^(K*V) candidates each

    def search(roll: int, masked: bool, stop: int, find_overlap: bool) -> None:
        for t in range(steps - 1):
            _viterbi_step_kernel[(G // block_j, triton.cdiv(rows, block_b))](
                x, table, history[max(t - 1, 0), :, :], overlap, history[t, :, :], t, rows, steps, roll,
                L=L, KV=shift, V=V, FIRST=t == 0, MASKED=masked, BLOCK_B=block_b, BLOCK_H=block_h, BLOCK_J=block_j, W=W,
                num_warps=4, enable_fp_fusion=False,
            )
        _viterbi_final_kernel[(rows,)](
            x, table, history[max(steps - 2, 0), :, :], overlap, state, steps, roll,
            L=L, KV=shift, V=V, HAS_PREV=steps > 1, MASKED=masked, BLOCK=min(1024, num_states), enable_fp_fusion=False,
        )
        _viterbi_backtrack_kernel[(triton.cdiv(rows, back_b),)](
            x, table, history, state, scale, out, state if states is None else states, overlap, rows, steps, stop, roll, rows * G,
            L=L, KV=shift, V=V, MASKED=masked, OVERLAP=find_overlap, STATES=states is not None, BLOCK_B=back_b,
            enable_fp_fusion=False,
        )

    with torch.cuda.device(x.device):
        if tail_biting:
            half = steps // 2
            search(roll=steps - half, masked=False, stop=half, find_overlap=True)  # x.roll(half * V) through indices
        search(roll=0, masked=tail_biting, stop=0, find_overlap=False)


def trellis_quantizer_super_entry_torch(
    y_flat: torch.Tensor,
    select: Callable[[torch.Tensor], torch.Tensor],
    scale: torch.Tensor,
    *,
    out: torch.Tensor,
    table: torch.Tensor,
    K: int,
    tail_biting: bool = True,
) -> None:
    """Trellis-quantize the selected antidiagonal of super-entries into ``out`` with PyTorch.

    Each super-entry of ``select(y_flat) / scale``, read row by row, is one
    sequence, and ``out`` gets ``scale`` times its values.  ``scale``
    broadcasts against the super-entries and is constant within each.
    Nothing synchronizes with the host, so the quantizer runs in CUDA graphs
    when ``table`` already has the device and dtype of ``y_flat``.
    """
    x = select(y_flat) / scale  # [*batch, d, p, q]
    sequences = x.reshape(-1, x.size(-2) * x.size(-1))  # [B, p*q], one sequence per row
    rows, T = sequences.shape
    # Zero sequences pad the rows to a power of two, so that the search's tensors take one size per power of two
    # and a CUDA graph of GPTQ-2D's sweep keeps few buffers, for the reason given in _trellis_quantize_triton.
    padded = torch.cat((sequences, sequences.new_zeros((1 << (rows - 1).bit_length()) - rows, T)))  # [2^k, p*q]
    values, _ = _trellis_quantize_torch(padded, table, K, tail_biting)  # [2^k, p*q]
    torch.mul(values[:rows, :].view_as(x), scale, out=out)


def trellis_quantizer_super_entry_triton(
    y_flat: torch.Tensor,
    select: Callable[[torch.Tensor], torch.Tensor],
    scale: torch.Tensor,
    *,
    out: torch.Tensor,
    table: torch.Tensor,
    K: int,
    tail_biting: bool = True,
) -> None:
    """Trellis-quantize the selected antidiagonal of super-entries into ``out`` with Triton.

    The same as ``trellis_quantizer_super_entry_torch``, CUDA graphs included,
    with the Viterbi search in Triton kernels.  ``y_flat``, ``scale``, and
    ``table`` must be on one CUDA device.
    """
    x = select(y_flat) / scale  # [*batch, d, p, q]
    sequences = x.flatten(start_dim=-2)  # [*batch, d, p*q]
    scale = scale.broadcast_to(x.shape)[..., 0, 0].to(x.dtype)  # [*batch, d], one per super-entry
    _trellis_quantize_triton(sequences, table, K, tail_biting, out=out.view(sequences.shape), scale=scale)


def _brute_force(x, table, K, init=None, final=None):
    """Smallest squared distance from x [T] to the values of an allowed trellis path."""
    num_states, V = table.shape
    best = torch.inf
    for first in range(num_states):
        for bits in itertools.product(range(1 << K * V), repeat=x.numel() // V - 1):
            states = list(itertools.accumulate(bits, lambda s, b: ((s << K * V) | b) % num_states, initial=first))
            if (init is None or init[first]) and (final is None or final[states[-1]]):
                best = min(best, (table[states, :].flatten() - x).square().sum().item())
    return best


class _TestTrellis(unittest.TestCase):
    def test_viterbi(self):
        generator = torch.Generator().manual_seed(0)
        for L, K, V, T in (5, 1, 1, 7), (4, 1, 2, 8):
            table = torch.randn(1 << L, V, generator=generator, dtype=torch.float64)
            x = torch.randn(3, T, generator=generator, dtype=torch.float64)
            init, final = torch.rand(2, 3, 1 << L, generator=generator) < .5
            for masks in ((None, None), (init, final)):
                values = table[_viterbi(x, table, K, *masks), :].flatten(start_dim=-2)
                for b in range(3):
                    expected = _brute_force(x[b, :], table, K, *(None if mask is None else mask[b, :] for mask in masks))
                    self.assertAlmostEqual((values[b, :] - x[b, :]).square().sum().item(), expected)

    def test_tail_biting(self):
        generator = torch.Generator().manual_seed(1)
        L, K, V = 10, 2, 2
        table = torch.randn(1 << L, V, generator=generator)
        values, states = _trellis_quantize_torch(torch.randn(4, 32, generator=generator), table, K)
        shared = (1 << (L - K * V)) - 1  # the L-K*V bits passed to the next state
        self.assertTrue(((states[:, 1:] >> K * V) == (states[:, :-1] & shared)).all())
        self.assertTrue(((states[:, 0] >> K * V) == (states[:, -1] & shared)).all())  # circular, T*K bits
        self.assertTrue(values.equal(table[states, :].flatten(start_dim=-2)))

    def test_codes(self):
        for code in one_mad(), three_inst():
            self.assertLess(code.mean().abs().item(), .01)
        self.assertLess(abs(one_mad().std().item() - 1.), .01)
        tlut = torch.randn(1 << 9, 2, generator=torch.Generator().manual_seed(2))
        table = hyb(tlut=tlut)
        rows = (table[:, 1, None] == tlut[:, 1]).int().argmax(dim=-1)  # the LUT row of every state
        self.assertTrue(table[:, 0].abs().equal(tlut[rows, 0].abs()))
        self.assertTrue(rows.bincount(minlength=1 << 9).eq(1 << 7).all())  # every row used equally
        self.assertLess(abs((table[:, 0] != tlut[rows, 0]).float().mean().item() - .5), .01)  # sign flips

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_codes_on_cuda(self):
        """The codes are the same on CUDA as on the CPU, and the default HYB table is deterministic there."""
        for code in one_mad, three_inst, hyb:
            table = code(device="cuda")
            self.assertTrue(table.is_cuda and table.cpu().equal(code()))
            self.assertTrue(code(device="cuda", dtype=torch.float64).equal(table.double()))  # dtype only casts the result
        self.assertTrue(hyb(device="cuda").equal(hyb(device="cuda")))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_trellis_quantize_triton(self):
        """Triton returns the PyTorch values and states, along valid paths whose values recover the code."""
        generator = torch.Generator(device="cuda").manual_seed(4)

        def check(x, table, K, tail_biting):
            shift, V = K * table.size(1), table.size(1)
            out, states = torch.empty_like(x), x.new_empty(*x.shape[:-1], x.size(-1) // V, dtype=torch.int64)
            _trellis_quantize_triton(x, table, K, tail_biting, out=out, scale=x.new_ones(x.shape[:-1]), states=states)
            values, expected = _trellis_quantize_torch(x, table, K, tail_biting)
            self.assertTrue(out.equal(values) and states.equal(expected))
            shared = (table.size(0) >> shift) - 1  # consecutive states share these bits, so the bits form one code
            self.assertTrue(((states[..., 1:] >> shift) == (states[..., :-1] & shared)).all())
            self.assertTrue(not tail_biting or ((states[..., 0] >> shift) == (states[..., -1] & shared)).all())
            self.assertTrue(out.equal(table[states, :].flatten(start_dim=-2)))

        shapes = (5, 1, 1, 7), (4, 1, 2, 8), (6, 1, 2, 14), (10, 2, 2, 64), (9, 3, 1, 21), (12, 4, 2, 40), (7, 1, 1, 1)
        for (L, K, V, T), dtype, tail_biting in itertools.product(shapes, (torch.float32, torch.float64), (False, True)):
            table = torch.randn(1 << L, V, generator=generator, dtype=dtype, device="cuda")
            with self.subTest(L=L, K=K, V=V, T=T, dtype=dtype, tail_biting=tail_biting):
                check(torch.randn(37, T, generator=generator, dtype=dtype, device="cuda"), table, K, tail_biting)
        table = torch.randn(1 << 10, 2, generator=generator, device="cuda")
        for shape in (2, 3, 64), (64,):  # sequences [..., T] with several and with no leading dimensions
            with self.subTest(shape=shape):
                check(torch.randn(shape, generator=generator, device="cuda"), table, 2, True)
        table = torch.randint(-3, 4, (1 << 8, 2), generator=generator, device="cuda").float()  # many equal costs
        check(torch.randint(-3, 4, (40, 64), generator=generator, device="cuda").float(), table, 2, True)
        for code, Ks in (hyb(), (2, 3, 4)), (one_mad(), (2, 4)), (three_inst(), (2,)):
            for K in Ks:
                with self.subTest(code=code.size(1), K=K):
                    check(torch.randn(16, 256, generator=generator, device="cuda"), code.cuda(), K, True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_trellis_quantizer_super_entry(self):
        """The two quantizers write the same values for a scale per matrix or per batch."""
        generator = torch.Generator(device="cuda").manual_seed(5)
        table = hyb(device="cuda")
        y_flat = torch.randn(2, 6 * 16 * 16, generator=generator, device="cuda")  # 2 batches of 6 super-entries
        select = lambda t: t.unflatten(-1, (6, 16, 16))
        for scale in torch.tensor(1.3, device="cuda"), torch.tensor([1.3, .7], device="cuda")[:, None, None, None]:
            expected, out = torch.empty(2, 6, 16, 16, device="cuda"), torch.empty(2, 6, 16, 16, device="cuda")
            trellis_quantizer_super_entry_torch(y_flat, select, scale, out=expected, table=table, K=2)
            trellis_quantizer_super_entry_triton(y_flat, select, scale, out=out, table=table, K=2)
            self.assertTrue(out.equal(expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_trellis_quantizer_cuda_graph(self):
        """Both captured quantizers replay on new inputs and scales."""
        generator = torch.Generator(device="cuda").manual_seed(6)
        table = hyb(device="cuda")
        select = lambda t: t.unflatten(-1, (6, 16, 16))
        for quantizer in trellis_quantizer_super_entry_torch, trellis_quantizer_super_entry_triton:
            y_flat, scale = torch.randn(6 * 16 * 16, generator=generator, device="cuda"), torch.tensor(1.3, device="cuda")
            out = torch.empty(6, 16, 16, device="cuda")
            run = lambda: quantizer(y_flat, select, scale, out=out, table=table, K=2)
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                run()  # compile outside the capture
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                run()
            for factor in (1., 2.5):
                new_y, new_scale = factor * torch.randn(6 * 16 * 16, generator=generator, device="cuda"), torch.tensor(.9 * factor, device="cuda")
                y_flat.copy_(new_y)
                scale.copy_(new_scale)
                graph.replay()
                expected = torch.empty_like(out)
                quantizer(new_y, select, new_scale, out=expected, table=table, K=2)
                with self.subTest(quantizer=quantizer.__name__, factor=factor):
                    self.assertTrue(out.equal(expected))


if __name__ == "__main__":
    unittest.main()
