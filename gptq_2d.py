"""Blocked GPTQ-2D on ordinary row-major tensors.

Stages are antidiagonals ``s = i + j``.  A block is an antidiagonal band; only the
future region receives a post-block flush.  For ``s0=2, s1=4``:

    columns j ->       P: earlier stage       (i + j < s0)
        0 1 2 3 4      B: current block       (s0 <= i + j < s1)
    i 0 P P B B F      F: future / flush area (i + j >= s1)
    | 1 P B B F F
    v 2 B B F F F
      3 B F F F F
      4 F F F F F

Python visits B one antidiagonal at a time so it can call an arbitrary Python
quantizer.  The push kernels update B locally.  Before the flush kernels update
F, a small repack kernel arranges the live errors into dense matrix-multiply
panels.  Batch dimensions are flattened to one leading axis inside the kernels.
The full matrices Y, C, L, and U remain in ordinary row-major layout; only the
block-sized error scratch space is repacked.

Blocks are grouped into outer blocks.  A finished block flushes only into the
rest of its outer block, so its F stops at the outer block's end; a finished
outer block then flushes its whole band into all later stages.  Every entry is
thus read and written by a few flushes per outer block and one per earlier
outer block, instead of one per earlier block, and the outer flushes reduce
over a band as wide as the outer block.

               z and the old values of Y on I_s
                               |
                               v
      push: form errors, push them down into C and Y and the
      completed C right into Y, commit Y on I_s = z, and keep
      Y on I_{s+1}, now final, as the next old values

      batched E_stage[d,b,k] --repack--> E_panel[b,j_tile,r,c]
              contiguous rows                 contiguous BK x BN

``gptq_2d_super_entry`` rounds p x q super-entries jointly.  Its stages are the
coarse antidiagonals ``i//p + j//q``, and it shares the blocked driver and the
repack and flush kernels, which take the super-entry shape; ``gptq_2d`` is the
case ``p = q = 1``.  Super-entries larger than 1 x 1 push with their own
``tl.dot`` kernels.
"""

from collections.abc import Callable
import functools
import math

import torch
import triton
import triton.language as tl

try:
    from quantizer_int import round_quantizer, round_quantizer_super_entry
except ImportError:
    round_quantizer = round_quantizer_super_entry = None


_FLUSH_DOWN_TILE: tuple[int, int, int, int] = 64, 32, 32, 1  # BM, BN, BK, GROUP
_DEEP_FLUSH_DOWN_TILE: tuple[int, int, int, int] = 64, 64, 32, 8  # for blocks of at least 256 columns
_FLUSH_RIGHT_TILE: tuple[int, int, int, int] = 32, 64, 32, 1  # BM, BN, BK, GROUP
_DEEP_FLUSH_RIGHT_TILE: tuple[int, int, int, int] = 64, 128, 32, 8  # for blocks of at least 256 rows


def _validate_inputs(
    x: torch.Tensor,
    l: torch.Tensor,
    u: torch.Tensor,
    inplace: bool,
    require_cuda: bool = False,
    super_entry_shape: tuple[int, int] = (1, 1),
) -> tuple[int, int, torch.Size]:
    """Validate the shared API contract and return m, n, and the batch shape."""
    assert x.dim() >= 2 and l.dim() >= 2 and u.dim() >= 2, "X, L, and U must have at least two dimensions"
    m, n = x.shape[-2:]
    assert l.shape[-2:] == (m, m) and u.shape[-2:] == (n, n), "expected X=[...,m,n], L=[...,m,m], and U=[...,n,n]"
    p, q = super_entry_shape
    assert p > 0 and q > 0 and m % p == 0 and n % q == 0, "the super-entry shape must divide the matrix shape"
    assert x.dtype in (torch.float32, torch.float64) and x.dtype == l.dtype == u.dtype, "X, L, and U must share dtype float32 or float64"
    assert x.device == l.device == u.device, "X, L, and U must be on the same device"
    assert not require_cuda or x.is_cuda, "X, L, and U must be CUDA tensors"
    batch_shape: torch.Size = torch.broadcast_shapes(x.shape[:-2], l.shape[:-2], u.shape[:-2])
    assert not inplace or (x.is_contiguous() and x.shape[:-2] == batch_shape), "inplace=True requires contiguous X with the exact broadcast batch shape"
    return m, n, batch_shape


def _prepare_output(
    x: torch.Tensor,
    l: torch.Tensor,
    u: torch.Tensor,
    inplace: bool,
    require_cuda: bool = False,
    super_entry_shape: tuple[int, int] = (1, 1),
) -> tuple[torch.Tensor, int, int, torch.Size]:
    """Validate the inputs and initialize the shared output/work matrix Y."""
    m, n, batch_shape = _validate_inputs(
        x=x, l=l, u=u, inplace=inplace, require_cuda=require_cuda, super_entry_shape=super_entry_shape,
    )
    y: torch.Tensor = x if inplace else x.expand(*batch_shape, m, n).clone(memory_format=torch.contiguous_format)  # [*batch, m, n]
    return y, m, n, batch_shape


def _unit_triangular(x: torch.Tensor, upper: bool, size: int = 1) -> torch.Tensor:
    """Copy the entries of x above (``upper``) or below its size x size
    diagonal blocks, and set these blocks to identity."""
    x = x.triu() if upper else x.tril()  # [*B, k, k], unused triangle is zero
    diagonal_blocks: torch.Tensor = x.unflatten(dim=-2, sizes=(-1, size)).unflatten(dim=-1, sizes=(-1, size)).diagonal(dim1=-4, dim2=-2)  # [*B, size, size, k/size], view
    diagonal_blocks.copy_(torch.eye(size, dtype=x.dtype, device=x.device)[:, :, None])
    return x


def _antidiagonal_slices(
    *, m: int, n: int, s: int, super_entry_shape: tuple[int, int] = (1, 1),
) -> tuple[slice, slice, slice]:
    """Return the coarse row and column slices of antidiagonal I_s and a flat slice of its first entries.

    The p x q super-entries form a coarse (m/p) x (n/q) matrix; plain entries
    are 1 x 1 super-entries.  Super-entry (ibar, jbar) starts at flat position
    ``ibar*p*n + jbar*q`` of a row-major matrix.  Along ``ibar + jbar = s``
    coarse rows increase while coarse columns decrease, so consecutive
    super-entries start ``p*n - q`` positions apart, and the flat slice steps
    through these starts; for plain entries it selects the antidiagonal.  The
    step is at least the ``(p-1)*n + q`` positions one super-entry spans, which
    ``p*n - q`` falls below only for ``n = q``, when each antidiagonal holds a
    single super-entry.
    """
    p, q = super_entry_shape
    i_lo, i_hi = max(0, s - n // q + 1), min(s + 1, m // p)
    start, step = i_lo * p * n + (s - i_lo) * q, max(p * n - q, (p - 1) * n + q)
    return slice(i_lo, i_hi), slice(s - i_hi + 1, s - i_lo + 1), slice(start, start + (i_hi - i_lo) * step, step)


def _super_entries(x_flat: torch.Tensor, antidiagonal_slice: slice, n: int, super_entry_shape: tuple[int, int]) -> torch.Tensor:
    """View any ``[..., m*n]`` tensor as the ``[..., d, p, q]`` super-entries of an antidiagonal.

    ``antidiagonal_slice`` comes from ``_antidiagonal_slices``.  Two unfolds,
    both views that write through to x_flat, cut the windows and their rows:

        windows of (p-1)*n + q entries,         each window viewed as
        one per super-entry, a step apart  ->  p rows of q entries, n apart
    """
    p, q = super_entry_shape
    x_flat = x_flat[..., antidiagonal_slice.start:antidiagonal_slice.stop]  # [..., d*step]
    return x_flat.unfold(dimension=-1, size=(p - 1) * n + q, step=antidiagonal_slice.step).unfold(dimension=-1, size=q, step=n)  # [..., d, p, q]


def _quantize_antidiagonal(
    z_scratch: torch.Tensor,
    y_flat: torch.Tensor,
    selection: slice | Callable[[torch.Tensor], torch.Tensor],
    quantizer: Callable,
    quantizer_inputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the selected entries of Y and their quantized values.

    ``selection`` is the antidiagonal's flat slice, or a selector of its
    super-entries.  Only the current antidiagonal of Z is live, so the
    quantizer completely overwrites a contiguous view of the selected shape at
    the beginning of the reusable ``z_scratch``.
    """
    y_selected: torch.Tensor = selection(y_flat) if callable(selection) else y_flat[..., selection]  # [*batch, d] or [*batch, d, p, q]
    z: torch.Tensor = z_scratch[:y_selected.numel()].view(y_selected.shape)  # contiguous
    quantizer(y_flat, selection, *quantizer_inputs, out=z)
    return y_selected, z


def _batch_offsets(x: torch.Tensor, batch_shape: torch.Size) -> torch.Tensor:
    """Return the source-matrix element offset for each logical batch.

    An expanded view gives broadcast dimensions stride zero, so multiple
    logical batches reuse the same stored matrix instead of copying it.
    The Triton implementations handle an empty batch before calling this
    helper, so B > 0.
    """
    batch_size: int = batch_shape.numel()
    batch_index: torch.Tensor = torch.arange(batch_size, dtype=torch.int64, device=x.device)  # [B]
    offsets: torch.Tensor = batch_index.new_zeros(batch_size)  # [B]
    padded_shape: tuple[int, ...] = (1,) * (len(batch_shape) - x.dim() + 2) + x.shape
    batch_strides = x.reshape(padded_shape).expand(*batch_shape, *x.shape[-2:]).stride()[:-2]
    divisor: int = batch_size
    for size, stride in zip(batch_shape, batch_strides):
        divisor //= size
        offsets += (batch_index // divisor % size) * stride
    return offsets


def _flush_down_tile(block_size: int, super_entry_shape: tuple[int, int] = (1, 1)) -> tuple[int, int, int, int]:
    """Return the down-flush tile of a block, which also shapes its error panels.  Wider tiles
    widen each column tile's band by BN/q stages, which blocks of at least 256 columns absorb."""
    return _DEEP_FLUSH_DOWN_TILE if block_size * super_entry_shape[1] >= 256 else _FLUSH_DOWN_TILE


def _error_panel_shape(n: int, block_size: int, super_entry_shape: tuple[int, int] = (1, 1)) -> tuple[int, int, int]:
    """Return the packed error-panel shape for one block of p x q super-entries."""
    _, BN, BK, _ = _flush_down_tile(block_size, super_entry_shape)
    p, q = super_entry_shape
    # BN columns span at most cdiv(BN-1, q) + 1 super-entry columns, whose bands shift by p rows.
    panel_height: int = triton.cdiv((block_size + triton.cdiv(BN - 1, q)) * p, BK) * BK
    return triton.cdiv(n, BN), panel_height, BN


def _super_entry_tiles(super_entry_shape: tuple[int, int]) -> tuple[int, int, int]:
    """Return the tl.dot tile ``BP x BQ`` covering one super-entry and the
    number of receiver rows or columns per push program.  Receiver tiles shrink
    for super-entries above 32 x 32, whose larger tiles spill registers."""
    BP, BQ = (max(16, triton.next_power_of_2(size)) for size in super_entry_shape)
    return BP, BQ, 32 if BP * BQ <= 32 * 32 else 16


@triton.jit
def _masked_add(ptr, offsets, value, mask):
    """Add value to selected elements; all tensor arguments share a shape."""
    target = ptr + offsets  # same shape as offsets
    target.store(tl.load(target, mask=mask) + value, mask=mask)  # same shape as offsets


@triton.jit
def _program_ids(num_programs_0, num_programs_1):
    """Return ``pid(0)``, ``pid(1)``, and the batch of a launch from ``_grid``.

    A three-dimensional launch has ``num_programs_0`` programs on axis 0.  A
    one-dimensional launch lists the same programs in the same order,
    ``pid(0)`` fastest and the batch slowest, so its program id is split with
    divisions.  When it also has ``num_programs_0`` programs, both readings
    agree.
    """
    if tl.num_programs(0) == num_programs_0:
        pid_0, pid_1, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    else:
        pid = tl.program_id(0)  # scalar
        pid_0, pid_1, batch = pid % num_programs_0, pid // num_programs_0 % num_programs_1, pid // (num_programs_0 * num_programs_1)
    return pid_0, pid_1, batch.to(tl.int64)


def _grid(num_programs_0: int, num_programs_1: int, batch_size: int) -> tuple[int, ...]:
    """Return a ``(num_programs_0, num_programs_1, batch_size)`` launch grid,
    or the same programs in one dimension past CUDA's y and z axis limit."""
    if max(num_programs_1, batch_size) <= 65_535:
        return num_programs_0, num_programs_1, batch_size
    return (num_programs_0 * num_programs_1 * batch_size,)


@triton.jit
def _push_kernel(
    y_ptr,
    c_ptr,
    e_ptr,
    z_ptr,
    old_ptr,
    next_old_ptr,
    l_ptr,
    l_batch_offsets_ptr,
    u_ptr,
    u_batch_offsets_ptr,
    m,
    n,
    s,
    s1,
    num_row_tiles,
    num_source_tiles,
    BATCHED: tl.constexpr,
    ROW_TILE_SIZE: tl.constexpr,
    SOURCE_TILE_SIZE: tl.constexpr,
):
    """Form one antidiagonal's errors, push them inside the block, and commit z.

    Lane ``k`` of ``z``, ``old``, and ``E`` is the source ``(i_lo+k, s-i_lo-k)``
    of the current antidiagonal ``I_s``.  ``E = z - old`` is the row ``s-s0``
    of ``E_stage``, counted from the start ``s0`` of the outer block because
    the flushes read the errors of whole blocks and outer blocks; a single
    block holding all stages is never flushed and reuses row zero.  For
    ``0 < p < s1-s``, every source ``S=(i,j)`` pushes ``L[i+p,i] * E[i,j]``
    down into ``C[i+p,j]`` and ``Y[i+p,j]`` (``d``), and ``C[i,j] * U[j,j+p]``
    right into ``Y[i,j+p]`` (``r``), where ``C[i,j]`` includes ``E[i,j]``
    because ``L[i,i]=1``.  Matrix view for ``s=3`` and ``s1=6``, where ``x``
    receives both:

                         columns j ->
                            0 1 2 3 4 5
                       i=0  . . . S r r
                         1  . . S x r .
                         2  . S x x . .
                         3  S x x . . .
                         4  d d . . . .
                         5  d . . . . .

    Programs own receiver rows, so a receiver gets both pushes, down first,
    in one read-modify-write.  Row r receives in the columns of the sources
    ``i=r-p``; a program takes ``ROW_TILE_SIZE`` rows and a chunk of these
    sources, including positions past the ends of ``I_s`` that only push
    right.  Neighboring sources reach neighboring columns, so the accesses to
    L, U, C, and Y along a tile row are contiguous.

    Errors come from ``old`` rather than Y, so the program owning a source's
    row commits z at once while other programs still form its error, and the
    next quantizer sees every earlier antidiagonal quantized.  The entries at
    ``p=1`` receive their last push here and are saved in ``next_old``;
    Python copies the first antidiagonal of each block after its flush.  As
    several programs complete ``C[i,j]`` in registers, none writes it, and
    ``_add_own_errors`` adds the sources' own errors before the block's flush.

    Backing PyTorch tensors are ``Y,C: [B,m,n]``, ``E: [B,min(m,n)]``,
    ``z,old: [B,antidiag_len]``, ``next_old: [B,antidiag_len of I_{s+1}]``,
    ``L: [*Bl,m,m]``, and ``U: [*Bu,n,n]``; the batch offset maps implement
    broadcasting without copying L and U.
    """
    row_tile, source_tile, batch = _program_ids(num_row_tiles, num_source_tiles)  # scalars
    i_lo = tl.maximum(0, s - n + 1)  # scalar
    i_hi = tl.minimum(s + 1, m)  # scalar
    lanes = batch * (i_hi - i_lo) - i_lo  # scalar, z and old hold row i at lanes + i
    r0 = i_lo + row_tile * ROW_TILE_SIZE  # scalar
    r = r0 + tl.arange(0, ROW_TILE_SIZE)  # [ROW_TILE_SIZE], receiver rows
    i = r0 - (s1 - s - 1) + source_tile * SOURCE_TILE_SIZE + tl.arange(0, SOURCE_TILE_SIZE)  # [SOURCE_TILE_SIZE], sources, also past the ends of I_s
    j = s - i  # [SOURCE_TILE_SIZE], since i+j=s
    source_ok = (i >= i_lo) & (i < i_hi)  # [SOURCE_TILE_SIZE]
    err = tl.load(z_ptr + lanes + i, mask=source_ok) - tl.load(old_ptr + lanes + i, mask=source_ok)  # [SOURCE_TILE_SIZE]
    # The program owning the source's row stores its error.
    (e_ptr + batch * tl.minimum(m, n) + i - i_lo).store(err, mask=source_ok & (i >= r0) & (i < r0 + ROW_TILE_SIZE))

    own = r < i_hi  # [ROW_TILE_SIZE], rows holding a source
    source = batch * m * n + r * n + s - r  # [ROW_TILE_SIZE], offsets of the rows' sources in Y/C
    z = tl.load(z_ptr + lanes + r, mask=own)  # [ROW_TILE_SIZE]
    own_err = z - tl.load(old_ptr + lanes + r, mask=own)  # [ROW_TILE_SIZE]
    own_c = tl.load(c_ptr + source, mask=own) + own_err  # [ROW_TILE_SIZE], completed C[r,s-r]
    (y_ptr + source).store(z, mask=own & (source_tile == 0))  # [ROW_TILE_SIZE], commit

    # Form all (receiver row, source) pairs owned by this program.
    p = r[:, None] - i[None, :]  # [ROW_TILE_SIZE, SOURCE_TILE_SIZE], push offsets
    receiver_ok = (p > 0) & (p < s1 - s) & (r[:, None] < m) & (j[None, :] < n)  # [ROW_TILE_SIZE, SOURCE_TILE_SIZE]
    down_ok = receiver_ok & source_ok[None, :]  # [ROW_TILE_SIZE, SOURCE_TILE_SIZE]
    right_ok = receiver_ok & own[:, None]  # [ROW_TILE_SIZE, SOURCE_TILE_SIZE]
    receiver = batch * m * n + r[:, None] * n + j[None, :]  # [ROW_TILE_SIZE, SOURCE_TILE_SIZE]
    l_batch_offset = tl.load(l_batch_offsets_ptr + batch) if BATCHED else 0  # scalar
    u_batch_offset = tl.load(u_batch_offsets_ptr + batch) if BATCHED else 0  # scalar
    down = tl.load(l_ptr + l_batch_offset + r[:, None] * m + i[None, :], mask=down_ok) * err[None, :]  # [ROW_TILE_SIZE, SOURCE_TILE_SIZE]
    right = own_c[:, None] * tl.load(u_ptr + u_batch_offset + (s - r)[:, None] * n + j[None, :], mask=right_ok)  # [ROW_TILE_SIZE, SOURCE_TILE_SIZE]
    _masked_add(c_ptr, receiver, down, down_ok)  # [ROW_TILE_SIZE, SOURCE_TILE_SIZE]
    y = tl.load(y_ptr + receiver, mask=down_ok | right_ok)  # [ROW_TILE_SIZE, SOURCE_TILE_SIZE]
    y = tl.where(down_ok, y + down, y)  # [ROW_TILE_SIZE, SOURCE_TILE_SIZE]
    y = tl.where(right_ok, y + right, y)  # [ROW_TILE_SIZE, SOURCE_TILE_SIZE]
    (y_ptr + receiver).store(y, mask=down_ok | right_ok)  # [ROW_TILE_SIZE, SOURCE_TILE_SIZE]

    # The entries at p=1 are final: they are the old values of the next stage.
    next_i_lo = tl.maximum(0, s - n + 2)  # scalar
    next_lane = batch * (tl.minimum(s + 2, m) - next_i_lo) + r - next_i_lo  # [ROW_TILE_SIZE], lanes of the rows in I_{s+1}
    (next_old_ptr + tl.broadcast_to(next_lane[:, None], (ROW_TILE_SIZE, SOURCE_TILE_SIZE))).store(y, mask=(p == 1) & (down_ok | right_ok))


def _push(
    y: torch.Tensor,
    c: torch.Tensor,
    e: torch.Tensor,
    z: torch.Tensor,
    old: torch.Tensor,
    next_old: torch.Tensor,
    l: torch.Tensor,
    l_batch_offsets: torch.Tensor,
    u: torch.Tensor,
    u_batch_offsets: torch.Tensor,
    s: int,
    s1: int,
) -> None:
    ROW_TILE_SIZE, SOURCE_TILE_SIZE = 8, 32
    batch_size, m, n = y.shape
    i_lo, i_hi = max(0, s - n + 1), min(s + 1, m)
    # Receiver rows run from the first source down to s1-s-1 rows below the last one,
    # and a row tile receives in the columns of ROW_TILE_SIZE+s1-s-1 sources.
    num_row_tiles: int = triton.cdiv(min(m, i_hi - 1 + s1 - s) - i_lo, ROW_TILE_SIZE)
    num_source_tiles: int = triton.cdiv(ROW_TILE_SIZE + s1 - s - 1, SOURCE_TILE_SIZE)
    _push_kernel[_grid(num_row_tiles, num_source_tiles, batch_size)](
        y, c, e, z, old, next_old, l, l_batch_offsets, u, u_batch_offsets,
        m=m, n=n, s=s, s1=s1,
        num_row_tiles=num_row_tiles, num_source_tiles=num_source_tiles,
        BATCHED=batch_size > 1,
        ROW_TILE_SIZE=ROW_TILE_SIZE,
        SOURCE_TILE_SIZE=SOURCE_TILE_SIZE,
        num_warps=4,
    )


@triton.jit
def _add_own_errors_kernel(
    c_ptr,
    e_ptr,
    m,
    n,
    s0,
    s1,
    num_row_tiles,
    num_stage_tiles,
    ROW_TILE_SIZE: tl.constexpr,
    STAGE_TILE_SIZE: tl.constexpr,
):
    """Add the errors of the stages ``s0..s1-1`` to their own sources in C.

    The push kernel leaves them out.  Row r holds these sources in the
    contiguous columns ``s0-r .. s1-1-r``, so programs own rows and walk
    their segments.  Backing PyTorch tensors are ``C: [B,m,n]`` and
    ``E: [s1-s0,B,min(m,n)]``.
    """
    row_tile, stage_tile, batch = _program_ids(num_row_tiles, num_stage_tiles)  # scalars
    batch_size = (tl.num_programs(0) * tl.num_programs(1) * tl.num_programs(2) // (num_row_tiles * num_stage_tiles)).to(tl.int64)  # scalar
    r = tl.maximum(0, s0 - n + 1) + row_tile * ROW_TILE_SIZE + tl.arange(0, ROW_TILE_SIZE)  # [ROW_TILE_SIZE]
    d = stage_tile * STAGE_TILE_SIZE + tl.arange(0, STAGE_TILE_SIZE)  # [STAGE_TILE_SIZE], local stages
    s = s0 + d  # [STAGE_TILE_SIZE]
    k = r[:, None] - tl.maximum(0, s - n + 1)[None, :]  # [ROW_TILE_SIZE, STAGE_TILE_SIZE], source lanes
    ok = (s[None, :] < s1) & (k >= 0) & (r[:, None] < tl.minimum(s + 1, m)[None, :])  # [ROW_TILE_SIZE, STAGE_TILE_SIZE]
    err = tl.load(e_ptr + (d[None, :] * batch_size + batch) * tl.minimum(m, n) + k, mask=ok)  # [ROW_TILE_SIZE, STAGE_TILE_SIZE]
    _masked_add(c_ptr, batch * m * n + r[:, None] * n + s[None, :] - r[:, None], err, ok)  # [ROW_TILE_SIZE, STAGE_TILE_SIZE]


def _add_own_errors(c: torch.Tensor, e_stage: torch.Tensor, s0: int, s1: int) -> None:
    ROW_TILE_SIZE, STAGE_TILE_SIZE = 32, 32
    batch_size, m, n = c.shape
    # The sources run from the first row of I_s0 to the last row of I_{s1-1}.
    num_row_tiles: int = triton.cdiv(min(s1, m) - max(0, s0 - n + 1), ROW_TILE_SIZE)
    num_stage_tiles: int = triton.cdiv(s1 - s0, STAGE_TILE_SIZE)
    _add_own_errors_kernel[_grid(num_row_tiles, num_stage_tiles, batch_size)](
        c, e_stage,
        m=m, n=n, s0=s0, s1=s1,
        num_row_tiles=num_row_tiles, num_stage_tiles=num_stage_tiles,
        ROW_TILE_SIZE=ROW_TILE_SIZE,
        STAGE_TILE_SIZE=STAGE_TILE_SIZE,
        num_warps=4,
    )


@triton.jit
def _push_down_super_entry_kernel(
    y_ptr,
    c_ptr,
    e_ptr,
    z_ptr,
    l_ptr,
    l_batch_offsets_ptr,
    m,
    n,
    p,
    q,
    s,
    s1,
    num_row_tiles,
    BATCHED: tl.constexpr,
    BP: tl.constexpr,
    BQ: tl.constexpr,
    ROW_TILE_SIZE: tl.constexpr,
):
    """Form one coarse antidiagonal's super-entry errors, fold them into C, and push them down.

    Source lane ``k`` is the p x q super-entry ``(ibar,jbar) = (i_lo+k, s-i_lo-k)``
    of the coarse antidiagonal ``ibar + jbar = s``.  Its error ``E = z - Y``
    reaches the column band ``jbar*q .. jbar*q+q-1`` of the rows
    ``ibar*p <= r < (s1-jbar)*p`` inside the block:

        C[r, jbar band] += L[r, ibar band] @ E,

    and the rows below the super-entry receive the same update in Y.  The
    identity diagonal block of L is implicit: the super-entry's own rows add E
    to C, and L is read only below that block.  The program owning those rows
    also stores E for the flushes.

    ``pid(0)`` selects the source lane and ``pid(1)`` a tile of
    ``ROW_TILE_SIZE`` rows counted from the source's first row.  Sources own
    different column bands, so programs never write the same element.
    ``BP x BQ`` is a ``tl.dot`` tile covering one super-entry.  Backing
    PyTorch tensors are ``Y,C: [B,m,n]``, ``E: [B,lanes,p,q]`` with
    ``lanes=min(m//p,n//q)``, ``z: [B,antidiag_len,p,q]``, and ``L: [*Bl,m,m]``.
    """
    i_lo = tl.maximum(0, s - n // q + 1)  # scalar, first coarse row of the antidiagonal
    antidiag_len = tl.minimum(s + 1, m // p) - i_lo  # scalar
    k, row_tile, batch = _program_ids(antidiag_len, num_row_tiles)  # scalars, k is the source lane
    ibar = i_lo + k  # scalar
    jbar = s - ibar  # scalar
    rows_end = tl.minimum(m, (s1 - jbar) * p)  # scalar, first row past the block
    r0 = ibar * p + row_tile * ROW_TILE_SIZE  # scalar
    if r0 >= rows_end:  # this source's push ends above the tile
        return
    r = r0 + tl.arange(0, ROW_TILE_SIZE)  # [ROW_TILE_SIZE], receiver rows
    a = tl.arange(0, BP)  # [BP], rows within a super-entry
    b = tl.arange(0, BQ)  # [BQ], columns within a super-entry
    entry_ok = (a[:, None] < p) & (b[None, :] < q)  # [BP, BQ]

    matrix_offset = batch * m * n  # scalar, Y/C batch offset
    source = matrix_offset + (ibar * p + a[:, None]) * n + jbar * q + b[None, :]  # [BP, BQ]
    z = tl.load(z_ptr + ((batch * antidiag_len + k) * p + a[:, None]) * q + b[None, :], mask=entry_ok, other=0.)  # [BP, BQ]
    err = z - tl.load(y_ptr + source, mask=entry_ok, other=0.)  # [BP, BQ], zero outside the super-entry
    if row_tile == 0:
        lanes = tl.minimum(m // p, n // q)  # scalar
        (e_ptr + ((batch * lanes + k) * p + a[:, None]) * q + b[None, :]).store(err, mask=entry_ok)  # [BP, BQ]

    own = r < ibar * p + p  # [ROW_TILE_SIZE], the super-entry's own rows
    l_batch_offset = tl.load(l_batch_offsets_ptr + batch) if BATCHED else 0  # scalar
    l_values = tl.load(
        l_ptr + l_batch_offset + r[:, None] * m + ibar * p + a[None, :],
        mask=(~own & (r < rows_end))[:, None] & (a[None, :] < p),
        other=0.,
    )  # [ROW_TILE_SIZE, BP]
    l_values = tl.where(own[:, None] & (r[:, None] - ibar * p == a[None, :]), 1., l_values)  # [ROW_TILE_SIZE, BP], implicit identity block
    update = tl.dot(l_values, err, input_precision="ieee")  # [ROW_TILE_SIZE, BQ]

    receiver = matrix_offset + r[:, None] * n + jbar * q + b[None, :]  # [ROW_TILE_SIZE, BQ]
    receiver_ok = (r[:, None] < rows_end) & (b[None, :] < q)  # [ROW_TILE_SIZE, BQ]
    _masked_add(c_ptr, receiver, update, receiver_ok)  # [ROW_TILE_SIZE, BQ]
    # The super-entry's own Y will hold z, so only rows below it receive this push.
    _masked_add(y_ptr, receiver, update, receiver_ok & ~own[:, None])  # [ROW_TILE_SIZE, BQ]


def _push_down_super_entry(
    y: torch.Tensor,
    c: torch.Tensor,
    e: torch.Tensor,
    z: torch.Tensor,
    l: torch.Tensor,
    l_batch_offsets: torch.Tensor,
    s: int,
    s1: int,
    super_entry_shape: tuple[int, int],
) -> None:
    BP, BQ, ROW_TILE_SIZE = _super_entry_tiles(super_entry_shape)
    batch_size, m, n = y.shape
    p, q = super_entry_shape
    i_lo, i_hi = max(0, s - n // q + 1), min(s + 1, m // p)
    # The first source's push is the longest: from its first row to the matrix or block end.
    num_row_tiles: int = triton.cdiv(min(m - i_lo * p, (s1 - s) * p), ROW_TILE_SIZE)
    _push_down_super_entry_kernel[_grid(i_hi - i_lo, num_row_tiles, batch_size)](
        y, c, e, z, l, l_batch_offsets,
        m=m, n=n, p=p, q=q, s=s, s1=s1,
        num_row_tiles=num_row_tiles,
        BATCHED=batch_size > 1,
        BP=BP, BQ=BQ,
        ROW_TILE_SIZE=ROW_TILE_SIZE,
        num_warps=4,
    )


@triton.jit
def _push_right_super_entry_kernel(
    y_ptr,
    c_ptr,
    z_ptr,
    u_ptr,
    u_batch_offsets_ptr,
    m,
    n,
    p,
    q,
    s,
    s1,
    num_column_tiles,
    BATCHED: tl.constexpr,
    BP: tl.constexpr,
    BQ: tl.constexpr,
    COLUMN_TILE_SIZE: tl.constexpr,
):
    """Push one coarse antidiagonal's completed C super-entries right and commit z.

    Source lane ``k`` is the super-entry ``(ibar,jbar)`` of the push-down
    kernel.  The columns ``(jbar+1)*q <= c < (s1-ibar)*q`` right of it inside
    the block receive

        Y[ibar band, c] += C[ibar band, jbar band] @ U[jbar band, c].

    The identity diagonal block of U would only reach the super-entry itself,
    which holds z afterwards, so U is read only right of that block.  ``pid(0)``
    selects the source lane and ``pid(1)`` a tile of ``COLUMN_TILE_SIZE``
    columns counted from the first column right of the source.  This launch
    reads only C and U, so the programs of tile zero commit z into the source
    first.  Sources own different row bands, so the writes are race-free.
    Backing PyTorch tensors are ``Y,C: [B,m,n]``, ``z: [B,antidiag_len,p,q]``,
    and ``U: [*Bu,n,n]``.
    """
    i_lo = tl.maximum(0, s - n // q + 1)  # scalar, first coarse row of the antidiagonal
    antidiag_len = tl.minimum(s + 1, m // p) - i_lo  # scalar
    k, column_tile, batch = _program_ids(antidiag_len, num_column_tiles)  # scalars, k is the source lane
    ibar = i_lo + k  # scalar
    jbar = s - ibar  # scalar
    a = tl.arange(0, BP)  # [BP], rows within a super-entry
    b = tl.arange(0, BQ)  # [BQ], columns within a super-entry
    entry_ok = (a[:, None] < p) & (b[None, :] < q)  # [BP, BQ]
    matrix_offset = batch * m * n  # scalar, Y/C batch offset
    source = matrix_offset + (ibar * p + a[:, None]) * n + jbar * q + b[None, :]  # [BP, BQ]

    if column_tile == 0:
        z = tl.load(z_ptr + ((batch * antidiag_len + k) * p + a[:, None]) * q + b[None, :], mask=entry_ok)  # [BP, BQ]
        (y_ptr + source).store(z, mask=entry_ok)  # [BP, BQ], commit

    cols_end = tl.minimum(n, (s1 - ibar) * q)  # scalar, first column past the block
    c0 = (jbar + 1) * q + column_tile * COLUMN_TILE_SIZE  # scalar
    if c0 >= cols_end:  # this source's push ends left of the tile
        return
    cols = c0 + tl.arange(0, COLUMN_TILE_SIZE)  # [COLUMN_TILE_SIZE], receiver columns
    c_values = tl.load(c_ptr + source, mask=entry_ok, other=0.)  # [BP, BQ]
    u_batch_offset = tl.load(u_batch_offsets_ptr + batch) if BATCHED else 0  # scalar
    u_values = tl.load(
        u_ptr + u_batch_offset + (jbar * q + b[:, None]) * n + cols[None, :],
        mask=(b[:, None] < q) & (cols[None, :] < cols_end),
        other=0.,
    )  # [BQ, COLUMN_TILE_SIZE]
    update = tl.dot(c_values, u_values, input_precision="ieee")  # [BP, COLUMN_TILE_SIZE]
    receiver = matrix_offset + (ibar * p + a[:, None]) * n + cols[None, :]  # [BP, COLUMN_TILE_SIZE]
    _masked_add(y_ptr, receiver, update, (a[:, None] < p) & (cols[None, :] < cols_end))  # [BP, COLUMN_TILE_SIZE]


def _push_right_super_entry(
    y: torch.Tensor,
    c: torch.Tensor,
    z: torch.Tensor,
    u: torch.Tensor,
    u_batch_offsets: torch.Tensor,
    s: int,
    s1: int,
    super_entry_shape: tuple[int, int],
) -> None:
    BP, BQ, COLUMN_TILE_SIZE = _super_entry_tiles(super_entry_shape)
    batch_size, m, n = y.shape
    p, q = super_entry_shape
    i_lo, i_hi = max(0, s - n // q + 1), min(s + 1, m // p)
    j_lo = s - (i_hi - 1)  # coarse column of the last source, whose push is the longest
    # At least one column tile runs because it also commits z.
    num_column_tiles: int = max(1, triton.cdiv(min(n - (j_lo + 1) * q, (s1 - s - 1) * q), COLUMN_TILE_SIZE))
    _push_right_super_entry_kernel[_grid(i_hi - i_lo, num_column_tiles, batch_size)](
        y, c, z, u, u_batch_offsets,
        m=m, n=n, p=p, q=q, s=s, s1=s1,
        num_column_tiles=num_column_tiles,
        BATCHED=batch_size > 1,
        BP=BP, BQ=BQ,
        COLUMN_TILE_SIZE=COLUMN_TILE_SIZE,
        num_warps=4,
    )


@triton.jit
def _repack_errors_kernel(
    e_panel_ptr,
    e_stage_ptr,
    m,
    n,
    p,
    q,
    s0,
    num_row_tiles,
    num_panels,
    BLOCK_SIZE: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """Repack antidiagonal error rows into dense column panels.

    The block ends at ``s1=s0+BLOCK_SIZE``.
    The push kernels store one antidiagonal per contiguous
    ``E_stage`` row.  For ``s0=6`` and ``s1=9``, local stage ``d=s-s0`` gives:

        source lane k ->      0       1       2       3       4       5       6       7       8
        d=0, s=6           E[0,6]  E[1,5]  E[2,4]  E[3,3]  E[4,2]  E[5,1]  E[6,0]     .       .
        d=1, s=7           E[0,7]  E[1,6]  E[2,5]  E[3,4]  E[4,3]  E[5,2]  E[6,1]  E[7,0]     .
        d=2, s=8           E[0,8]  E[1,7]  E[2,6]  E[3,5]  E[4,4]  E[5,3]  E[6,2]  E[7,1]  E[8,0]

    This layout is convenient while processing antidiagonals, but the down
    flush needs ordinary matrix tiles ``E[i,j]`` for ``L @ E``.  The repack
    groups ``BN`` consecutive matrix columns into each output panel:

        panel = pid(1)       output panel number
        row tile = pid(0)    BK consecutive rows within that panel
        j0 = panel * BN      first matrix column in the panel
        c = 0 .. BN-1        column within the panel
        j = j0 + c           corresponding matrix column

    For each column j, the block contains rows ``s0-j <= i < s1-j``.  These row
    ranges are shifted across the panel.  Row zero is therefore aligned with
    the smallest required matrix row:

        i_start = s0 - (j0 + BN - 1)
        i = i_start + r      matrix row represented by panel row r

    For example, panel 1 has ``j0=3`` when ``BN=3``.  It owns matrix columns
    3, 4, and 5, and ``i_start = 6 - (3 + 3 - 1) = 1``.  Thus panel row r
    corresponds to matrix row ``i=r+1``:

                                local column c ->
                                     0       1       2
                                matrix j ->
                                     3       4       5
        panel row r=0, i=1        .       .     E[1,5]
                  r=1, i=2        .     E[2,4]  E[2,5]
                  r=2, i=3      E[3,3]  E[3,4]  E[3,5]
                  r=3, i=4      E[4,3]  E[4,4]    .
                  r=4, i=5      E[5,3]    .       .

    An output position ``(r,c)`` belongs to local stage

        d = i + j - s0 = r + c - (BN - 1).

    If ``0 <= d < BLOCK_SIZE``, its input is ``E_stage[d,k]``, where
    ``k = i - i_lo(s0+d)`` selects the source lane within that antidiagonal and
    ``i_lo(s)=max(0,s-n+1)`` is its first matrix row.  Other positions become
    zero.  The result is a dense rectangular panel, so the flush kernel can
    load contiguous ``BK x BN`` chunks directly into ``tl.dot``.

    For p x q super-entries the stages are coarse: entry ``(i,j)`` belongs to
    ``s = i//p + j//q``, lane ``k`` counts super-entries, and each ``E_stage``
    lane holds a whole p x q super-entry.  Column j then needs the rows
    ``(s0 - j//q)*p <= i < (s1 - j//q)*p``, so ``i_start = (s0 -
    (j0+BN-1)//q)*p``.  The description above is the case ``p = q = 1``.

    The batch selects an independent problem.  Backing PyTorch tensors are
    ``E_stage: [BLOCK_SIZE,B,lanes,p,q]`` with ``lanes=min(m//p,n//q)`` and
    ``E_panel: [B,ceil(n/BN),panel_height,BN]``.  Shapes below describe one
    ``BK x BN`` repack tile in one batch.
    """
    row_tile, panel, batch = _program_ids(num_row_tiles, num_panels)  # scalars, panel is the output j tile
    batch_size = (tl.num_programs(0) * tl.num_programs(1) * tl.num_programs(2) // (num_row_tiles * num_panels)).to(tl.int64)  # scalar
    panel_height = tl.cdiv((BLOCK_SIZE + tl.cdiv(BN - 1, q)) * p, BK) * BK  # scalar, as in _error_panel_shape
    lanes = tl.minimum(m // p, n // q)  # scalar, super-entries per E_stage row
    r = row_tile * BK + tl.arange(0, BK)  # [BK], panel rows
    c = tl.arange(0, BN)  # [BN], panel columns
    j0 = panel * BN  # scalar
    j = j0 + c  # [BN], matrix columns
    i_start = (s0 - (j0 + BN - 1) // q) * p  # scalar
    i = i_start + r  # [BK], matrix rows
    ibar, jbar = i // p, j // q  # [BK], [BN], super-entry rows (meaningful where i >= 0) and columns
    d = ibar[:, None] + jbar[None, :] - s0  # [BK, BN], local stages
    s = s0 + d  # [BK, BN], global stages
    i_lo = tl.maximum(0, s - n // q + 1)  # [BK, BN]
    k = ibar[:, None] - i_lo  # [BK, BN], source lanes
    valid = (
        (d >= 0)
        & (d < BLOCK_SIZE)
        & (i[:, None] >= 0)
        & (i[:, None] < m)
        & (j[None, :] < n)
    )  # [BK, BN]

    stage_offset = (((d * batch_size + batch) * lanes + k) * p + (i - ibar * p)[:, None]) * q + (j - jbar * q)[None, :]  # [BK, BN]
    error = tl.load(e_stage_ptr + stage_offset, mask=valid, other=0.)  # [BK, BN]
    panel_offset = ((batch * tl.cdiv(n, BN) + panel) * panel_height + r[:, None]) * BN + c[None, :]  # [BK, BN]
    # panel_height is a multiple of BK, so every launched output is allocated.
    (e_panel_ptr + panel_offset).store(error)  # [BK, BN]


def _repack_errors(
    e_panel: torch.Tensor,
    e_stage: torch.Tensor,
    m: int,
    n: int,
    s0: int,
    s1: int,
    super_entry_shape: tuple[int, int] = (1, 1),
) -> None:
    _, BN, BK, _ = _flush_down_tile(s1 - s0, super_entry_shape)
    p, q = super_entry_shape
    num_panels, panel_height, _ = _error_panel_shape(n=n, block_size=s1 - s0, super_entry_shape=super_entry_shape)
    num_row_tiles: int = panel_height // BK
    # Row tiles change fastest, so adjacent programs write consecutive row tiles of one panel.
    _repack_errors_kernel[_grid(num_row_tiles, num_panels, e_stage.size(1))](
        e_panel, e_stage,
        m=m, n=n, p=p, q=q, s0=s0,
        num_row_tiles=num_row_tiles, num_panels=num_panels,
        BLOCK_SIZE=s1 - s0,
        BN=BN,
        BK=BK,
        num_warps=4,
    )


@triton.jit
def _flush_down_kernel(
    y_ptr,
    c_ptr,
    e_panel_ptr,
    l_ptr,
    l_batch_offsets_ptr,
    m,
    n,
    p,
    q,
    s0,
    s_end,
    num_column_tiles,
    num_row_tiles,
    BATCHED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    """Flush the block's errors down through L, updating both C and Y.

    The block ends at ``s1=s0+BLOCK_SIZE``, and its targets are the entries
    with ``s1 <= i+j < s_end``: the rest of the enclosing outer block, or all
    later stages.  For every target entry (i,j), this kernel computes

        M[i,j] = sum_k L[i,k] E[k,j],  for s0 <= k+j < s1,
        C[i,j] += M[i,j],
        Y[i,j] += M[i,j].

    The first update completes the deferred part of C=LE.  The second applies
    the same quantity as downward feedback to Y.

    Each program owns one ``BM x BN`` output tile in natural Y/C:

                         source rows k (BK at a time)
                     +---------------------------+
        L[i, k]       |                           | E[k, j]
        BM x BK       +---------------------------+ BK x BN
                 \\____________ tl.dot ____________/
                                BM x BN

    For one output column j, the required source rows are
    ``s0-j <= k < s1-j``.  These ranges shift left across the BN columns.  A
    rectangular tl.dot needs one shared k range, so ``k_start`` is the smallest
    k needed anywhere in the tile.  The panel height covers that union in
    BK-sized tiles.

    The repack kernel has already materialized the shifted source-row ranges as
    contiguous panels.  For the column tile owned by ``pid(0)``, panel row zero is
    ``k_start`` and every ``BK x BN`` load below is dense.  Entries outside the
    exact antidiagonal band, including rows outside the matrix and columns past
    n, are stored zeros, so the error loads need no mask.

    The rectangular dot reuses each loaded L tile across BN output columns.  A
    literal source-oriented push would make several errors update the same
    output and would require atomics.

    Results are stored only where ``s1 <= i+j < s_end``.  The down flush must
    finish before the right flush because the latter consumes the completed C.
    Programs whose entire output tile lies before ``s1`` or from ``s_end`` on
    return before loading or computing.  The element mask still clips the
    boundary tiles.  The launch grid covers only columns ``j < s1``, since a
    later column would need the source rows ``s0-j <= k < s1-j``, which are all
    negative, and rows ``i < s_end``.

    For p x q super-entries every antidiagonal is coarse, ``i//p + j//q``:
    column j needs the source rows ``(s0 - j//q)*p <= k < (s1 - j//q)*p``, L is
    read only below its diagonal blocks, and the grid covers the columns
    ``j < s1*q`` and rows ``i < s_end*p``.  The description above is the case
    ``p = q = 1``.

    The batch selects an independent problem.  Backing PyTorch tensors are
    ``Y,C: [B,m,n]``, ``L: [*Bl,m,m]``, and
    ``E_panel: [B,ceil(n/BN),panel_height,BN]``.  The L batch offset map
    implements broadcasting.  Shapes below describe one ``BM x BN`` output
    tile and one ``BK`` reduction chunk.
    """
    # pid(0) is the column tile, so programs scan one output tile row from left
    # to right before advancing to the next; adjacent tiles have overlapping
    # L[i,k] source ranges.  With GROUP > 1, they scan columns of GROUP row
    # tiles instead, which then also share each error panel in L2.  The final
    # `out` mask selects the targets.
    column_tile, row_tile, batch = _program_ids(num_column_tiles, num_row_tiles)  # scalars
    if GROUP > 1:
        tile = row_tile * num_column_tiles + column_tile  # scalar, position in the launch order
        first_row_tile = tile // (GROUP * num_column_tiles) * GROUP  # scalar
        group_size = tl.minimum(num_row_tiles - first_row_tile, GROUP)  # scalar
        row_tile = first_row_tile + tile % (GROUP * num_column_tiles) % group_size  # scalar
        column_tile = tile % (GROUP * num_column_tiles) // group_size  # scalar
    s1 = s0 + BLOCK_SIZE  # scalar
    panel_height = tl.cdiv((BLOCK_SIZE + tl.cdiv(BN - 1, q)) * p, BK) * BK  # scalar, as in _error_panel_shape
    matrix_offset = batch * m * n  # scalar, Y/C batch offset
    l_batch_offset = tl.load(l_batch_offsets_ptr + batch) if BATCHED else 0  # scalar
    panel_batch_offset = batch * tl.cdiv(n, BN) * panel_height * BN  # scalar
    i0 = row_tile * BM  # scalar
    j0 = column_tile * BN  # scalar
    # The largest valid antidiagonal in this tile comes from its last valid row and column,
    # and the smallest from its first row and column.
    if (tl.minimum(i0 + BM, m) - 1) // p + (tl.minimum(j0 + BN, n) - 1) // q < s1 or i0 // p + j0 // q >= s_end:
        return
    i = i0 + tl.arange(0, BM)  # [BM]
    j = j0 + tl.arange(0, BN)  # [BN]
    i_ok = i < m  # [BM]
    j_ok = j < n  # [BN]
    # Lowest k needed by any column in this tile.  The panel height covers the
    # union of the shifted bands for all BN columns.
    k_start = (s0 - (j0 + BN - 1) // q) * p  # scalar
    diagonal_block_start = i // p * p  # [BM], first column of the diagonal block of L in row i
    acc = tl.zeros((BM, BN), dtype=y_ptr.dtype.element_ty)  # [BM, BN]

    for k_tile in range(panel_height // BK):
        k = k_start + k_tile * BK + tl.arange(0, BK)  # [BK], source rows
        # Load only below the diagonal blocks; other entries may contain arbitrary data.
        # Stored outputs never need the diagonal blocks: k//p + j//q < s1 <= i//p + j//q.
        l = tl.load(
            l_ptr + l_batch_offset + i[:, None] * m + k[None, :],
            mask=i_ok[:, None] & (k[None, :] >= 0) & (k[None, :] < diagonal_block_start[:, None]),
            other=0.,
        )  # [BM, BK]
        # Since j0=panel*BN, j0*panel_height skips preceding flat panels.
        e_offset = (
            panel_batch_offset
            + j0 * panel_height
            + (k_tile * BK + tl.arange(0, BK)[:, None]) * BN
            + tl.arange(0, BN)[None, :]
        )  # [BK, BN]
        e = tl.load(e_panel_ptr + e_offset)  # [BK, BN]
        acc += tl.dot(l, e, input_precision="ieee")  # [BM, BN]

    # Only the targets receive this feedback.
    stage = (i // p)[:, None] + (j // q)[None, :]  # [BM, BN]
    out = i_ok[:, None] & j_ok[None, :] & (stage >= s1) & (stage < s_end)  # [BM, BN]
    offsets = matrix_offset + i[:, None] * n + j[None, :]  # [BM, BN], C/Y offsets
    _masked_add(c_ptr, offsets, acc, out)  # [BM, BN]
    _masked_add(y_ptr, offsets, acc, out)  # [BM, BN]


def _flush_down(
    y: torch.Tensor,
    c: torch.Tensor,
    e_panel: torch.Tensor,
    l: torch.Tensor,
    l_batch_offsets: torch.Tensor,
    s0: int,
    s1: int,
    s_end: int,
    super_entry_shape: tuple[int, int] = (1, 1),
) -> None:
    BM, BN, BK, GROUP = _flush_down_tile(s1 - s0, super_entry_shape)
    batch_size, m, n = y.shape
    p, q = super_entry_shape
    # Only columns j < s1*q receive down feedback; the block holds rows (s0-j//q)*p..(s1-j//q)*p-1 of column j.
    # The targets end before stage s_end, so rows from s_end*p on receive nothing.
    num_column_tiles: int = triton.cdiv(min(n, s1 * q), BN)
    num_row_tiles: int = triton.cdiv(min(m, s_end * p), BM)
    _flush_down_kernel[_grid(num_column_tiles, num_row_tiles, batch_size)](
        y, c, e_panel, l, l_batch_offsets,
        m=m, n=n, p=p, q=q, s0=s0, s_end=s_end,
        num_column_tiles=num_column_tiles, num_row_tiles=num_row_tiles,
        BATCHED=batch_size > 1,
        BLOCK_SIZE=s1 - s0,
        BM=BM, BN=BN, BK=BK,
        GROUP=GROUP,
        num_warps=4,
    )


@triton.jit
def _flush_right_kernel(
    y_ptr,
    c_ptr,
    u_ptr,
    u_batch_offsets_ptr,
    m,
    n,
    p,
    q,
    s0,
    s_end,
    num_row_tiles,
    num_column_tiles,
    BATCHED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    """Flush the completed block band of C to the right through U.

    The block ends at ``s1=s0+BLOCK_SIZE``, and its targets are the entries
    with ``s1 <= i+j < s_end``.  For every target entry (i,j), this kernel computes

        Y[i,j] += sum_k C[i,k] U[k,j],  for s0 <= i+k < s1.

    C already contains all downward feedback because the down flush ran first.
    This product supplies the block's remaining rightward feedback.

    Each program owns one ``BM x BN`` future output tile:

                         source columns k (BK at a time)
                     +---------------------------+
        C[i, k]       |                           | U[k, j]
        BM x BK       +---------------------------+ BK x BN
                 \\____________ tl.dot ____________/
                                BM x BN

    For one output row i, the required source columns are
    ``s0-i <= k < s1-i``.  These ranges shift left across the BM rows.
    ``k_start`` is the smallest k needed by any row, and ``num_k_tiles`` walks
    their union in BK-sized tiles so it can be expressed as one rectangular
    tl.dot.

    The ``band`` mask prevents C loads outside each row's exact band, while
    matrix-bound masks handle invalid k.  The dot performs some zero products
    but reuses each U tile across BM rows and performs the shared-output
    reduction without atomics.  Only entries satisfying ``s1 <= i+j < s_end``
    are written.
    Programs whose entire output tile lies before ``s1`` or from ``s_end`` on
    return before loading or computing.  The element mask still clips the
    boundary tiles.  The launch grid covers only rows ``i < s1``, since a later
    row would need the source columns ``s0-i <= k < s1-i``, which are all
    negative, and columns ``j < s_end``.

    For p x q super-entries every antidiagonal is coarse, ``i//p + j//q``: row
    i needs the source columns ``(s0 - i//p)*q <= k < (s1 - i//p)*q``, U is
    read only right of its diagonal blocks, and the grid covers the rows
    ``i < s1*p`` and columns ``j < s_end*q``.  The description above is the
    case ``p = q = 1``.

    The batch selects an independent problem.  Backing PyTorch tensors are
    ``Y,C: [B,m,n]`` and ``U: [*Bu,n,n]``.  The U batch offset map implements
    broadcasting.  Shapes below describe one ``BM x BN`` output tile and one
    ``BK`` reduction chunk.
    """
    # pid(0) is the row tile, so programs scan one output tile column from top
    # to bottom before advancing to the next; adjacent tiles have overlapping
    # U[k,j] source ranges.  With GROUP > 1, they scan rows of GROUP column
    # tiles instead, which then also share each C band in L2.
    row_tile, column_tile, batch = _program_ids(num_row_tiles, num_column_tiles)  # scalars
    if GROUP > 1:
        tile = column_tile * num_row_tiles + row_tile  # scalar, position in the launch order
        first_column_tile = tile // (GROUP * num_row_tiles) * GROUP  # scalar
        group_size = tl.minimum(num_column_tiles - first_column_tile, GROUP)  # scalar
        column_tile = first_column_tile + tile % (GROUP * num_row_tiles) % group_size  # scalar
        row_tile = tile % (GROUP * num_row_tiles) // group_size  # scalar
    s1 = s0 + BLOCK_SIZE  # scalar
    num_k_tiles = tl.cdiv((BLOCK_SIZE + tl.cdiv(BM - 1, p)) * q, BK)  # scalar
    matrix_offset = batch * m * n  # scalar, Y/C batch offset
    u_batch_offset = tl.load(u_batch_offsets_ptr + batch) if BATCHED else 0  # scalar
    i0 = row_tile * BM  # scalar
    j0 = column_tile * BN  # scalar
    # The largest valid antidiagonal in this tile comes from its last valid row and column,
    # and the smallest from its first row and column.
    if (tl.minimum(i0 + BM, m) - 1) // p + (tl.minimum(j0 + BN, n) - 1) // q < s1 or i0 // p + j0 // q >= s_end:
        return
    i = i0 + tl.arange(0, BM)  # [BM]
    j = j0 + tl.arange(0, BN)  # [BN]
    i_ok = i < m  # [BM]
    j_ok = j < n  # [BN]
    ibar = i // p  # [BM], super-entry rows
    # Lowest k needed by any row; num_k_tiles spans all BM shifted bands.
    k_start = (s0 - (i0 + BM - 1) // p) * q  # scalar
    diagonal_block_start = j // q * q  # [BN], first row of the diagonal block of U in column j
    acc = tl.zeros((BM, BN), dtype=y_ptr.dtype.element_ty)  # [BM, BN]

    for k_tile in range(num_k_tiles):
        k = k_start + k_tile * BK + tl.arange(0, BK)  # [BK], source columns
        k_ok = (k >= 0) & (k < n)  # [BK]
        stage = ibar[:, None] + (k // q)[None, :]  # [BM, BK], antidiagonal of C[i,k] (meaningful where k >= 0)
        band = (stage >= s0) & (stage < s1)  # [BM, BK]
        c = tl.load(
            c_ptr + matrix_offset + i[:, None] * n + k[None, :],
            mask=i_ok[:, None] & k_ok[None, :] & band,
            other=0.,
        )  # [BM, BK]
        # Load only right of the diagonal blocks; other entries may contain arbitrary data.
        # Stored outputs never need the diagonal blocks: i//p + k//q < s1 <= i//p + j//q.
        u = tl.load(
            u_ptr + u_batch_offset + k[:, None] * n + j[None, :],
            mask=k_ok[:, None] & j_ok[None, :] & (k[:, None] < diagonal_block_start[None, :]),
            other=0.,
        )  # [BK, BN]
        acc += tl.dot(c, u, input_precision="ieee")  # [BM, BN]

    # Only the targets receive the deferred right feedback.
    stage = ibar[:, None] + (j // q)[None, :]  # [BM, BN]
    out = i_ok[:, None] & j_ok[None, :] & (stage >= s1) & (stage < s_end)  # [BM, BN]
    offsets = matrix_offset + i[:, None] * n + j[None, :]  # [BM, BN], Y offsets
    _masked_add(y_ptr, offsets, acc, out)  # [BM, BN]


def _flush_right(
    y: torch.Tensor,
    c: torch.Tensor,
    u: torch.Tensor,
    u_batch_offsets: torch.Tensor,
    s0: int,
    s1: int,
    s_end: int,
    super_entry_shape: tuple[int, int] = (1, 1),
) -> None:
    batch_size, m, n = y.shape
    p, q = super_entry_shape
    # Taller tiles widen each row tile's band by BM/p stages, which blocks of at least 256 rows absorb.
    BM, BN, BK, GROUP = _DEEP_FLUSH_RIGHT_TILE if (s1 - s0) * p >= 256 else _FLUSH_RIGHT_TILE
    # Only rows i < s1*p receive right feedback; the block holds columns (s0-i//p)*q..(s1-i//p)*q-1 of row i.
    # The targets end before stage s_end, so columns from s_end*q on receive nothing.
    num_row_tiles: int = triton.cdiv(min(m, s1 * p), BM)
    num_column_tiles: int = triton.cdiv(min(n, s_end * q), BN)
    _flush_right_kernel[_grid(num_row_tiles, num_column_tiles, batch_size)](
        y, c, u, u_batch_offsets,
        m=m, n=n, p=p, q=q, s0=s0, s_end=s_end,
        num_row_tiles=num_row_tiles, num_column_tiles=num_column_tiles,
        BATCHED=batch_size > 1,
        BLOCK_SIZE=s1 - s0,
        BM=BM, BN=BN, BK=BK,
        GROUP=GROUP,
        num_warps=4,
    )


def _flush_block(
    y: torch.Tensor,
    c: torch.Tensor,
    e_stage: torch.Tensor,
    e_panel_scratch: torch.Tensor,
    l: torch.Tensor,
    l_batch_offsets: torch.Tensor,
    u: torch.Tensor,
    u_batch_offsets: torch.Tensor,
    s0: int,
    s1: int,
    s_end: int,
    super_entry_shape: tuple[int, int] = (1, 1),
) -> None:
    """Apply the errors of the stages ``s0..s1-1``, stored in ``e_stage``, to the stages ``s1..s_end-1``.

    ``e_panel_scratch`` is flat space at least as large as this block's error panels.
    """
    batch_size, m, n = y.shape
    panel_shape: tuple[int, int, int] = _error_panel_shape(n=n, block_size=s1 - s0, super_entry_shape=super_entry_shape)
    e_panel: torch.Tensor = e_panel_scratch[:batch_size * math.prod(panel_shape)].view(batch_size, *panel_shape)  # [B, num_panels, panel_height, BN]
    _repack_errors(e_panel=e_panel, e_stage=e_stage, m=m, n=n, s0=s0, s1=s1, super_entry_shape=super_entry_shape)
    _flush_down(
        y=y, c=c, e_panel=e_panel, l=l, l_batch_offsets=l_batch_offsets,
        s0=s0, s1=s1, s_end=s_end, super_entry_shape=super_entry_shape,
    )
    _flush_right(
        y=y, c=c, u=u, u_batch_offsets=u_batch_offsets,
        s0=s0, s1=s1, s_end=s_end, super_entry_shape=super_entry_shape,
    )


@torch.no_grad()
def direct_2d(
    x: torch.Tensor,
    l: torch.Tensor,
    u: torch.Tensor,
    *,
    quantizer: Callable = round_quantizer,
    inplace: bool = False,
    quantizer_inputs: tuple[torch.Tensor, ...] = (),
) -> torch.Tensor:
    """Run direct quartic 2-D quantization, optionally overwriting x."""
    y, m, n, batch_shape = _prepare_output(x=x, l=l, u=u, inplace=inplace, require_cuda=False)
    if y.numel() == 0:
        return y

    batch_size: int = batch_shape.numel()  # 1 with no batch dimensions
    y_flat: torch.Tensor = y.flatten(start_dim=-2)  # [*batch, m*n], storage-sharing view
    l: torch.Tensor = _unit_triangular(l, upper=False)  # [*Bl, m, m]
    u_flip: torch.Tensor = _unit_triangular(u, upper=True).flip(dims=(-2,))  # [*Bu, n, n], rows in decreasing j order
    z_scratch: torch.Tensor = y.new_empty(batch_size * min(m, n))  # [B*min(m,n)], reused by every antidiagonal
    for s in range(m + n - 1):
        i_slice, j_slice, antidiagonal_slice = _antidiagonal_slices(m=m, n=n, s=s)
        y_antidiagonal, z = _quantize_antidiagonal(
            z_scratch=z_scratch, y_flat=y_flat, selection=antidiagonal_slice, quantizer=quantizer, quantizer_inputs=quantizer_inputs,
        )  # [*batch, antidiag_len] each
        e: torch.Tensor = z - y_antidiagonal  # [*batch, antidiag_len], increasing i and decreasing j
        # L vanishes above row i_slice.start in these columns, and U left of column j_slice.start in these rows
        y[..., i_slice.start:, j_slice.start:] += (l[..., i_slice.start:, i_slice] * e[..., None, :]) @ u_flip[..., n - j_slice.stop : n - j_slice.start, j_slice.start:]  # [*batch, m - i_slice.start, n - j_slice.start]
        y_flat[..., antidiagonal_slice] = z  # [*batch, antidiag_len]
    return y  # [*batch, m, n]


@torch.no_grad()
def direct_2d_super_entry(
    x: torch.Tensor,
    l: torch.Tensor,
    u: torch.Tensor,
    *,
    quantizer: Callable = round_quantizer_super_entry,
    super_entry_shape: tuple[int, int] = (1, 1),
    inplace: bool = False,
    quantizer_inputs: tuple[torch.Tensor, ...] = (),
) -> torch.Tensor:
    """Run direct quartic 2-D quantization over p x q super-entries, optionally overwriting x.

    The inputs and the quantizer follow ``gptq_2d_super_entry_torch``.
    """
    y, m, n, batch_shape = _prepare_output(x=x, l=l, u=u, inplace=inplace, require_cuda=False, super_entry_shape=super_entry_shape)
    if y.numel() == 0:
        return y

    p, q = super_entry_shape
    mbar, nbar = m // p, n // q
    batch_size: int = batch_shape.numel()  # 1 with no batch dimensions
    y_flat: torch.Tensor = y.flatten(start_dim=-2)  # [*batch, m*n], storage-sharing view
    l: torch.Tensor = _unit_triangular(l, upper=False, size=p)  # [*Bl, m, m]
    # Row bands of U, reversed so that each antidiagonal reads one contiguous slice.
    u_bands: torch.Tensor = _unit_triangular(u, upper=True, size=q).unflatten(dim=-2, sizes=(nbar, q)).flip(dims=(-3,))  # [*Bu, nbar, q, n], decreasing jbar
    z_scratch: torch.Tensor = y.new_empty(batch_size * min(mbar, nbar) * p * q)  # reused by every antidiagonal
    for s in range(mbar + nbar - 1):
        i_slice, j_slice, antidiagonal_slice = _antidiagonal_slices(m=m, n=n, s=s, super_entry_shape=super_entry_shape)
        select = functools.partial(_super_entries, antidiagonal_slice=antidiagonal_slice, n=n, super_entry_shape=super_entry_shape)
        y_super, z_super = _quantize_antidiagonal(
            z_scratch=z_scratch, y_flat=y_flat, selection=select, quantizer=quantizer, quantizer_inputs=quantizer_inputs,
        )  # [*batch, d, p, q] each, increasing ibar and decreasing jbar
        i0, j0 = i_slice.start * p, j_slice.start * q
        # Each error goes right through its row band of U, which vanishes left of column j0, and then down
        # through its column band of L, which vanishes above row i0; the antidiagonal's column bands are adjacent.
        right: torch.Tensor = ((z_super - y_super) @ u_bands[..., nbar - j_slice.stop : nbar - j_slice.start, :, j0:]).flatten(start_dim=-3, end_dim=-2)  # [*batch, d*p, n - j0]
        y[..., i0:, j0:] += l[..., i0:, i0 : i_slice.stop * p] @ right  # [*batch, m - i0, n - j0]
        y_super.copy_(z_super)  # [*batch, d, p, q]
    return y  # [*batch, m, n]


@torch.no_grad()
def gptq_2d_torch(
    x: torch.Tensor,
    l: torch.Tensor,
    u: torch.Tensor,
    *,
    quantizer: Callable = round_quantizer,
    inplace: bool = False,
    quantizer_inputs: tuple[torch.Tensor, ...] = (),
) -> torch.Tensor:
    """Run the batched PyTorch implementation, optionally overwriting x."""
    y, m, n, batch_shape = _prepare_output(x=x, l=l, u=u, inplace=inplace, require_cuda=False)
    if y.numel() == 0:
        return y

    batch_size: int = batch_shape.numel()  # 1 with no batch dimensions
    c: torch.Tensor = y.new_zeros(y.shape)  # [*batch, m, n], buffer C = L E
    y_flat, c_flat = y.flatten(start_dim=-2), c.flatten(start_dim=-2)  # each [*batch, m*n], storage-sharing views
    l_flip: torch.Tensor = _unit_triangular(l, upper=False).flip(dims=(-1,))  # [*Bl, m, m], unused upper triangle is zero
    u_flip: torch.Tensor = _unit_triangular(u, upper=True).flip(dims=(-2,))  # [*Bu, n, n], unused lower triangle is zero
    z_scratch: torch.Tensor = y.new_empty(batch_size * min(m, n))  # [B*min(m,n)], reused by every antidiagonal
    for s in range(m + n - 1):
        i_slice, j_slice, antidiagonal_slice = _antidiagonal_slices(m=m, n=n, s=s)
        y_antidiagonal, z = _quantize_antidiagonal(
            z_scratch=z_scratch, y_flat=y_flat, selection=antidiagonal_slice, quantizer=quantizer, quantizer_inputs=quantizer_inputs,
        )  # [*batch, antidiag_len] each
        e: torch.Tensor = z - y_antidiagonal  # [*batch, antidiag_len], errors in ascending-i order
        # L vanishes above row i_slice.start in these columns, and U left of column j_slice.start in these rows
        # down direction: M[:, j] = L[:, i] E_ij, folded into C = LE and pushed down into Y with fused multiply-adds.
        # These updates are memory bound: recomputing M in each addcmul_ is cheaper than writing M once and
        # reading it twice (6 instead of 8 passes over the block, and no temporary).  The right push likewise
        # takes 3 passes instead of 5.
        l_columns: torch.Tensor = l_flip[..., i_slice.start:, m - i_slice.stop : m - i_slice.start]  # [*Bl, m - i_slice.start, antidiag_len], increasing j
        e_row: torch.Tensor = e.flip(dims=(-1,))[..., None, :]  # [*batch, 1, antidiag_len], reverse e for increasing-j order
        c[..., i_slice.start:, j_slice].addcmul_(l_columns, e_row)  # [*batch, m - i_slice.start, antidiag_len], fold into C = LE
        y[..., i_slice.start:, j_slice].addcmul_(l_columns, e_row)  # [*batch, m - i_slice.start, antidiag_len], push down; the diagonal write is harmless
        # right direction: push the updated C anti-diagonal right through U
        u_rows: torch.Tensor = u_flip[..., n - j_slice.stop : n - j_slice.start, j_slice.start:]  # [*Bu, antidiag_len, n - j_slice.start], decreasing j
        y[..., i_slice, j_slice.start:].addcmul_(c_flat[..., antidiagonal_slice, None], u_rows)  # [*batch, antidiag_len, n - j_slice.start]
        y_flat[..., antidiagonal_slice] = z  # [*batch, antidiag_len]
    return y  # [*batch, m, n]


@torch.no_grad()
def gptq_2d_super_entry_torch(
    x: torch.Tensor,
    l: torch.Tensor,
    u: torch.Tensor,
    *,
    quantizer: Callable = round_quantizer_super_entry,
    super_entry_shape: tuple[int, int] = (1, 1),
    inplace: bool = False,
    quantizer_inputs: tuple[torch.Tensor, ...] = (),
) -> torch.Tensor:
    """Run the batched PyTorch implementation over p x q super-entries.

    GPTQ-2D runs on the coarse (m/p) x (n/q) matrix of super-entries, and all
    super-entries of one coarse antidiagonal are rounded jointly by
    ``quantizer(y_flat, select, *quantizer_inputs, out=z)``.  ``select`` takes
    the place of ``antidiagonal_slice`` in ``gptq_2d_torch``: it views any
    ``[..., m*n]`` tensor, such as live Y or a flattened quantizer input, as
    the ``[..., d, p, q]`` super-entries of the antidiagonal in increasing
    coarse row order.  The quantizer must treat its inputs as read-only and
    completely overwrite the contiguous ``out=[*batch, d, p, q]``.  ``l`` and
    ``u`` are unit block lower and upper triangular with implicit identity
    p x p and q x q diagonal blocks: only their entries below and above these
    blocks are read.
    """
    y, m, n, batch_shape = _prepare_output(x=x, l=l, u=u, inplace=inplace, require_cuda=False, super_entry_shape=super_entry_shape)
    if y.numel() == 0:
        return y

    p, q = super_entry_shape
    mbar, nbar = m // p, n // q
    batch_size: int = batch_shape.numel()  # 1 with no batch dimensions
    c: torch.Tensor = y.new_zeros(y.shape)  # [*batch, m, n], buffer C = L E
    y_flat, c_flat = y.flatten(start_dim=-2), c.flatten(start_dim=-2)  # each [*batch, m*n], storage-sharing views
    # Column bands of L and row bands of U, reversed so that each antidiagonal reads one contiguous slice.
    l_bands: torch.Tensor = _unit_triangular(l, upper=False, size=p).unflatten(dim=-1, sizes=(mbar, p)).flip(dims=(-2,)).movedim(-2, -3).contiguous()  # [*Bl, mbar, m, p], decreasing ibar
    u_bands: torch.Tensor = _unit_triangular(u, upper=True, size=q).unflatten(dim=-2, sizes=(nbar, q)).flip(dims=(-3,)).contiguous()  # [*Bu, nbar, q, n], decreasing jbar
    z_scratch: torch.Tensor = y.new_empty(batch_size * min(mbar, nbar) * p * q)  # reused by every antidiagonal
    for s in range(mbar + nbar - 1):
        i_slice, j_slice, antidiagonal_slice = _antidiagonal_slices(m=m, n=n, s=s, super_entry_shape=super_entry_shape)
        select = functools.partial(_super_entries, antidiagonal_slice=antidiagonal_slice, n=n, super_entry_shape=super_entry_shape)
        y_super, z_super = _quantize_antidiagonal(
            z_scratch=z_scratch, y_flat=y_flat, selection=select, quantizer=quantizer, quantizer_inputs=quantizer_inputs,
        )  # [*batch, d, p, q] each, increasing ibar and decreasing jbar
        e: torch.Tensor = z_super - y_super  # [*batch, d, p, q], errors in increasing-ibar order
        # The L bands vanish above row i_slice.start * p, and the U bands left of column j_slice.start * q
        # down direction: M[:, jbar band] = L[:, ibar band] E, folded into C = LE and pushed down into Y
        update: torch.Tensor = l_bands[..., mbar - i_slice.stop : mbar - i_slice.start, i_slice.start * p:, :] @ e.flip(dims=(-3,))  # [*batch, d, m - i_slice.start * p, q], increasing jbar
        c.unflatten(dim=-1, sizes=(nbar, q))[..., i_slice.start * p:, j_slice, :] += update.transpose(-3, -2)  # [*batch, m - i_slice.start * p, d, q], fold into C = LE
        y.unflatten(dim=-1, sizes=(nbar, q))[..., i_slice.start * p:, j_slice, :] += update.transpose(-3, -2)  # push down; the band write is harmless
        # right direction: push the completed C super-entries right through the U bands
        y.unflatten(dim=-2, sizes=(mbar, p))[..., i_slice, :, j_slice.start * q:] += select(c_flat) @ u_bands[..., nbar - j_slice.stop : nbar - j_slice.start, :, j_slice.start * q:]  # [*batch, d, p, n - j_slice.start * q]
        y_super.copy_(z_super)  # [*batch, d, p, q]
    return y  # [*batch, m, n]


def _gptq_2d_blocked(
    x: torch.Tensor,
    l: torch.Tensor,
    u: torch.Tensor,
    quantizer: Callable,
    super_entry_shape: tuple[int, int] | None,
    inner_block_size: int,
    outer_block_size: int,
    inplace: bool,
    quantizer_inputs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Run blocked GPTQ-2D for ``gptq_2d`` (``super_entry_shape=None``) or ``gptq_2d_super_entry``.

    The two differ only in what the quantizer receives: a flat slice or a
    selector.  Plain entries, also as 1 x 1 super-entries, use the fused push
    kernel; larger super-entries push down and right with ``tl.dot``.
    """
    p, q = super_entry_shape or (1, 1)
    y, m, n, batch_shape = _prepare_output(x=x, l=l, u=u, inplace=inplace, require_cuda=True, super_entry_shape=(p, q))
    if y.numel() == 0:
        return y
    batch_size: int = batch_shape.numel()  # 1 with no batch dimensions
    lanes: int = min(m // p, n // q)  # super-entries of the longest antidiagonal

    y_flat: torch.Tensor = y.flatten(start_dim=-2)  # [*batch, m*n], storage-sharing view
    y_batches: torch.Tensor = y.reshape(batch_size, m, n)  # [B, m, n]
    l, u = l.contiguous(), u.contiguous()  # [*Bl, m, m], [*Bu, n, n]
    l_batch_offsets: torch.Tensor = _batch_offsets(x=l, batch_shape=batch_shape)  # [B]
    u_batch_offsets: torch.Tensor = _batch_offsets(x=u, batch_shape=batch_shape)  # [B]
    c: torch.Tensor = y_batches.new_zeros(y_batches.shape)  # [B, m, n], running buffer C = L E
    z_scratch: torch.Tensor = y.new_empty(batch_size * lanes * p * q)  # reused by every antidiagonal
    old_scratch: torch.Tensor = y.new_empty(2, batch_size * lanes)  # [2, B*lanes], fused push: Y on two antidiagonals before quantization
    stages: int = m // p + n // q - 1
    inner_block_size: int = inner_block_size if 0 < inner_block_size <= stages else stages
    outer_block_size: int = outer_block_size if inner_block_size < outer_block_size < stages else inner_block_size
    # Flushes read the errors of the current outer block.  A single block holding all stages is never
    # flushed, so it reuses one error row instead of allocating unused O(stages * lanes) space.
    num_error_rows: int = outer_block_size if outer_block_size < stages else 1
    e_stage: torch.Tensor = y_batches.new_empty(num_error_rows, batch_size, lanes, p, q)  # [num_error_rows, B, lanes, p, q]
    e_panel_scratch: torch.Tensor = y_batches.new_empty(batch_size * math.prod(_error_panel_shape(n=n, block_size=num_error_rows, super_entry_shape=(p, q))))
    flush: Callable[..., None] = functools.partial(
        _flush_block, y=y_batches, c=c, e_panel_scratch=e_panel_scratch,
        l=l, l_batch_offsets=l_batch_offsets, u=u, u_batch_offsets=u_batch_offsets, super_entry_shape=(p, q),
    )
    fused: bool = p == q == 1

    with torch.cuda.device(x.device):
        for outer_start in range(0, stages, outer_block_size):
            outer_stop: int = min(outer_start + outer_block_size, stages)
            for s0 in range(outer_start, outer_stop, inner_block_size):
                s1: int = min(s0 + inner_block_size, outer_stop)
                if fused:  # the flushes completed the block's first antidiagonal; each push completes the next one
                    y_first: torch.Tensor = y_flat[..., _antidiagonal_slices(m=m, n=n, s=s0)[2]]  # [*batch, antidiag_len]
                    old_scratch[s0 % 2, :y_first.numel()].view_as(y_first).copy_(y_first)
                for s in range(s0, s1):
                    _, _, antidiagonal_slice = _antidiagonal_slices(m=m, n=n, s=s, super_entry_shape=(p, q))
                    if super_entry_shape is not None:
                        antidiagonal_slice = functools.partial(_super_entries, antidiagonal_slice=antidiagonal_slice, n=n, super_entry_shape=(p, q))
                    _, z = _quantize_antidiagonal(
                        z_scratch=z_scratch, y_flat=y_flat, selection=antidiagonal_slice, quantizer=quantizer, quantizer_inputs=quantizer_inputs,
                    )  # [*batch, antidiag_len] or [*batch, antidiag_len, p, q], contiguous
                    e: torch.Tensor = e_stage[s - outer_start if outer_block_size < stages else 0, :, :, :, :]  # [B, lanes, p, q]
                    if fused:
                        _push(
                            y=y_batches, c=c, e=e, z=z, old=old_scratch[s % 2, :], next_old=old_scratch[(s + 1) % 2, :],
                            l=l, l_batch_offsets=l_batch_offsets, u=u, u_batch_offsets=u_batch_offsets, s=s, s1=s1,
                        )
                    else:
                        _push_down_super_entry(
                            y=y_batches, c=c, e=e, z=z, l=l, l_batch_offsets=l_batch_offsets, s=s, s1=s1, super_entry_shape=(p, q),
                        )
                        _push_right_super_entry(
                            y=y_batches, c=c, z=z, u=u, u_batch_offsets=u_batch_offsets, s=s, s1=s1, super_entry_shape=(p, q),
                        )
                if fused and s1 < stages:  # the flush reads C on the block
                    _add_own_errors(c=c, e_stage=e_stage[s0 - outer_start:s1 - outer_start, :, :, :, :], s0=s0, s1=s1)
                if s1 < outer_stop:  # flush the block into the rest of its outer block
                    flush(e_stage=e_stage[s0 - outer_start:s1 - outer_start, :, :, :, :], s0=s0, s1=s1, s_end=outer_stop)
            if outer_stop < stages:  # flush the outer block into all later stages
                flush(e_stage=e_stage[:outer_stop - outer_start, :, :, :, :], s0=outer_start, s1=outer_stop, s_end=stages)
    return y  # [*batch, m, n]


@torch.no_grad()
def gptq_2d(
    x: torch.Tensor,
    l: torch.Tensor,
    u: torch.Tensor,
    *,
    quantizer: Callable = round_quantizer,
    inner_block_size: int = 32,
    outer_block_size: int = 512,
    inplace: bool = False,
    quantizer_inputs: tuple[torch.Tensor, ...] = (),
) -> torch.Tensor:
    """Run blocked GPTQ-2D with ``quantizer`` as a separate Python callable.

    The last two dimensions are matrix dimensions; leading dimensions of x, l,
    and u follow PyTorch broadcasting rules.  ``l`` and ``u`` are unit lower
    and upper triangular with implicit unit diagonals: only their entries
    below and above the diagonal are read.  The quantizer receives live Y,
    whose earlier antidiagonals already hold their quantized values, with its
    matrix dimensions flattened, ``[*batch,m*n]``, and a slice selecting the
    current antidiagonal, followed by ``quantizer_inputs`` and a contiguous
    ``out=[*batch,antidiag_len]`` keyword argument.  It must treat its inputs
    as read-only and completely overwrite ``out``.  If ``inplace`` is true, x
    must already have the exact broadcast shape and be contiguous.

    Blocking has two levels.  Each antidiagonal pushes its feedback only
    inside its block of ``inner_block_size`` stages.  A finished block flushes
    into the rest of its outer block of ``outer_block_size`` stages, and a
    finished outer block flushes into all later stages.  An
    ``inner_block_size`` outside ``1..m+n-1`` selects one block containing all
    stages, and an ``outer_block_size`` outside ``inner_block_size+1..m+n-2``,
    such as 0, gives single-level blocking.  Inner blocks restart at each outer
    block.  If ``outer_block_size`` is not a multiple of ``inner_block_size``,
    the blocks thus differ from those of the paper's Algorithm 5, which splits
    each level at multiples of its width, but the result is the same in exact
    arithmetic.
    """
    return _gptq_2d_blocked(
        x=x, l=l, u=u, quantizer=quantizer, super_entry_shape=None, inner_block_size=inner_block_size,
        outer_block_size=outer_block_size, inplace=inplace, quantizer_inputs=quantizer_inputs,
    )


@torch.no_grad()
def gptq_2d_super_entry(
    x: torch.Tensor,
    l: torch.Tensor,
    u: torch.Tensor,
    *,
    quantizer: Callable = round_quantizer_super_entry,
    super_entry_shape: tuple[int, int] = (1, 1),
    inner_block_size: int = 8,
    outer_block_size: int = 64,
    inplace: bool = False,
    quantizer_inputs: tuple[torch.Tensor, ...] = (),
) -> torch.Tensor:
    """Run blocked GPTQ-2D over p x q super-entries with Triton kernels.

    The inputs and the quantizer follow ``gptq_2d_super_entry_torch``, and the
    two blocking levels follow ``gptq_2d`` with stages counted in coarse
    antidiagonals of super-entries.  Large super-entries such as 64 x 64 have
    few stages and run slightly faster with single-level blocking,
    ``outer_block_size=0``.  Each push program holds a whole super-entry in one
    ``tl.dot`` tile, so super-entries beyond 64 x 64 spill registers or exceed
    shared memory; ``gptq_2d_super_entry_torch`` suits them better.
    """
    return _gptq_2d_blocked(
        x=x, l=l, u=u, quantizer=quantizer, super_entry_shape=super_entry_shape, inner_block_size=inner_block_size,
        outer_block_size=outer_block_size, inplace=inplace, quantizer_inputs=quantizer_inputs,
    )


@torch.no_grad()
def capture_gptq_2d(
    x: torch.Tensor,
    l: torch.Tensor,
    u: torch.Tensor,
    *,
    quantizer: Callable | None = None,
    super_entry_shape: tuple[int, int] | None = None,
    inner_block_size: int | None = None,
    outer_block_size: int | None = None,
    inplace: bool = False,
    quantizer_inputs: tuple[torch.Tensor, ...] = (),
) -> Callable[..., torch.Tensor]:
    """Capture GPTQ-2D and return a callable that replays it on new inputs.

    The graph runs ``gptq_2d``, or ``gptq_2d_super_entry`` when
    ``super_entry_shape`` is given; ``quantizer``, ``inner_block_size``, and
    ``outer_block_size`` default to those of the captured function.  Replay
    receives X, L, U, then the tensors from ``quantizer_inputs`` in the same
    order.  They must match the capture examples' shapes, dtypes, and devices
    and are copied into private static buffers before replay.  The quantizer
    must use CUDA-graph-safe tensor operations: its Python code runs during
    warmup and capture, but not during replay.

    Out-of-place replay clones the graph's static result so a later replay
    cannot overwrite it.  With ``inplace=True``, the result is instead copied
    back into the caller's contiguous X and that tensor is returned.
    One replay callable owns shared static buffers and must be used sequentially.
    """
    _validate_inputs(x=x, l=l, u=u, inplace=inplace, require_cuda=True, super_entry_shape=super_entry_shape or (1, 1))
    assert all(tensor.is_cuda and tensor.device == x.device for tensor in quantizer_inputs), (
        "quantizer inputs must be CUDA tensors on the same device as X"
    )
    # Options left as None keep the defaults of the captured function.
    options = dict(
        quantizer=quantizer, super_entry_shape=super_entry_shape,
        inner_block_size=inner_block_size, outer_block_size=outer_block_size,
    )
    run = functools.partial(
        gptq_2d if super_entry_shape is None else gptq_2d_super_entry,
        inplace=inplace, **{name: value for name, value in options.items() if value is not None},
    )

    names = ("X", "L", "U", *(f"quantizer input {i}" for i in range(len(quantizer_inputs))))
    inputs = (x, l, u, *quantizer_inputs)
    expected = tuple((tensor.shape, tensor.dtype, tensor.device) for tensor in inputs)

    def validate_replay(x: torch.Tensor, l: torch.Tensor, u: torch.Tensor, *quantizer_inputs: torch.Tensor) -> None:
        replay_inputs = (x, l, u, *quantizer_inputs)
        assert len(replay_inputs) == len(expected), f"expected {len(expected) - 3} quantizer inputs"
        for name, tensor, (shape, dtype, device) in zip(names, replay_inputs, expected):
            assert tensor.shape == shape, f"{name} shape must match its capture example"
            assert tensor.dtype == dtype, f"{name} dtype must match its capture example"
            assert tensor.device == device, f"{name} device must match its capture example"
        if inplace:
            assert x.is_contiguous(), "inplace=True requires contiguous X"

    if x.numel() == 0 or l.numel() == 0 or u.numel() == 0:
        # Validate the example normally, but avoid capturing a graph with no CUDA work.
        run(x=x, l=l, u=u, quantizer_inputs=quantizer_inputs)

        @torch.no_grad()
        def replay_empty(
            x: torch.Tensor,
            l: torch.Tensor,
            u: torch.Tensor,
            *replay_quantizer_inputs: torch.Tensor,
        ) -> torch.Tensor:
            validate_replay(x, l, u, *replay_quantizer_inputs)
            return run(x=x, l=l, u=u, quantizer_inputs=replay_quantizer_inputs)

        return replay_empty

    static_inputs = tuple(
        tensor.detach().clone(memory_format=torch.contiguous_format)
        for tensor in inputs
    )
    static_x, static_l, static_u = static_inputs[:3]
    static_quantizer_inputs = static_inputs[3:]

    num_warmup: int = 2
    with torch.cuda.device(x.device):
        capture_stream: torch.cuda.Stream = torch.cuda.Stream(device=x.device)
        capture_stream.wait_stream(torch.cuda.current_stream())  # the static inputs' copies may still be pending there
        with torch.cuda.stream(capture_stream):
            for _ in range(num_warmup):
                run(x=static_x, l=static_l, u=static_u, quantizer_inputs=static_quantizer_inputs)
            static_x.copy_(x)
        torch.cuda.synchronize()

        graph: torch.cuda.CUDAGraph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph=graph, stream=capture_stream):
            static_y: torch.Tensor = run(x=static_x, l=static_l, u=static_u, quantizer_inputs=static_quantizer_inputs)

    @torch.no_grad()
    def replay(
        x: torch.Tensor,
        l: torch.Tensor,
        u: torch.Tensor,
        *replay_quantizer_inputs: torch.Tensor,
    ) -> torch.Tensor:
        validate_replay(x, l, u, *replay_quantizer_inputs)

        static_x.copy_(x)
        static_l.copy_(l)
        static_u.copy_(u)
        for static_input, replay_input in zip(static_quantizer_inputs, replay_quantizer_inputs):
            static_input.copy_(replay_input)
        graph.replay()
        if inplace:
            x.copy_(static_y)
            return x
        return static_y.clone()

    return replay
