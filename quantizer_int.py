"""Quantizers for GPTQ-2D.

The affine variants write ``(clamp(round(y / scale + zero_point), qmin,
qmax) - zero_point) * scale`` and require nonzero scale values;
``int_abs_max_scale_torch`` and ``int_abs_max_scale_triton`` give GPTQ's symmetric ones.
"""

from collections.abc import Callable
import functools
import unittest

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


def round_quantizer(
    y_flat: torch.Tensor,
    antidiagonal_slice: slice,
    *,
    out: torch.Tensor,
) -> None:
    """Round the selected antidiagonal into ``out``."""
    torch.round(y_flat[..., antidiagonal_slice], out=out)


def round_quantizer_super_entry(
    y_flat: torch.Tensor,
    select: Callable[[torch.Tensor], torch.Tensor],
    *,
    out: torch.Tensor,
) -> None:
    """Round the selected antidiagonal of super-entries into ``out``."""
    torch.round(select(y_flat), out=out)


def _validate_affine_inputs(
    y_flat: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    out: torch.Tensor,
    qmin: float,
    qmax: float,
) -> torch.Size:
    """Validate shared affine-quantizer inputs and return Y's batch shape."""
    assert qmin < qmax, "expected qmin < qmax"
    assert scale.dim() >= 2 and zero_point.dim() >= 2, "scale and zero point must have matrix dimensions"
    assert scale.shape[-2:] == zero_point.shape[-2:], "scale and zero point matrix shapes must match"
    assert scale.shape[-2:].numel() == y_flat.size(-1), "scale and zero point matrix shapes must match Y"
    assert scale.dtype == zero_point.dtype == y_flat.dtype, "scale, zero point, and Y must share dtype"
    assert scale.device == zero_point.device == y_flat.device, "scale, zero point, and Y must share device"
    batch_shape: torch.Size = y_flat.shape[:-1]
    assert torch.broadcast_shapes(scale.shape[:-2], zero_point.shape[:-2], batch_shape) == batch_shape, (
        "scale and zero point batch dimensions must broadcast to Y"
    )
    assert out.shape[:-1] == batch_shape and out.dtype == y_flat.dtype and out.device == y_flat.device, (
        "out must match Y's batch shape, dtype, and device"
    )
    return batch_shape


def affine_quantizer_torch(
    y_flat: torch.Tensor,
    antidiagonal_slice: slice,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    *,
    out: torch.Tensor,
    qmin: float = -torch.inf,
    qmax: float = torch.inf,
) -> None:
    """Affine-quantize one antidiagonal into ``out`` with PyTorch."""
    _validate_affine_inputs(
        y_flat=y_flat, scale=scale, zero_point=zero_point, out=out,
        qmin=qmin, qmax=qmax,
    )
    y = y_flat[..., antidiagonal_slice]  # [*batch, antidiag_len]
    scale = scale.flatten(start_dim=-2)[..., antidiagonal_slice]  # [*Bs, antidiag_len]
    zero_point = zero_point.flatten(start_dim=-2)[..., antidiagonal_slice]  # [*Bz, antidiag_len]
    torch.div(y, scale, out=out)  # [*batch, antidiag_len]
    out.add_(zero_point).round_().clamp_(qmin, qmax)  # [*batch, antidiag_len]
    out.sub_(zero_point).mul_(scale)  # [*batch, antidiag_len]


def affine_quantizer_super_entry_torch(
    y_flat: torch.Tensor,
    select: Callable[[torch.Tensor], torch.Tensor],
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    *,
    out: torch.Tensor,
    qmin: float = -torch.inf,
    qmax: float = torch.inf,
) -> None:
    """Affine-quantize the selected antidiagonal of super-entries into ``out`` with PyTorch, entry by entry
    with the scales and zero points that ``select`` picks from theirs like from Y."""
    _validate_affine_inputs(
        y_flat=y_flat, scale=scale, zero_point=zero_point, out=out.flatten(start_dim=-3),
        qmin=qmin, qmax=qmax,
    )
    y = select(y_flat)  # [*batch, d, p, q]
    scale = select(scale.flatten(start_dim=-2))  # [*Bs, d, p, q]
    zero_point = select(zero_point.flatten(start_dim=-2))  # [*Bz, d, p, q]
    torch.div(y, scale, out=out)  # [*batch, d, p, q]
    out.add_(zero_point).round_().clamp_(qmin, qmax)  # [*batch, d, p, q]
    out.sub_(zero_point).mul_(scale)  # [*batch, d, p, q]


def int_abs_max_scale_torch(x: torch.Tensor, *, qmin: float, qmax: float, dtype: torch.dtype) -> torch.Tensor:
    """GPTQ's symmetric scale of each row of x for the codes qmin..qmax: 2 absmax / (qmax - qmin) over the
    last dimension, rounded to ``dtype`` and returned in x's dtype as [..., 1], or 1 for a row of zeros."""
    # absmax / ((qmax - qmin) / 2) rounds the same exact quotient, and the divisor is not a Python scalar,
    # which CUDA divides by through its reciprocal
    divisor = x.new_full((), (qmax - qmin) / 2)
    scale = torch.linalg.vector_norm(x, ord=torch.inf, dim=-1, keepdim=True).div_(divisor).to(dtype).to(x.dtype)
    return scale.masked_fill_(scale == 0, 1.)


def int_abs_max_quantizer_torch(
    y_flat: torch.Tensor,
    select: Callable[[torch.Tensor], torch.Tensor],
    *,
    out: torch.Tensor,
    qmin: float,
    qmax: float,
    dtype: torch.dtype,
) -> None:
    """Round each row of the selected super-entries to the integers qmin..qmax times a scale from its
    current values, 2 absmax / (qmax - qmin) rounded to ``dtype``, as GPTQ's symmetric format does with a
    scale per group, with PyTorch."""
    x = select(y_flat)  # [*batch, d, p, q]
    scale = int_abs_max_scale_torch(x, qmin=qmin, qmax=qmax, dtype=dtype)  # [*batch, d, p, 1]
    torch.div(x, scale, out=out).round_().clamp_(qmin, qmax).mul_(scale)


def _batch_metadata(x: torch.Tensor, batch_shape: torch.Size) -> tuple[tuple[int, ...], tuple[int, ...]]:
    padding: int = len(batch_shape) - x.dim() + 2
    return (1,) * padding + x.shape[:-2], (0,) * padding + x.stride()[:-2]


@triton.jit
def _batch_offset(
    batch,
    BATCH_SHAPE: tl.constexpr,
    X_BATCH_SHAPE: tl.constexpr,
    X_BATCH_STRIDES: tl.constexpr,
):
    """Map a flattened output batch index to one broadcasted input offset."""
    offset = 0
    divisor: tl.constexpr = 1
    for axis in tl.static_range(len(BATCH_SHAPE) - 1, -1, -1):
        coordinate = batch // divisor % BATCH_SHAPE[axis]
        if X_BATCH_SHAPE[axis] != 1:
            offset += coordinate * X_BATCH_STRIDES[axis]
        divisor *= BATCH_SHAPE[axis]
    return offset


@triton.jit
def _affine_quantize_kernel(
    out_ptr,
    y_ptr,
    scale_ptr,
    zero_point_ptr,
    matrix_size,
    start,
    stop,
    stride,
    num_tiles,
    QMIN: tl.constexpr,
    QMAX: tl.constexpr,
    BATCH_SHAPE: tl.constexpr,
    SCALE_BATCH_SHAPE: tl.constexpr,
    SCALE_BATCH_STRIDES: tl.constexpr,
    ZERO_BATCH_SHAPE: tl.constexpr,
    ZERO_BATCH_STRIDES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Affine-quantize and dequantize one antidiagonal in one fused pass."""
    pid = tl.program_id(0)
    length = tl.cdiv(stop - start, stride)  # scalar
    batch = (pid // num_tiles).to(tl.int64)  # scalar
    k = pid % num_tiles * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # [BLOCK_SIZE]
    valid = k < length  # [BLOCK_SIZE]
    position = start + k.to(tl.int64) * stride  # [BLOCK_SIZE]

    scale_offset = _batch_offset(
        batch,
        BATCH_SHAPE=BATCH_SHAPE,
        X_BATCH_SHAPE=SCALE_BATCH_SHAPE,
        X_BATCH_STRIDES=SCALE_BATCH_STRIDES,
    )  # scalar
    zero_offset = _batch_offset(
        batch,
        BATCH_SHAPE=BATCH_SHAPE,
        X_BATCH_SHAPE=ZERO_BATCH_SHAPE,
        X_BATCH_STRIDES=ZERO_BATCH_STRIDES,
    )  # scalar

    y = tl.load(y_ptr + batch * matrix_size + position, mask=valid)  # [BLOCK_SIZE]
    scale = tl.load(scale_ptr + scale_offset + position, mask=valid)  # [BLOCK_SIZE]
    zero_point = tl.load(zero_point_ptr + zero_offset + position, mask=valid)  # [BLOCK_SIZE]
    code = tl.clamp(libdevice.rint(libdevice.div_rn(y, scale) + zero_point), QMIN, QMAX)  # [BLOCK_SIZE]
    (out_ptr + batch * length + k).store((code - zero_point) * scale, mask=valid)


def affine_quantizer_triton(
    y_flat: torch.Tensor,
    antidiagonal_slice: slice,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    *,
    out: torch.Tensor,
    qmin: float = -torch.inf,
    qmax: float = torch.inf,
) -> None:
    """Affine-quantize one antidiagonal into ``out`` with Triton."""
    batch_shape: torch.Size = _validate_affine_inputs(
        y_flat=y_flat, scale=scale, zero_point=zero_point, out=out,
        qmin=qmin, qmax=qmax,
    )
    assert y_flat.is_cuda, "the Triton affine quantizer requires CUDA tensors"
    assert y_flat.is_contiguous() and scale.is_contiguous() and zero_point.is_contiguous() and out.is_contiguous(), (
        "the Triton affine quantizer requires contiguous Y, scale, zero point, and out"
    )
    assert out.shape == y_flat[..., antidiagonal_slice].shape, "out must match the selected antidiagonal"
    num_tiles: int = triton.cdiv(out.size(-1), 256)
    scale_shape, scale_strides = _batch_metadata(x=scale, batch_shape=batch_shape)
    zero_shape, zero_strides = _batch_metadata(x=zero_point, batch_shape=batch_shape)
    with torch.cuda.device(y_flat.device):
        _affine_quantize_kernel[(batch_shape.numel() * num_tiles,)](
            out, y_flat, scale, zero_point,
            matrix_size=y_flat.size(-1),
            start=antidiagonal_slice.start, stop=antidiagonal_slice.stop, stride=antidiagonal_slice.step,
            num_tiles=num_tiles,
            QMIN=qmin, QMAX=qmax,
            BATCH_SHAPE=tuple(batch_shape),
            SCALE_BATCH_SHAPE=scale_shape, SCALE_BATCH_STRIDES=scale_strides,
            ZERO_BATCH_SHAPE=zero_shape, ZERO_BATCH_STRIDES=zero_strides,
            BLOCK_SIZE=256,
            num_warps=4,
            enable_reflect_ftz=False,
        )


@triton.jit
def _affine_quantize_super_entry_kernel(
    out_ptr,
    y_ptr,
    scale_ptr,
    zero_point_ptr,
    matrix_size,
    length,
    num_tiles,
    stride_d,
    stride_p,
    stride_q,
    P: tl.constexpr,
    Q: tl.constexpr,
    QMIN: tl.constexpr,
    QMAX: tl.constexpr,
    BATCH_SHAPE: tl.constexpr,
    SCALE_BATCH_SHAPE: tl.constexpr,
    SCALE_BATCH_STRIDES: tl.constexpr,
    ZERO_BATCH_SHAPE: tl.constexpr,
    ZERO_BATCH_STRIDES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Affine-quantize and dequantize the selected super-entries of one antidiagonal in one fused pass.

    The pointers start at each tensor's first selected entry; entry (k, r, c) of super-entry k lies
    k*stride_d + r*stride_p + c*stride_q further, in Y, scale, and zero point alike.
    """
    pid = tl.program_id(0)
    batch = (pid // num_tiles).to(tl.int64)  # scalar
    e = pid % num_tiles * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # [BLOCK_SIZE], selected entries row by row
    valid = e < length  # [BLOCK_SIZE]
    k, r, c = e // (P * Q), e // Q % P, e % Q  # [BLOCK_SIZE] each
    position = k.to(tl.int64) * stride_d + r * stride_p + c * stride_q  # [BLOCK_SIZE]

    scale_offset = _batch_offset(
        batch,
        BATCH_SHAPE=BATCH_SHAPE,
        X_BATCH_SHAPE=SCALE_BATCH_SHAPE,
        X_BATCH_STRIDES=SCALE_BATCH_STRIDES,
    )  # scalar
    zero_offset = _batch_offset(
        batch,
        BATCH_SHAPE=BATCH_SHAPE,
        X_BATCH_SHAPE=ZERO_BATCH_SHAPE,
        X_BATCH_STRIDES=ZERO_BATCH_STRIDES,
    )  # scalar

    y = tl.load(y_ptr + batch * matrix_size + position, mask=valid)  # [BLOCK_SIZE]
    scale = tl.load(scale_ptr + scale_offset + position, mask=valid)  # [BLOCK_SIZE]
    zero_point = tl.load(zero_point_ptr + zero_offset + position, mask=valid)  # [BLOCK_SIZE]
    code = tl.clamp(libdevice.rint(libdevice.div_rn(y, scale) + zero_point), QMIN, QMAX)  # [BLOCK_SIZE]
    (out_ptr + batch * length + e).store((code - zero_point) * scale, mask=valid)


def affine_quantizer_super_entry_triton(
    y_flat: torch.Tensor,
    select: Callable[[torch.Tensor], torch.Tensor],
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    *,
    out: torch.Tensor,
    qmin: float = -torch.inf,
    qmax: float = torch.inf,
) -> None:
    """Affine-quantize the selected antidiagonal of super-entries into ``out`` with Triton.

    The same as ``affine_quantizer_super_entry_torch``, for a ``select`` that returns views, as
    ``gptq_2d_super_entry``'s does: the kernel reads the entries through the views' strides.
    """
    batch_shape: torch.Size = _validate_affine_inputs(
        y_flat=y_flat, scale=scale, zero_point=zero_point, out=out.flatten(start_dim=-3),
        qmin=qmin, qmax=qmax,
    )
    assert y_flat.is_cuda, "the Triton affine quantizer requires CUDA tensors"
    assert y_flat.is_contiguous() and scale.is_contiguous() and zero_point.is_contiguous() and out.is_contiguous(), (
        "the Triton affine quantizer requires contiguous Y, scale, zero point, and out"
    )
    sources = y_flat, scale.flatten(start_dim=-2), zero_point.flatten(start_dim=-2)
    selected = tuple(select(source) for source in sources)  # [*batch, d, p, q], [*Bs, d, p, q], [*Bz, d, p, q]
    assert all(view.untyped_storage().data_ptr() == source.untyped_storage().data_ptr() for view, source in zip(selected, sources)), (
        "select must return views"
    )
    assert selected[0].stride()[-3:] == selected[1].stride()[-3:] == selected[2].stride()[-3:], "Y, scale, and zero point must share their layout"
    assert out.shape == selected[0].shape, "out must match the selected super-entries"
    d, p, q = selected[0].shape[-3:]
    num_tiles: int = triton.cdiv(d * p * q, 256)
    scale_shape, scale_strides = _batch_metadata(x=scale, batch_shape=batch_shape)
    zero_shape, zero_strides = _batch_metadata(x=zero_point, batch_shape=batch_shape)
    with torch.cuda.device(y_flat.device):
        _affine_quantize_super_entry_kernel[(batch_shape.numel() * num_tiles,)](
            out, *selected,
            matrix_size=y_flat.size(-1),
            length=d * p * q,
            num_tiles=num_tiles,
            stride_d=selected[0].stride(-3), stride_p=selected[0].stride(-2), stride_q=selected[0].stride(-1),
            P=p, Q=q,
            QMIN=qmin, QMAX=qmax,
            BATCH_SHAPE=tuple(batch_shape),
            SCALE_BATCH_SHAPE=scale_shape, SCALE_BATCH_STRIDES=scale_strides,
            ZERO_BATCH_SHAPE=zero_shape, ZERO_BATCH_STRIDES=zero_strides,
            BLOCK_SIZE=256,
            num_warps=4,
            enable_reflect_ftz=False,
        )


_SCALE_DTYPES = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}  # the dtypes the Triton scales round to


@triton.jit
def _int_abs_max_scale(absmax, RANGE: tl.constexpr, DTYPE: tl.constexpr):
    """2 absmax / RANGE, rounded to the 16-bit DTYPE through float32 as PyTorch converts and returned in
    absmax's dtype, or 1 for a row of zeros."""
    scale = libdevice.div_rn(absmax, tl.full(absmax.shape, RANGE / 2, absmax.dtype))  # not 2 absmax / RANGE, which can overflow
    scale = scale.to(tl.float32).to(DTYPE, fp_downcast_rounding="rtne").to(absmax.dtype)
    return tl.where(scale == 0, 1., scale)


@triton.jit
def _int_abs_max_scale_kernel(
    out_ptr,
    x_ptr,
    num_rows,
    size,
    row_stride,
    column_stride,
    RANGE: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """The scales of BLOCK_ROWS rows in one pass."""
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)  # [BLOCK_ROWS]
    columns = tl.arange(0, BLOCK_SIZE)  # [BLOCK_SIZE]
    valid = rows < num_rows  # [BLOCK_ROWS]
    offsets = rows[:, None].to(tl.int64) * row_stride + columns[None, :] * column_stride  # [BLOCK_ROWS, BLOCK_SIZE]
    x = tl.load(x_ptr + offsets, mask=valid[:, None] & (columns[None, :] < size), other=0.)  # [BLOCK_ROWS, BLOCK_SIZE]
    tl.store(out_ptr + rows, _int_abs_max_scale(tl.max(tl.abs(x), axis=1), RANGE=RANGE, DTYPE=DTYPE), mask=valid)


def int_abs_max_scale_triton(x: torch.Tensor, *, qmin: float, qmax: float, dtype: torch.dtype) -> torch.Tensor:
    """The same as ``int_abs_max_scale_torch`` for a 16-bit ``dtype`` with Triton, reading the rows of x through a
    view when its leading dimensions merge, as for the super-entries of one row each."""
    assert x.is_cuda, "the Triton scale requires a CUDA tensor"
    assert dtype in _SCALE_DTYPES, "the Triton scale rounds to fp16 or bf16"
    rows = x.reshape(-1, x.size(-1))  # [num_rows, size]
    out = x.new_empty(*x.shape[:-1], 1)
    with torch.cuda.device(x.device):
        _int_abs_max_scale_kernel[(triton.cdiv(rows.size(0), 64),)](
            out, rows,
            num_rows=rows.size(0), size=rows.size(1), row_stride=rows.stride(0), column_stride=rows.stride(1),
            RANGE=float(qmax - qmin), DTYPE=_SCALE_DTYPES[dtype],
            BLOCK_ROWS=64, BLOCK_SIZE=triton.next_power_of_2(rows.size(1)),
            num_warps=4,
            enable_reflect_ftz=False,
        )
    return out


@triton.jit
def _int_abs_max_quantize_kernel(
    out_ptr,
    y_ptr,
    matrix_size,
    num_rows,
    num_tiles,
    stride_d,
    stride_p,
    stride_q,
    P: tl.constexpr,
    Q: tl.constexpr,
    QMIN: tl.constexpr,
    QMAX: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    """Round BLOCK_ROWS rows of the selected super-entries of one antidiagonal with their scales in one pass.

    Row k*P + r is row r of super-entry k; y_ptr starts at the first selected entry, and entry c of row r
    of super-entry k lies k*stride_d + r*stride_p + c*stride_q further.
    """
    pid = tl.program_id(0)
    batch = (pid // num_tiles).to(tl.int64)  # scalar
    rows = pid % num_tiles * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)  # [BLOCK_ROWS]
    columns = tl.arange(0, BLOCK_Q)  # [BLOCK_Q]
    valid = (rows < num_rows)[:, None] & (columns < Q)[None, :]  # [BLOCK_ROWS, BLOCK_Q]
    position = (rows // P).to(tl.int64)[:, None] * stride_d + (rows % P)[:, None] * stride_p + columns[None, :] * stride_q  # [BLOCK_ROWS, BLOCK_Q]
    y = tl.load(y_ptr + batch * matrix_size + position, mask=valid, other=0.)  # [BLOCK_ROWS, BLOCK_Q]
    scale = _int_abs_max_scale(tl.max(tl.abs(y), axis=1), RANGE=QMAX - QMIN, DTYPE=DTYPE)[:, None]  # [BLOCK_ROWS, 1]
    code = tl.clamp(libdevice.rint(libdevice.div_rn(y, scale)), QMIN, QMAX)  # [BLOCK_ROWS, BLOCK_Q]
    tl.store(out_ptr + (batch * num_rows + rows[:, None]) * Q + columns[None, :], code * scale, mask=valid)


def int_abs_max_quantizer_triton(
    y_flat: torch.Tensor,
    select: Callable[[torch.Tensor], torch.Tensor],
    *,
    out: torch.Tensor,
    qmin: float,
    qmax: float,
    dtype: torch.dtype,
) -> None:
    """The same as ``int_abs_max_quantizer_torch`` for a 16-bit ``dtype`` with Triton, for a ``select`` that
    returns views: one kernel reads each row through the view's strides and writes its rounded values."""
    x = select(y_flat)  # [*batch, d, p, q]
    assert y_flat.is_cuda, "the Triton quantizer requires CUDA tensors"
    assert y_flat.is_contiguous() and out.is_contiguous(), "the Triton quantizer requires contiguous Y and out"
    assert x.untyped_storage().data_ptr() == y_flat.untyped_storage().data_ptr(), "select must return views"
    assert out.shape == x.shape and out.dtype == x.dtype, "out must match the selected super-entries"
    assert qmin < qmax and dtype in _SCALE_DTYPES, "expected qmin < qmax and a scale in fp16 or bf16"
    d, p, q = x.shape[-3:]
    num_tiles: int = triton.cdiv(d * p, 4)  # few rows per program: the kernel is latency-bound
    with torch.cuda.device(y_flat.device):
        _int_abs_max_quantize_kernel[(y_flat.shape[:-1].numel() * num_tiles,)](
            out, x,
            matrix_size=y_flat.size(-1), num_rows=d * p, num_tiles=num_tiles,
            stride_d=x.stride(-3), stride_p=x.stride(-2), stride_q=x.stride(-1),
            P=p, Q=q, QMIN=qmin, QMAX=qmax, DTYPE=_SCALE_DTYPES[dtype],
            BLOCK_ROWS=4, BLOCK_Q=triton.next_power_of_2(q),
            num_warps=4,
            enable_reflect_ftz=False,
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class _TestQuantizers(unittest.TestCase):
    def test_affine_quantizer(self):
        y = torch.zeros(3, 3, dtype=torch.float64, device="cuda")
        scale, zero_point = torch.ones_like(y), torch.zeros_like(y)
        y[0, 2], y[1, 1], y[2, 0] = .5, 1.75, -9.
        scale[1, 1], scale[2, 0], zero_point[2, 0] = .5, 2., 1.
        antidiagonal_slice = slice(2, 8, 2)
        for affine_quantizer in (affine_quantizer_torch, affine_quantizer_triton):
            for quantizer, expected in (
                (functools.partial(affine_quantizer), y.new_tensor([0., 2., -10.])),
                (functools.partial(affine_quantizer, qmin=-2, qmax=2), y.new_tensor([0., 1., -6.])),
            ):
                actual = torch.empty_like(expected)
                quantizer(y.flatten(), antidiagonal_slice, scale, zero_point, out=actual)
                self.assertTrue(actual.equal(expected))

    def test_affine_quantizer_broadcasting(self):
        """The Triton quantizer broadcasts batched scales and zero points like the PyTorch one."""
        generator = torch.Generator(device="cuda").manual_seed(0)
        y_flat = 4. * torch.randn(2, 3, 5 * 7, generator=generator, dtype=torch.float64, device="cuda")
        scale = .1 + torch.rand(1, 5, 7, generator=generator, dtype=torch.float64, device="cuda")
        zero_point = torch.randint(-2, 3, (2, 1, 5, 7), generator=generator, device="cuda").double()
        expected = torch.empty(2, 3, 5, dtype=torch.float64, device="cuda")
        actual = torch.empty_like(expected)
        antidiagonal_slice = slice(6, 36, 6)  # i + j = 6 in a 5 x 7 matrix
        affine_quantizer_torch(y_flat, antidiagonal_slice, scale, zero_point, qmin=-3, qmax=4, out=expected)
        affine_quantizer_triton(y_flat, antidiagonal_slice, scale, zero_point, qmin=-3, qmax=4, out=actual)
        self.assertTrue(actual.equal(expected))

    def test_affine_quantizer_super_entry(self):
        """Super-entries are quantized entry by entry with their own broadcast scales and zero points, the
        same in PyTorch and Triton."""
        generator = torch.Generator(device="cuda").manual_seed(0)
        y_flat = 4. * torch.randn(2, 3, 4 * 6, generator=generator, dtype=torch.float64, device="cuda")
        scale = .1 + torch.rand(1, 4, 6, generator=generator, dtype=torch.float64, device="cuda")
        zero_point = torch.randint(-2, 3, (2, 1, 4, 6), generator=generator, device="cuda").double()
        select = lambda x: x[..., 3:21].unfold(-1, 9, 9).unfold(-1, 3, 6)  # 2 x 3 super-entries (0, 1) and (1, 0) of a 4 x 6 matrix
        expected = torch.empty_like(y_flat)
        affine_quantizer_torch(y_flat, slice(0, 4 * 6, 1), scale, zero_point, qmin=-3, qmax=4, out=expected)
        for affine_quantizer in affine_quantizer_super_entry_torch, affine_quantizer_super_entry_triton:
            actual = torch.empty(2, 3, 2, 2, 3, dtype=torch.float64, device="cuda")
            affine_quantizer(y_flat, select, scale, zero_point, qmin=-3, qmax=4, out=actual)
            self.assertTrue(actual.equal(select(expected)))

    def test_int_abs_max_scale(self):
        """The two scales agree with the CPU's for super-entry rows read through views, a zero row and a
        denormal scale included, in both 16-bit dtypes."""
        generator = torch.Generator(device="cuda").manual_seed(3)
        for x_dtype in torch.float32, torch.float64:
            y_flat = 4. * torch.randn(3, 64 * 96, generator=generator, dtype=x_dtype, device="cuda")
            y_flat[1, 128:160] = 0.
            y_flat[2, 192:224] *= 1e-38  # a scale so small that it is denormal
            x = y_flat[..., 64:224].unfold(-1, 32, 64).unflatten(-1, (1, 32))  # [3, 3, 1, 32]: the 1 x 32 super-entries (i, j) with i + j = 2
            for x in x, x[0, ...]:  # leading dimensions copied, or merged into a view
                for dtype in torch.float16, torch.bfloat16:
                    expected = int_abs_max_scale_torch(x.cpu(), qmin=-8, qmax=7, dtype=dtype)
                    for scale in int_abs_max_scale_torch(x, qmin=-8, qmax=7, dtype=dtype), int_abs_max_scale_triton(x, qmin=-8, qmax=7, dtype=dtype):
                        self.assertTrue(scale.cpu().equal(expected))
        scale = int_abs_max_scale_torch(torch.tensor([[0., 0.], [3., -1.5]]), qmin=-8, qmax=7, dtype=torch.float16)
        self.assertTrue(scale.equal(torch.tensor([[1.], [.4]]).half().float()))

    def test_int_abs_max_quantizer(self):
        """Each row of the selected super-entries is rounded to qmin..qmax times its abs-max scale, the same
        in PyTorch and Triton, a zero row and a scale that rounds to zero or is denormal included."""
        generator = torch.Generator(device="cuda").manual_seed(5)
        y_flat = 4. * torch.randn(2, 64 * 96, generator=generator, device="cuda")
        y_flat[1, 128:168] = 0.
        y_flat[0, 192:224] *= 1e-38
        for select in (
            lambda x: x[..., 64:224].unfold(-1, 32, 64).unflatten(-1, (1, 32)),  # [2, 3, 1, 32]: the 1 x 32 super-entries (i, j) with i + j = 2
            lambda x: x[..., 48:552].unfold(-1, 120, 168).unfold(-1, 24, 96),  # [2, 3, 2, 24]: the 2 x 24 super-entries (i, j) with i + j = 2
        ):
            x = select(y_flat)
            for qmin, qmax in (-2, 1), (-4, 3), (-8, 7), (-128, 127):
                for dtype in torch.float16, torch.bfloat16:
                    scale = int_abs_max_scale_torch(x, qmin=qmin, qmax=qmax, dtype=dtype)
                    for quantizer in int_abs_max_quantizer_torch, int_abs_max_quantizer_triton:
                        out = torch.empty_like(x)
                        quantizer(y_flat, select, out=out, qmin=qmin, qmax=qmax, dtype=dtype)
                        self.assertTrue(out.equal((x / scale).round().clamp(qmin, qmax) * scale))


if __name__ == "__main__":
    unittest.main()
