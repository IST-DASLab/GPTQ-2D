import functools
import unittest
import warnings

import torch

from gptq_2d import (
    capture_gptq_2d,
    direct_2d,
    direct_2d_super_entry,
    gptq_2d,
    gptq_2d_torch,
    gptq_2d_super_entry,
    gptq_2d_super_entry_torch,
)
from gptq_2d_reference import alg3_super_entry
from quantizer_trellis import trellis_quantizer_super_entry_torch, trellis_quantizer_super_entry_triton

# Small blocks make the Triton versions flush at both levels even on small instances.
_IMPLEMENTATIONS = {
    "direct": direct_2d,
    "torch": gptq_2d_torch,
    "triton": functools.partial(gptq_2d, inner_block_size=3, outer_block_size=7),
}
_SUPER_ENTRY_IMPLEMENTATIONS = {
    "direct": direct_2d_super_entry,
    "torch": gptq_2d_super_entry_torch,
    "triton": functools.partial(gptq_2d_super_entry, inner_block_size=2, outer_block_size=5),
}


def _instance(m, n, seed, super_entry_shape=(1, 1)):
    """X, L, and U, with NaN in the unused triangles and the implicit identity diagonal blocks."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = 4. * torch.randn(m, n, generator=generator, dtype=torch.float64, device="cuda")
    l, u = (.05 * torch.randn(k, k, generator=generator, dtype=torch.float64, device="cuda") for k in (m, n))
    for factor, size, unused in ((l, super_entry_shape[0], torch.ge), (u, super_entry_shape[1], torch.le)):
        block = torch.arange(factor.size(0), device="cuda") // size
        factor[unused(block[None, :], block[:, None])] = torch.nan  # L is read only below its diagonal blocks, U only above
    return x, l, u


def _unit_factors(l, u, super_entry_shape=(1, 1)):
    """The factors that l and u stand for: zero unused triangles and identity diagonal blocks."""
    factors = []
    for factor, size in ((l.tril(), super_entry_shape[0]), (u.triu(), super_entry_shape[1])):
        block = torch.arange(factor.size(-1), device="cuda") // size
        eye = torch.eye(factor.size(-1), dtype=factor.dtype, device="cuda")
        factors.append(torch.where(block[:, None] == block[None, :], eye, factor))
    return factors


def _all_implementations(l, u, super_entry_shape):
    """Yield (name, run, l, u) for every implementation; the plain ones read inside the diagonal blocks."""
    l_unit, u_unit = _unit_factors(l, u, super_entry_shape)
    for name, run in _IMPLEMENTATIONS.items():
        yield name, run, l_unit, u_unit
    for name, run in _SUPER_ENTRY_IMPLEMENTATIONS.items():
        yield f"super-entry {name}", functools.partial(run, super_entry_shape=super_entry_shape), l, u


def _round_to_scale(y_flat, selection, scale, *, out):
    """Round to multiples of scale, through a flat slice or a super-entry selector."""
    select = selection if callable(selection) else lambda t: t[..., selection]
    torch.div(select(y_flat), select(scale), out=out).round_().mul_(select(scale))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class _TestGPTQ2D(unittest.TestCase):
    def test_reference(self):
        """Every implementation matches the reference exactly on these FP64 instances, Triton with one block, one level, and two levels."""
        cases = (
            (1, 1, (1, 1)), (1, 7, (1, 1)), (7, 1, (1, 1)), (13, 5, (1, 1)), (9, 13, (1, 1)), (65, 97, (1, 1)),
            (8, 12, (2, 4)), (6, 6, (3, 2)), (4, 64, (1, 32)), (12, 8, (4, 4)), (16, 16, (16, 16)), (24, 8, (1, 8)),
            (32, 64, (16, 16)), (96, 160, (16, 16)), (90, 75, (3, 5)),
        )
        for m, n, shape in cases:
            x, l, u = _instance(m, n, 100 * m + n, shape)
            expected = alg3_super_entry(x, *_unit_factors(l, u, shape), super_entry_shape=shape)
            runs = [(name, run) for name, run, l_run, u_run in _all_implementations(l, u, shape) if shape == (1, 1) or name.startswith("super")]
            for inner, outer in ((0, 0), (1, 0), (2, 8), (3, 11)):  # outer blocks with and without a whole number of blocks
                runs.append((f"super-entry triton {inner}, {outer}", functools.partial(
                    gptq_2d_super_entry, super_entry_shape=shape, inner_block_size=inner, outer_block_size=outer,
                )))
                if shape == (1, 1):
                    runs.append((f"triton {inner}, {outer}", functools.partial(gptq_2d, inner_block_size=inner, outer_block_size=outer)))
            for name, run in runs:
                with self.subTest(shape=(m, n), super_entry_shape=shape, implementation=name):
                    self.assertTrue(run(x, l, u).equal(expected))

    def test_joint_quantizer(self):
        """A trellis rounds each 16 x 16 super-entry as one sequence, also in CUDA graphs with new inputs."""
        table = torch.randn(1 << 8, 2, generator=torch.Generator().manual_seed(0), dtype=torch.float64).cuda()
        options = dict(quantizer=functools.partial(trellis_quantizer_super_entry_triton, table=table, K=2), super_entry_shape=(16, 16))

        def reference(x, l, u, scale):
            def trellis(y, i, j):  # Q(x, i, j) of the reference algorithms
                out = torch.empty_like(y)
                trellis_quantizer_super_entry_torch(y, lambda t: t, scale, out=out, table=table, K=2)
                return out
            return alg3_super_entry(x, *_unit_factors(l, u, (16, 16)), Q=trellis, super_entry_shape=(16, 16))

        x, l, u = _instance(32, 64, 7, (16, 16))
        scale = torch.tensor(4., dtype=torch.float64, device="cuda")
        expected = reference(x, l, u, scale)
        for name, run in _SUPER_ENTRY_IMPLEMENTATIONS.items():
            with self.subTest(implementation=name):
                self.assertTrue(run(x, l, u, quantizer_inputs=(scale,), **options).equal(expected))
        replay = capture_gptq_2d(x, l, u, quantizer_inputs=(scale,), **options)
        self.assertTrue(replay(x, l, u, scale).equal(expected))
        self.assertTrue(replay(x + .375, 1.5 * l, .5 * u, 2 * scale).equal(reference(x + .375, 1.5 * l, .5 * u, 2 * scale)))

    def test_quantizer_sees_quantized_values(self):
        """Live Y holds the quantized values of all earlier antidiagonals."""
        x, l, u = _instance(12, 8, 7, (2, 2))
        for name, run, l_run, u_run in _all_implementations(l, u, (2, 2)):
            earlier = []

            def quantizer(y_flat, selection, *, out):
                select = selection if callable(selection) else lambda t: t[..., selection]
                for earlier_select, earlier_z in earlier:
                    self.assertTrue(earlier_select(y_flat).equal(earlier_z))
                torch.round(select(y_flat), out=out)
                earlier.append((select, out.clone()))

            with self.subTest(implementation=name):
                run(x, l_run, u_run, quantizer=quantizer)

    def test_batches_inplace_and_quantizer_inputs(self):
        """Broadcast batches and quantizer inputs, in and out of place; an elementwise quantizer gives the same
        result with or without super-entries."""
        for m, n, shape in ((12, 8, (4, 2)), (9, 15, (3, 5)), (8, 24, (1, 8))):
            x, l, u = _instance(m, n, 10 * m + n, shape)
            x = torch.stack((x, x + .25))[:, None, :, :]  # [2, 1, m, n]
            l = torch.stack((l, .5 * l, 2. * l))[None, :, :, :]  # [1, 3, m, m]
            scale = x.abs().flatten(start_dim=-2) * .05 + .2  # [2, 1, m*n]
            x_before = x.clone()
            expected = gptq_2d_torch(x, *_unit_factors(l, u, shape), quantizer=_round_to_scale, quantizer_inputs=(scale,))
            self.assertEqual(expected.shape, (2, 3, m, n))
            for name, run, l_run, u_run in _all_implementations(l, u, shape):
                with self.subTest(shape=(m, n), super_entry_shape=shape, implementation=name):
                    self.assertTrue(run(x, l_run, u_run, quantizer=_round_to_scale, quantizer_inputs=(scale,)).equal(expected))
                    target = x.expand(2, 3, m, n).clone()
                    result = run(target, l_run, u_run, quantizer=_round_to_scale, quantizer_inputs=(scale,), inplace=True)
                    self.assertEqual(result.data_ptr(), target.data_ptr())
                    self.assertTrue(result.equal(expected))
            self.assertTrue(x.equal(x_before))

    def test_empty_inputs(self):
        """Empty batches and matrices give empty results without calling the quantizer or warning."""
        calls = []
        quantizer = lambda *args, **kwargs: calls.append(args)
        x, l, u = _instance(6, 10, 0, (3, 5))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for empty, l_empty, u_empty in ((x.new_empty(0, 6, 10), l, u), (x.new_empty(0, 10), l[:0, :0], u), (x.new_empty(6, 0), l, u[:0, :0])):
                for name, run, l_run, u_run in _all_implementations(l_empty, u_empty, (3, 5)):
                    with self.subTest(shape=tuple(empty.shape), implementation=name):
                        self.assertEqual(run(empty, l_run, u_run, quantizer=quantizer).shape, empty.shape)
                for options in ({}, dict(super_entry_shape=(3, 5))):
                    replay = capture_gptq_2d(empty, l_empty, u_empty, quantizer=quantizer, **options)
                    self.assertEqual(replay(empty, l_empty, u_empty).shape, empty.shape)
        self.assertEqual(calls, [])

    def test_large_batch(self):
        """The one-dimensional launch grids do not limit the batch to 65535."""
        x, l, u = _instance(4, 6, 9, (2, 2))
        x = x + torch.arange(70_000, dtype=torch.float64, device="cuda")[:, None, None] / 7  # [70000, 4, 6]
        expected = gptq_2d_super_entry_torch(x, l, u, super_entry_shape=(2, 2))
        self.assertTrue(gptq_2d_super_entry(x, l, u, super_entry_shape=(2, 2), inner_block_size=1).equal(expected))
        l_unit, u_unit = _unit_factors(l, u, (2, 2))  # the plain path reads inside the diagonal blocks
        self.assertTrue(gptq_2d(x, l_unit, u_unit, inner_block_size=2).equal(gptq_2d_torch(x, l_unit, u_unit)))

    def test_cuda_graph(self):
        """Replays match direct runs on new inputs, in and out of place, with and without super-entries."""
        for shape in (None, (3, 4)):
            x, l, u = _instance(12, 20, 4321, shape or (1, 1))
            x = torch.stack((x, x + .25))
            scale = x.abs().flatten(start_dim=-2) * .05 + .2  # [2, m*n]
            options = dict(quantizer=_round_to_scale, inner_block_size=2, outer_block_size=4)
            if shape is not None:
                options["super_entry_shape"] = shape
            run = functools.partial(gptq_2d if shape is None else gptq_2d_super_entry, **options)
            with self.subTest(super_entry_shape=shape):
                replay = capture_gptq_2d(x, l, u, quantizer_inputs=(scale,), **options)
                first = replay(x, l, u, scale)
                self.assertTrue(first.equal(run(x, l, u, quantizer_inputs=(scale,))))
                second = replay(x + .375, 1.5 * l, .5 * u, 1.5 * scale)  # the unused entries stay NaN
                self.assertTrue(second.equal(run(x + .375, 1.5 * l, .5 * u, quantizer_inputs=(1.5 * scale,))))
                self.assertTrue(first.equal(run(x, l, u, quantizer_inputs=(scale,))))  # replays return copies
                replay = capture_gptq_2d(x, l, u, quantizer_inputs=(scale,), inplace=True, **options)
                target = x + .25
                self.assertEqual(replay(target, l, u, scale).data_ptr(), target.data_ptr())
                self.assertTrue(target.equal(run(x + .25, l, u, quantizer_inputs=(scale,))))

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA devices are required")
    def test_noncurrent_cuda_device(self):
        original_device = torch.cuda.current_device()
        with torch.cuda.device((original_device + 1) % torch.cuda.device_count()):
            x, l, u = _instance(7, 9, 6789)
        expected = gptq_2d_torch(x, l, u)
        self.assertTrue(gptq_2d(x, l, u, inner_block_size=3).equal(expected))
        self.assertTrue(capture_gptq_2d(x, l, u, inner_block_size=3)(x, l, u).equal(expected))
        self.assertEqual(torch.cuda.current_device(), original_device)


if __name__ == "__main__":
    unittest.main()
