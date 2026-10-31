"""PyTorch reference implementations of the algorithms of the GPTQ-2D paper.

``alg1`` to ``alg7`` follow Algorithms 1-7, with the paper's inputs and letters.
``alg4`` is the buffered D-sided rounding of an n_1 x ... x n_D tensor, and
``alg2_d_sided`` applies Algorithm 2's dense updates to such a tensor.  Indices
and stages count from 0, so entry (i, j) is the paper's (i + 1, j + 1), and
stage s is the paper's stage s + 2 (s + D in D dimensions).  Each algorithm
rounds with a quantizer Q in place of the nearest integer, which receives the
values and their positions.

The super-entry variants round p x q super-entries jointly, as in YAQA: X is a
coarse mbar x nbar matrix of super-entries, mbar = m/p and nbar = n/q, the
feedback factors are unit block triangular with identity diagonal blocks, and
GPTQ-2D runs unchanged on the coarse matrix.  ``yaqa_iterative`` follows the
YAQA_iterative listing of the YAQA paper term by term, as an independent check.
"""

import functools
import unittest

import torch


def Q(x, *coordinates):
    """Default coordinate-independent nearest-integer quantizer."""
    return x.round()


def _unit_diagonal_blocks(L, size):
    """L D^{-1} = (L - D) D^{-1} + I for lower triangular L and D its size x size
    block diagonal: unit block lower triangular, with exactly identity diagonal
    blocks because L - D vanishes there."""
    if size == 1:
        return L / L.diagonal()
    k = L.size(0) // size
    D = torch.block_diag(*L.reshape(k, size, k, size).diagonal(dim1=0, dim2=2).permute(2, 0, 1))
    eye = torch.eye(L.size(0), dtype=L.dtype, device=L.device)
    return torch.linalg.solve_triangular(D, L - D, upper=False, left=False) + eye


def ldl_of_inverse(M, size=1):
    """Unit lower (block) triangular L with M^{-1} = L Lambda L^T, Lambda block
    diagonal with size x size blocks, computed without forming M^{-1}: reverse-order
    Cholesky M = Um Um^T (Um upper triangular) gives M^{-1} = Um^{-T} Um^{-1},
    and L is Um^{-T} normalized by its diagonal blocks."""
    Um = torch.linalg.cholesky(M.flip(dims=(0, 1))).flip(dims=(0, 1))  # upper, M = Um Um^T
    eye = torch.eye(M.size(0), dtype=M.dtype, device=M.device)
    L = torch.linalg.solve_triangular(Um, eye, upper=True).t()  # lower, = Um^{-T}
    return _unit_diagonal_blocks(L, size)


def ldl(M, size=1):
    """Unit lower (block) triangular L with M = L Lambda L^T, Lambda block diagonal
    with size x size blocks: YAQA's factorization of the Gram itself."""
    return _unit_diagonal_blocks(torch.linalg.cholesky(M), size)


def feedback_factors(A, B, super_entry_shape=(1, 1)):
    """The feedback factors L and U, the inputs of Algorithms 2, 3 and 5-7:
    G^{-1} = L Lambda_L L^T and H^{-1} = U^T Lambda_U U for the Grams G = A^T A
    and H = B B^T, with p x p and q x q identity diagonal blocks for p x q
    super-entries."""
    p, q = super_entry_shape
    L = ldl_of_inverse(A.t() @ A, size=p)  # unit lower triangular
    U = ldl_of_inverse(B @ B.t(), size=q).t()  # unit upper triangular
    return L, U


def _antidiagonal(s, m, n, device):
    """Entries (i, j) of stage s, the anti-diagonal i + j = s, 0-indexed (the
    paper's I_{s+2}), ordered by ascending row i."""
    i = torch.arange(max(0, s - n + 1), min(s, m - 1) + 1, device=device)
    return i, s - i


def _hyper_antidiagonal(s, shape, device):
    """Entries i of stage s in D dimensions, i_1 + ... + i_D = s, 0-indexed
    (Algorithm 4's I_{s+D}), as D index tensors in lexicographic order; in 2D,
    those of _antidiagonal."""
    grids = torch.meshgrid(*(torch.arange(n, device=device) for n in shape), indexing="ij")
    return (sum(grids) == s).nonzero(as_tuple=True)


@torch.no_grad()
def alg1(x, F, Q=Q):
    """Algorithm 1: one-dimensional adaptive rounding of x under the unit lower
    triangular feedback F.  Q receives each coordinate's value and index k."""
    N = x.size(0)
    z = torch.empty_like(x)
    e = torch.empty_like(x)
    y = x.clone()
    for k in range(N):
        z[k] = Q(y[k], k)
        e[k] = z[k] - y[k]
        y[k:] += F[k:, k] * e[k]  # F unit lower triangular
    return z


@torch.no_grad()
def alg2(X, L, U, Q=Q):
    """Algorithm 2: direct anti-diagonal dense updates."""
    m, n = X.shape
    Z = torch.empty_like(X)
    E = torch.empty_like(X)
    Y = X.clone()
    for s in range(m + n - 1):
        i, j = _antidiagonal(s, m, n, X.device)
        Z[i, j] = Q(Y[i, j], i, j)
        E[i, j] = Z[i, j] - Y[i, j]
        Y += (L[:, i] * E[i, j]) @ U[j, :]  # fused dense rank-one updates
    return Z


@torch.no_grad()
def alg3(X, L, U, Q=Q):
    """Algorithm 3: GPTQ-2D (buffered anti-diagonal sweep)."""
    m, n = X.shape
    Z = torch.empty_like(X)
    E = torch.empty_like(X)
    Y = X.clone()
    C = torch.zeros_like(X)  # buffer C = LE
    for s in range(m + n - 1):
        i, j = _antidiagonal(s, m, n, X.device)
        Z[i, j] = Q(Y[i, j], i, j)
        E[i, j] = Z[i, j] - Y[i, j]
        M = L[:, i] * E[i, j]  # M_{i:m,j} = L_{i:m,i} E_{i,j}, zero above row i since L[:i, i] = 0
        C[:, j] += M  # fold
        Y[:, j] += M  # push down (the row-i term hits the rounded Y_{i,j}, harmless)
        Y[i, :] += C[i, j, None] * U[j, :]  # push right (U[j, :j] = 0, and the U_{j,j} = 1 term hits the rounded Y_{i,j})
    return Z


@torch.no_grad()
def alg5(X, L, U, w=(), Q=Q):
    """Algorithm 5: blocked GPTQ-2D, with blocks of w = (w_1, ..., w_h) stages at
    levels 1, ..., h, between the single stages of level 0 and the whole sweep of
    level h + 1.  After each stage s, the largest block of level <= h that ends at
    s is flushed into the rest of the next larger block, from its first stage s0
    into the stages s + 1, ..., s1.  For h = 0, each stage flushes into all later
    stages, as in Algorithm 3.  lp is the paper's l'."""
    m, n = X.shape
    h = len(w)
    w = (1, *w, m + n - 1)  # w_0 = 1 and w_{h+1} = m + n - 1
    Z = torch.empty_like(X)
    E = torch.empty_like(X)
    Y = X.clone()
    C = torch.zeros_like(X)
    stage = torch.arange(m, device=X.device)[:, None] + torch.arange(n, device=X.device)  # stage i + j of entry (i, j)
    for s in range(m + n - 1):
        i, j = _antidiagonal(s, m, n, X.device)
        Z[i, j] = Q(Y[i, j], i, j)
        E[i, j] = Z[i, j] - Y[i, j]
        C[i, j] += E[i, j]  # fold own error
        l = max(l for l in range(h + 1) if (s + 1) % w[l] == 0)  # the largest level <= h of a block ending at s
        s0 = max(s // w[lp] * w[lp] for lp in range(l, h + 2))  # the first stage of that block
        s1 = min(-(-(s + 1) // w[lp]) * w[lp] - 1 for lp in range(l + 1, h + 2))  # the last stage of the next larger block
        # the flush, for all targets at once; where() keeps the unset errors of later stages out of the products
        sources = (s0 <= stage) & (stage <= s)
        targets = (s < stage) & (stage <= s1)
        M = L @ E.where(sources, 0.)  # down products, through the sources' errors in each column
        C += M.where(targets, 0.)
        Y += (M + C.where(sources, 0.) @ U).where(targets, 0.)  # and right products, through their buffer values in each row
    return Z


@torch.no_grad()
def alg6(X, L, U, Q=Q):
    """Algorithm 6: nested form (column-major)."""
    m, n = X.shape
    Z = torch.empty_like(X)
    E = torch.empty_like(X)
    Y = X.clone()
    for j in range(n):
        y = Y[:, j].clone()
        for i in range(m):
            Z[i, j] = Q(y[i], i, j)
            E[i, j] = Z[i, j] - y[i]
            y[i:] += L[i:, i] * E[i, j]
        Y += (y - Y[:, j])[:, None] * U[j, :]  # propagate the column correction through U
    return Z


@torch.no_grad()
def alg7(X, L, U, Q=Q):
    """Algorithm 7: implicit-Y form (anti-diagonal).  Y holds each Y_{i,j} only
    from its own stage on, reconstructed from C, and no update reaches it."""
    m, n = X.shape
    Z = torch.empty_like(X)
    E = torch.empty_like(X)
    Y = torch.empty_like(X)
    C = torch.zeros_like(X)  # buffer C = LE
    for s in range(m + n - 1):
        i, j = _antidiagonal(s, m, n, X.device)
        # Y_{i,j} = X_{i,j} + C_{i,1:j} U_{1:j,j}; exact: U[j+1:, j] = 0, so the
        # full-row dot equals the pseudocode's prefix dot
        Y[i, j] = X[i, j] + (C[i, :] * U[:, j].t()).sum(dim=1)
        Z[i, j] = Q(Y[i, j], i, j)
        E[i, j] = Z[i, j] - Y[i, j]
        C[:, j] += L[:, i] * E[i, j]  # fold (exact, as in alg3)
    return Z


@torch.no_grad()
def alg1_super_entry(x, F, Q=Q, super_entry_shape=(1, 1)):
    """Algorithm 1 over p x q super-entries: x holds the super-entries one after
    another, each column-major, F is unit lower block triangular with pq x pq
    identity diagonal blocks, and Q rounds each super-entry jointly as a p x q
    matrix, receiving its index k."""
    p, q = super_entry_shape
    z = torch.empty_like(x)
    e = torch.empty_like(x)
    y = x.clone()
    for k in range(x.size(0) // (p * q)):
        sl = slice(k * p * q, (k + 1) * p * q)  # the coordinates of super-entry k
        z[sl] = Q(y[sl].reshape(q, p).t(), k).t().flatten()
        e[sl] = z[sl] - y[sl]
        y[sl.start:] += F[sl.start:, sl] @ e[sl]  # F unit lower block triangular
    return z


@torch.no_grad()
def alg3_super_entry(X, L, U, Q=Q, super_entry_shape=(1, 1)):
    """Algorithm 3 over p x q super-entries: GPTQ-2D on X as an mbar x nbar matrix
    of super-entries, rounding each jointly with Q(x, i, j), x of shape
    (|I_s|, p, q) and coordinates (|I_s|, 1, 1).  L and U need identity diagonal
    blocks."""
    m, n = X.shape
    p, q = super_entry_shape
    mbar, nbar = m // p, n // q
    Z = torch.empty_like(X)
    E = torch.empty_like(X)
    Y = X.clone()
    C = torch.zeros_like(X)  # buffer C = LE
    Z4, E4, Y4, C4 = (T.reshape(mbar, p, nbar, q) for T in (Z, E, Y, C))  # super-entry (i, j) of T is T4[i, :, j, :]
    for s in range(mbar + nbar - 1):
        i, j = _antidiagonal(s, mbar, nbar, X.device)
        Z4[i, :, j, :] = Q(Y4[i, :, j, :], i[:, None, None], j[:, None, None])  # [|I_s|, p, q]
        E4[i, :, j, :] = Z4[i, :, j, :] - Y4[i, :, j, :]
        M = (L.reshape(m, mbar, 1, p)[:, i, :, :] @ E4[i, :, j, :])[:, :, 0, :]  # [m, |I_s|, q]
        C.reshape(m, nbar, q)[:, j, :] += M  # fold
        Y.reshape(m, nbar, q)[:, j, :] += M  # push down (the band term hits the rounded Y_{i,j}, harmless)
        Y.reshape(mbar, p, n)[i, :, :] += C4[i, :, j, :] @ U.reshape(nbar, q, n)[j, :, :]  # push right
    return Z


@torch.no_grad()
def alg2_d_sided(X, L, Q=Q):
    """Algorithm 2 in D dimensions: direct dense updates stage by stage, with X,
    L, and Q as in alg4.  The error E_i of entry i adds E_i times the outer
    product of the columns L^(1)_{:,i_1}, ..., L^(D)_{:,i_D} to all of Y (zero on
    the entries with a smaller index than i in some mode), and the updates of each
    stage fuse into one sum: 1 + sum_d (n_d - 1) stages, as in alg4, but O(N^2)
    work for N = n_1 ... n_D entries, as for Algorithm 1 on vec(X)."""
    D = X.dim()
    Z = torch.empty_like(X)
    E = torch.empty_like(X)
    Y = X.clone()
    for s in range(sum(X.shape) - D + 1):  # 1 + sum_d (n_d - 1) stages
        i = _hyper_antidiagonal(s, X.shape, X.device)  # the multi-indices (i_1, ..., i_D) of stage s
        Z[*i] = Q(Y[*i], *i)
        E[*i] = Z[*i] - Y[*i]
        update = E[*i]  # each entry's rank-one update, grown one mode at a time to [n_1, ..., n_D, |I_s|]
        for d in range(D):
            update = update[..., None, :] * L[d][:, i[d]]  # outer product with column i[d] of L[d]
        Y += update.sum(dim=-1)  # fused dense rank-one updates
    return Z


@torch.no_grad()
def alg4(X, L, Q=Q):
    """Algorithm 4: buffered D-sided rounding of an n_1 x ... x n_D tensor X.
    L = (L^(1), ..., L^(D)) holds one unit lower triangular factor per mode, so
    L[d] is the paper's L^(d+1) (in 2D, the L and U^T of Algorithms 2-7).  C[0]
    holds the errors E, and C[1], ..., C[D] are the incremental feedback buffers.
    Each stage reconstructs Y_i = X_i + C[1]_i + ... + C[D]_i and rounds it. It
    then folds mode by mode, C[d] through L[d] into C[d + 1] for d = 0, ..., D - 1,
    including each entry's diagonal contribution, so every fold reads final buffer
    values.  After its stage, C[D]_i = Z_i - X_i.  Y holds each Y_i only from its
    own stage on, and no update reaches it.  The sweep has 1 + sum_d (n_d - 1)
    stages and O(N max_d n_d) work for fixed D and N = n_1 ... n_D entries.  Q
    receives the values and all D index tensors."""
    D = X.dim()
    Z = torch.empty_like(X)
    C = [torch.zeros_like(X) for _ in range(D + 1)]  # buffers C[d] = C^(d), C[0] = E
    Y = torch.empty_like(X)
    for s in range(sum(X.shape) - D + 1):  # 1 + sum_d (n_d - 1) stages
        i = _hyper_antidiagonal(s, X.shape, X.device)  # the multi-indices (i_1, ..., i_D) of stage s
        Y[*i] = X[*i] + sum(C[d][*i] for d in range(1, D + 1))  # reconstruction
        Z[*i] = Q(Y[*i], *i)
        C[0][*i] = Z[*i] - Y[*i]  # error
        for d in range(D):  # one mode at a time: fold C[d] along dim d into C[d + 1]
            fiber = torch.arange(X.size(d), device=X.device)[:, None]  # every index along dim d
            C[d + 1][*i[:d], fiber, *i[d + 1:]] += L[d][:, i[d]] * C[d][*i]  # [X.size(d), |I_s|] multiply-adds; L[d][:i, i] = 0 handles i:
    return Z


def yaqa_factors(A, B, super_entry_shape=(1, 1)):
    """Lines 1-2 of YAQA's Algorithm 1 (Tseng et al., 2026, App. A.6) in the
    form its YAQA_iterative listing takes them: Lin = L_I - I and Lout = L_O - I
    for the block LDL of the input and output Grams H_I = B B^T and H_O = A^T A,
    with q x q and p x p diagonal blocks, which therefore vanish."""
    p, q = super_entry_shape
    Lin = ldl(B @ B.t(), size=q) - torch.eye(B.size(0), dtype=B.dtype, device=B.device)
    Lout = ldl(A.t() @ A, size=p) - torch.eye(A.size(1), dtype=A.dtype, device=A.device)
    return Lin, Lout


@torch.no_grad()
def yaqa_iterative(W, Lin, Lout, td_x, td_y, Q=Q):
    """YAQA's rounding, following the YAQA_iterative listing of Tseng et al.
    (2026, App. A.6) term by term, with Q(x, i, j) in place of the codebook:
    td_x x td_y super-entries are rounded one anti-diagonal at a time from the
    bottom right, each target adding the feedback of the rounded super-entries
    below and to its right.  W is X and the returned hatW is Z."""
    m, n = W.shape
    hatW = torch.zeros_like(W)
    # the listing's `starts`, without its repeated anti-diagonal n // td_y - 1
    for s in reversed(range(m // td_x + n // td_y - 1)):
        ii, jj = _antidiagonal(s, m // td_x, n // td_y, W.device)
        target_idx = list(zip(ii.tolist(), jj.tolist()))
        target = []
        for jm, jn in target_idx:
            rows, cols = slice(jm * td_x, (jm + 1) * td_x), slice(jn * td_y, (jn + 1) * td_y)
            down, right = slice(jm * td_x, m), slice(jn * td_y, n)  # from the super-entry onward
            target.append(W[rows, cols] + (
                Lout[down, rows].t() @ (W[down, right] - hatW[down, right]) @ Lin[right, cols]
                + Lout[down, rows].t() @ (W[down, cols] - hatW[down, cols])
                + (W[rows, right] - hatW[rows, right]) @ Lin[right, cols]
            ))
        qtarget = Q(torch.stack(target), ii[:, None, None], jj[:, None, None])
        for k, (jm, jn) in enumerate(target_idx):
            rows, cols = slice(jm * td_x, (jm + 1) * td_x), slice(jn * td_y, (jn + 1) * td_y)
            hatW[rows, cols] = qtarget[k, :, :]
    return hatW


def _random_instance(shape, seed, device, dtype=torch.float64, cond=10.):
    """A random rounding instance: one nonsingular basis per mode (in 2D, A and B)
    with condition number <= cond (random orthogonal factors, log-uniform singular
    values), and a Gaussian X of the given shape."""
    g = torch.Generator(device=device).manual_seed(seed)

    def basis(k):
        Q1, _ = torch.linalg.qr(torch.randn(k, k, generator=g, dtype=dtype, device=device))
        Q2, _ = torch.linalg.qr(torch.randn(k, k, generator=g, dtype=dtype, device=device))
        s = cond ** torch.rand(k, generator=g, dtype=dtype, device=device)
        return (Q1 * s) @ Q2.t()

    bases = [basis(n) for n in shape]
    X = 4. * torch.randn(shape, generator=g, dtype=dtype, device=device)
    return *bases, X


def _coordinate_round(x, i, j):
    """Round to a clipped coordinate-dependent, scaled-integer codebook."""
    scale = 2. ** ((i + 2 * j) % 3 - 1)
    return (x / scale).round().clamp(-3., 2.) * scale


def _vectorized_alg1(X, A, B, Q=Q, super_entry_shape=None):
    """The reference trajectory for the matrix algorithms: alg1 on x = vec(X), or
    alg1_super_entry for a super_entry_shape, with x ordered super-entry by
    super-entry, column-major over the super-entries and within each.  F comes
    from the LDL decomposition of (H (x) G)^{-1}, G = A^T A and H = B B^T, factored
    directly on the equally ordered mn x mn Kronecker Gram, not assembled as
    U^T (x) L.  Q(x, i, j) receives each (super-)entry's position."""
    m, n = X.shape
    p, q = super_entry_shape or (1, 1)
    mbar = m // p
    rows = torch.arange(m, device=X.device).reshape(1, mbar, 1, p)
    cols = torch.arange(n, device=X.device).reshape(n // q, 1, q, 1)
    perm = (cols * m + rows).flatten()  # positions in the column-major vec(X)
    F = ldl_of_inverse(torch.kron(B @ B.t(), A.t() @ A)[perm, :][:, perm], size=p * q)
    x = X.t().flatten()[perm]
    Qk = lambda y, k: Q(y, k % mbar, k // mbar)  # (super-)entry k is (k mod mbar, k div mbar)
    if super_entry_shape is None:
        z = alg1(x, F, Q=Qk)
    else:
        z = alg1_super_entry(x, F, Q=Qk, super_entry_shape=super_entry_shape)
    Z = torch.empty_like(z)
    Z[perm] = z
    return Z.reshape(n, m).t()


class _TestAlgorithms(unittest.TestCase):
    def test_algorithms(self):
        """Run the exactness checks."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        shapes = (8, 8), (5, 13), (13, 5), (1, 7), (7, 1), (32, 32), (24, 40), (40, 24)
        n_seeds = 5
        # block widths w = (w_1, ..., w_h) for h = 0, ..., 3; in (3, 5) and (2, 3, 8), a width does not divide the next
        blocked = {f"alg5{list(w)}": functools.partial(alg5, w=w) for w in ((), (2,), (2, 4), (3, 5), (2, 3, 8))}
        others = {"alg2": alg2, "alg3": alg3, **blocked, "alg6": alg6, "alg7": alg7}
        all_ok = True

        print(f"{'shape':>10} | exact matches out of {n_seeds} seeds")
        for m, n in shapes:
            ok = {name: 0 for name in others}
            for seed in range(n_seeds):
                A, B, X = _random_instance((m, n), 1000 * m + 10 * n + seed, device)
                L, U = feedback_factors(A, B)
                Zref = _vectorized_alg1(X, A, B, Q=_coordinate_round)
                for name, fn in others.items():
                    ok[name] += fn(X, L, U, Q=_coordinate_round).equal(Zref)
            line = "  ".join(f"{name} {count}/{n_seeds}" for name, count in ok.items())
            print(f"{str((m, n)):>10} | {line}")
            all_ok &= all(count == n_seeds for count in ok.values())

        assert all_ok, "some algorithm deviated from the Algorithm 1 trajectory!"
        print("\nAll algorithms match Algorithm 1 exactly on these FP64 instances, with coordinate-dependent codebooks.")

    def test_super_entry_algorithms(self):
        """Run the super-entry exactness checks."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cases = (8, 12, (2, 4)), (6, 6, (3, 2)), (4, 64, (1, 32)), (12, 8, (4, 4)), (16, 16, (16, 16)), (13, 5, (1, 1))
        n_seeds = 3
        all_ok = True

        print(f"{'shape':>10} {'entry':>8} | exact matches out of {n_seeds} seeds")
        for m, n, super_entry_shape in cases:
            p, q = super_entry_shape
            mbar, nbar = m // p, n // q
            ok = {"alg3": 0, "yaqa": 0}
            for seed in range(n_seeds):
                A, B, X = _random_instance((m, n), 1000 * m + 10 * n + seed, device)
                L, U = feedback_factors(A, B, super_entry_shape)
                Zref = _vectorized_alg1(X, A, B, Q=_coordinate_round, super_entry_shape=super_entry_shape)
                ok["alg3"] += alg3_super_entry(X, L, U, Q=_coordinate_round, super_entry_shape=super_entry_shape).equal(Zref)
                # YAQA is our forward sweep on the 180-degree rotation, where each
                # rotated super-entry is rounded at its original coordinates
                Lr, Ur = feedback_factors(A.flip(dims=(1,)), B.flip(dims=(0,)), super_entry_shape=super_entry_shape)
                Qr = lambda x, i, j: _coordinate_round(x, mbar - 1 - i, nbar - 1 - j)
                Zr = alg3_super_entry(X.flip(dims=(0, 1)), Lr, Ur, Q=Qr, super_entry_shape=super_entry_shape).flip(dims=(0, 1))
                Lin, Lout = yaqa_factors(A, B, super_entry_shape=super_entry_shape)
                ok["yaqa"] += yaqa_iterative(X, Lin, Lout, p, q, Q=_coordinate_round).equal(Zr)
            line = "  ".join(f"{name} {count}/{n_seeds}" for name, count in ok.items())
            print(f"{str((m, n)):>10} {str(super_entry_shape):>8} | {line}")
            all_ok &= all(count == n_seeds for count in ok.values())

        # 1 x 1 super-entries are ordinary entries
        A, B, X = _random_instance((9, 13), 0, device)
        L, U = feedback_factors(A, B)
        Zref = _vectorized_alg1(X, A, B, Q=_coordinate_round)
        all_ok &= _vectorized_alg1(X, A, B, Q=_coordinate_round, super_entry_shape=(1, 1)).equal(Zref)
        all_ok &= alg3_super_entry(X, L, U, Q=_coordinate_round).equal(Zref)

        assert all_ok, "some super-entry algorithm deviated from its reference trajectory!"
        print("\nSuper-entry Algorithm 3 matches super-entry Algorithm 1 exactly on these FP64 instances, and YAQA_iterative matches its rotation.")

    def test_d_sided_algorithms(self):
        """Run the D-sided exactness checks."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        shapes = (1,), (7,), (4, 6), (1, 1, 1), (1, 4, 3), (3, 4, 5), (3, 1, 1, 2), (2, 3, 2, 3)
        n_seeds = 3
        others = {"alg2_d_sided": alg2_d_sided, "alg4": alg4}
        all_ok = True

        print(f"{'shape':>12} | exact matches out of {n_seeds} seeds")
        for shape in shapes:
            D = len(shape)
            ok = {name: 0 for name in others}
            for seed in range(n_seeds):
                *As, X = _random_instance(shape, seed, device)
                Gs = [A.t() @ A for A in As]  # G^(d) = A^(d)^T A^(d)
                # Algorithm 1 on the column-major vec(X), with the feedback factored
                # directly on the N x N Kronecker Gram G^(D) (x) ... (x) G^(1)
                F = ldl_of_inverse(functools.reduce(torch.kron, reversed(Gs)))
                z = alg1(X.permute(*reversed(range(D))).flatten(), F)
                Zref = z.reshape(shape[::-1]).permute(*reversed(range(D)))
                L = [ldl_of_inverse(G) for G in Gs]
                for name, fn in others.items():
                    ok[name] += fn(X, L).equal(Zref)
            line = "  ".join(f"{name} {count}/{n_seeds}" for name, count in ok.items())
            print(f"{str(shape):>12} | {line}")
            all_ok &= all(count == n_seeds for count in ok.values())

        assert all_ok, "some D-sided algorithm deviated from the Algorithm 1 trajectory!"
        print("\nDirect and buffered D-sided rounding match the one-dimensional reference on all test instances.")


if __name__ == "__main__":
    unittest.main()
