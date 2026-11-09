"""
Indexer latency predictor and rank load balancer for mlp_lightning_indexer.

Predictor: hybrid model
  - q*kv <= THRESHOLD: lookup table + bilinear interpolation on (q, kv)
  - q*kv >  THRESHOLD: linear fit  latency = alpha * q*kv + beta * q + gamma

Balancer: contiguous token partitioning across ranks to minimize max-rank latency.
"""

import numpy as np

# ============================================================================
# Benchmark data (B=1, from bench_lightning_indexer.py)
# Format: (q_len, kv_len, latency_ms)
# ============================================================================

# --- npu_lightning_indexer ---
NPU_BENCH_DATA = [
    # q=1
    (1, 512, 0.02),
    (1, 1024, 0.02),
    (1, 2048, 0.02),
    (1, 4096, 0.03),
    (1, 8192, 0.03),
    (1, 16384, 0.04),
    (1, 32768, 0.04),
    (1, 65536, 0.04),
    (1, 131072, 0.06),
    (1, 262144, 0.07),
    (1, 524288, 0.11),
    (1, 1048576, 0.23),
    # q=2
    (2, 512, 0.02),
    (2, 1024, 0.02),
    (2, 2048, 0.02),
    (2, 4096, 0.03),
    (2, 8192, 0.03),
    (2, 16384, 0.04),
    (2, 32768, 0.04),
    (2, 65536, 0.05),
    (2, 131072, 0.07),
    (2, 262144, 0.11),
    (2, 524288, 0.20),
    (2, 1048576, 0.31),
    # q=4
    (4, 512, 0.03),
    (4, 1024, 0.03),
    (4, 2048, 0.03),
    (4, 4096, 0.03),
    (4, 8192, 0.04),
    (4, 16384, 0.05),
    (4, 32768, 0.06),
    (4, 65536, 0.09),
    (4, 131072, 0.12),
    (4, 262144, 0.20),
    (4, 524288, 0.36),
    (4, 1048576, 0.67),
    # q=8
    (8, 512, 0.03),
    (8, 1024, 0.04),
    (8, 2048, 0.04),
    (8, 4096, 0.05),
    (8, 8192, 0.06),
    (8, 16384, 0.08),
    (8, 32768, 0.10),
    (8, 65536, 0.14),
    (8, 131072, 0.22),
    (8, 262144, 0.36),
    (8, 524288, 0.66),
    (8, 1048576, 1.24),
    # q=16
    (16, 512, 0.03),
    (16, 1024, 0.04),
    (16, 2048, 0.04),
    (16, 4096, 0.05),
    (16, 8192, 0.07),
    (16, 16384, 0.08),
    (16, 32768, 0.10),
    (16, 65536, 0.15),
    (16, 131072, 0.23),
    (16, 262144, 0.41),
    (16, 524288, 0.76),
    (16, 1048576, 1.46),
    # q=32
    (32, 512, 0.03),
    (32, 1024, 0.04),
    (32, 2048, 0.04),
    (32, 4096, 0.06),
    (32, 8192, 0.07),
    (32, 16384, 0.09),
    (32, 32768, 0.14),
    (32, 65536, 0.23),
    (32, 131072, 0.40),
    (32, 262144, 0.75),
    (32, 524288, 1.45),
    (32, 1048576, 2.86),
    # q=64
    (64, 512, 0.03),
    (64, 1024, 0.04),
    (64, 2048, 0.05),
    (64, 4096, 0.06),
    (64, 8192, 0.09),
    (64, 16384, 0.13),
    (64, 32768, 0.22),
    (64, 65536, 0.39),
    (64, 131072, 0.74),
    (64, 262144, 1.44),
    (64, 524288, 2.85),
    (64, 1048576, 5.66),
    # q=128
    (128, 512, 0.04),
    (128, 1024, 0.05),
    (128, 2048, 0.06),
    (128, 4096, 0.08),
    (128, 8192, 0.12),
    (128, 16384, 0.21),
    (128, 32768, 0.39),
    (128, 65536, 0.74),
    (128, 131072, 1.44),
    (128, 262144, 2.85),
    (128, 524288, 5.66),
    (128, 1048576, 11.28),
    # q=256
    (256, 512, 0.05),
    (256, 1024, 0.06),
    (256, 2048, 0.08),
    (256, 4096, 0.13),
    (256, 8192, 0.22),
    (256, 16384, 0.39),
    (256, 32768, 0.75),
    (256, 65536, 1.45),
    (256, 131072, 2.85),
    (256, 262144, 5.66),
    (256, 524288, 11.28),
    (256, 1048576, 22.52),
    # q=512
    (512, 512, 0.06),
    (512, 1024, 0.09),
    (512, 2048, 0.13),
    (512, 4096, 0.23),
    (512, 8192, 0.40),
    (512, 16384, 0.75),
    (512, 32768, 1.45),
    (512, 65536, 2.86),
    (512, 131072, 5.66),
    (512, 262144, 11.28),
    (512, 524288, 22.52),
    (512, 1048576, 45.02),
    # q=1024
    (1024, 1024, 0.13),
    (1024, 2048, 0.22),
    (1024, 4096, 0.39),
    (1024, 8192, 0.74),
    (1024, 16384, 1.44),
    (1024, 32768, 2.85),
    (1024, 65536, 5.65),
    (1024, 131072, 11.27),
    (1024, 262144, 22.50),
    (1024, 524288, 44.99),
    (1024, 1048576, 89.97),
    # q=2048
    (2048, 2048, 0.39),
    (2048, 4096, 0.68),
    (2048, 8192, 1.37),
    (2048, 16384, 2.77),
    (2048, 32768, 5.57),
    (2048, 65536, 11.18),
    (2048, 131072, 22.41),
    (2048, 262144, 44.89),
    (2048, 524288, 89.86),
    (2048, 1048576, 179.81),
    # q=4096
    (4096, 4096, 1.23),
    (4096, 8192, 2.38),
    (4096, 16384, 5.16),
    (4096, 32768, 10.76),
    (4096, 65536, 21.98),
    (4096, 131072, 44.44),
    (4096, 262144, 89.39),
    (4096, 524288, 179.34),
    (4096, 1048576, 359.23),
    # q=8192
    (8192, 8192, 3.98),
    (8192, 16384, 8.96),
    (8192, 32768, 20.11),
    (8192, 65536, 42.54),
    (8192, 131072, 87.45),
    (8192, 262144, 177.34),
    (8192, 524288, 357.24),
    (8192, 1048576, 717.01),
    # q=16384
    (16384, 16384, 13.69),
    (16384, 32768, 34.72),
    (16384, 65536, 79.48),
    (16384, 131072, 169.26),
    (16384, 262144, 349.03),
    (16384, 524288, 708.81),
    (16384, 1048576, 1428.37),
]

# --- mlp_lightning_indexer ---
MLP_BENCH_DATA = [
    (1, 512, 0.02),
    (1, 1024, 0.03),
    (1, 2048, 0.03),
    (1, 4096, 0.03),
    (1, 8192, 0.03),
    (1, 16384, 0.04),
    (1, 32768, 0.04),
    (1, 65536, 0.05),
    (1, 131072, 0.06),
    (1, 262144, 0.10),
    (1, 524288, 0.16),
    (1, 1048576, 0.30),
    (2, 512, 0.02),
    (2, 1024, 0.03),
    (2, 2048, 0.03),
    (2, 4096, 0.03),
    (2, 8192, 0.03),
    (2, 16384, 0.04),
    (2, 32768, 0.04),
    (2, 65536, 0.05),
    (2, 131072, 0.07),
    (2, 262144, 0.10),
    (2, 524288, 0.17),
    (2, 1048576, 0.33),
    (4, 512, 0.03),
    (4, 1024, 0.03),
    (4, 2048, 0.03),
    (4, 4096, 0.04),
    (4, 8192, 0.04),
    (4, 16384, 0.05),
    (4, 32768, 0.06),
    (4, 65536, 0.09),
    (4, 131072, 0.12),
    (4, 262144, 0.20),
    (4, 524288, 0.35),
    (4, 1048576, 0.67),
    (8, 512, 0.04),
    (8, 1024, 0.04),
    (8, 2048, 0.04),
    (8, 4096, 0.05),
    (8, 8192, 0.06),
    (8, 16384, 0.09),
    (8, 32768, 0.10),
    (8, 65536, 0.14),
    (8, 131072, 0.20),
    (8, 262144, 0.33),
    (8, 524288, 0.58),
    (8, 1048576, 1.11),
    (16, 512, 0.04),
    (16, 1024, 0.04),
    (16, 2048, 0.04),
    (16, 4096, 0.05),
    (16, 8192, 0.07),
    (16, 16384, 0.09),
    (16, 32768, 0.12),
    (16, 65536, 0.18),
    (16, 131072, 0.31),
    (16, 262144, 0.56),
    (16, 524288, 1.07),
    (16, 1048576, 2.10),
    (32, 512, 0.04),
    (32, 1024, 0.04),
    (32, 2048, 0.04),
    (32, 4096, 0.06),
    (32, 8192, 0.08),
    (32, 16384, 0.11),
    (32, 32768, 0.17),
    (32, 65536, 0.30),
    (32, 131072, 0.55),
    (32, 262144, 1.07),
    (32, 524288, 2.08),
    (32, 1048576, 4.13),
    (64, 512, 0.04),
    (64, 1024, 0.04),
    (64, 2048, 0.05),
    (64, 4096, 0.08),
    (64, 8192, 0.11),
    (64, 16384, 0.17),
    (64, 32768, 0.30),
    (64, 65536, 0.55),
    (64, 131072, 1.06),
    (64, 262144, 2.08),
    (64, 524288, 4.13),
    (64, 1048576, 8.18),
    (128, 512, 0.04),
    (128, 1024, 0.05),
    (128, 2048, 0.07),
    (128, 4096, 0.10),
    (128, 8192, 0.16),
    (128, 16384, 0.29),
    (128, 32768, 0.54),
    (128, 65536, 1.05),
    (128, 131072, 2.07),
    (128, 262144, 4.11),
    (128, 524288, 8.18),
    (128, 1048576, 16.30),
    (256, 512, 0.05),
    (256, 1024, 0.07),
    (256, 2048, 0.10),
    (256, 4096, 0.17),
    (256, 8192, 0.30),
    (256, 16384, 0.55),
    (256, 32768, 1.06),
    (256, 65536, 2.08),
    (256, 131072, 4.12),
    (256, 262144, 8.18),
    (256, 524288, 16.29),
    (256, 1048576, 32.63),
    (512, 512, 0.07),
    (512, 1024, 0.11),
    (512, 2048, 0.17),
    (512, 4096, 0.30),
    (512, 8192, 0.55),
    (512, 16384, 1.06),
    (512, 32768, 2.07),
    (512, 65536, 4.11),
    (512, 131072, 8.15),
    (512, 262144, 16.26),
    (512, 524288, 32.58),
    (512, 1048576, 65.15),
    (1024, 1024, 0.15),
    (1024, 2048, 0.28),
    (1024, 4096, 0.53),
    (1024, 8192, 1.04),
    (1024, 16384, 2.05),
    (1024, 32768, 4.11),
    (1024, 65536, 8.15),
    (1024, 131072, 16.28),
    (1024, 262144, 32.50),
    (1024, 524288, 65.08),
    (1024, 1048576, 130.15),
    (2048, 2048, 0.45),
    (2048, 4096, 0.91),
    (2048, 8192, 1.93),
    (2048, 16384, 3.96),
    (2048, 32768, 8.03),
    (2048, 65536, 16.16),
    (2048, 131072, 32.52),
    (2048, 262144, 64.86),
    (2048, 524288, 129.78),
    (2048, 1048576, 259.92),
    (4096, 4096, 1.45),
    (4096, 8192, 3.33),
    (4096, 16384, 7.38),
    (4096, 32768, 15.50),
    (4096, 65536, 31.84),
    (4096, 131072, 64.26),
    (4096, 262144, 129.27),
    (4096, 524288, 259.52),
    (4096, 1048576, 521.67),
    (8192, 8192, 5.04),
    (8192, 16384, 12.71),
    (8192, 32768, 28.95),
    (8192, 65536, 61.51),
    (8192, 131072, 126.53),
    (8192, 262144, 256.03),
    (8192, 524288, 516.32),
    (8192, 1048576, 1042.87),
    (16384, 16384, 18.27),
    (16384, 32768, 49.82),
    (16384, 65536, 114.84),
    (16384, 131072, 245.63),
    (16384, 262144, 504.43),
    (16384, 524288, 1023.03),
    (16384, 1048576, 2064.38),
]


# ============================================================================
# Predictor
# ============================================================================
class IndexerLatencyPredictor:
    """Hybrid model:
    - q*kv <= THRESHOLD: lookup table + bilinear interpolation on (q, kv)
    - q*kv >  THRESHOLD: linear fit  latency = alpha * q*kv + beta * q + gamma
    """

    THRESHOLD = 2048 * 2048  # 4194304

    def __init__(self, bench_data=None, use_mlp=True):
        self.lookup_q_vals = np.array([])
        self.lookup_kv_vals = np.array([])
        self.lookup_grid = np.array([]).reshape(0, 0)
        self.alpha = 0.0
        self.beta = 0.0
        self.gamma = 0.0
        BENCH_DATA = MLP_BENCH_DATA if use_mlp else NPU_BENCH_DATA
        if bench_data is not None:
            self.fit(bench_data)
        else:
            self.fit(BENCH_DATA)

        # ---- Check fit quality on bench data ----
        errs = []
        for q, kv, actual in BENCH_DATA:
            pred = self.predict_single(q, kv, q * kv)
            err = abs(pred - actual) / max(actual, 1e-9) * 100
            errs.append(err)
        errs = np.array(errs)
        print(f"Fit quality on bench data: MAE={errs.mean():.1f}%  P95={np.percentile(errs, 95):.1f}%  Max={errs.max():.1f}%")

    def fit(self, bench_data):
        small_data = [(q, kv, lat) for q, kv, lat in bench_data if q * kv <= self.THRESHOLD]
        large_data = [(q, kv, lat) for q, kv, lat in bench_data if q * kv > self.THRESHOLD]

        if small_data:
            q_vals = sorted(set(q for q, _, _ in small_data))
            kv_vals = sorted(set(kv for _, kv, _ in small_data))
            self.lookup_q_vals = np.array(q_vals, dtype=np.float64)
            self.lookup_kv_vals = np.array(kv_vals, dtype=np.float64)
            grid = np.full((len(q_vals), len(kv_vals)), np.nan)
            q_idx = {q: i for i, q in enumerate(q_vals)}
            kv_idx = {kv: i for i, kv in enumerate(kv_vals)}
            for q, kv, lat in small_data:
                grid[q_idx[q], kv_idx[kv]] = lat
            self.lookup_grid = self._fill_grid(grid)

        if large_data:
            X, y = [], []
            for q, kv, lat in large_data:
                X.append([q * kv, q, 1.0])
                y.append(lat)
            X, y = np.array(X), np.array(y)
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            self.alpha, self.beta, self.gamma = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    @staticmethod
    def _fill_grid(grid):
        for i in range(grid.shape[0]):
            row = grid[i]
            nans = np.isnan(row)
            if nans.any() and not nans.all():
                valid = ~nans
                row[nans] = np.interp(np.where(nans)[0], np.where(valid)[0], row[valid])
                grid[i] = row
        for j in range(grid.shape[1]):
            col = grid[:, j]
            nans = np.isnan(col)
            if nans.any() and not nans.all():
                valid = ~nans
                col[nans] = np.interp(np.where(nans)[0], np.where(valid)[0], col[valid])
                grid[:, j] = col
        grid = np.nan_to_num(grid, nan=np.nanmax(grid))
        return grid

    def _lookup_small(self, q, kv):
        q_clamped = np.clip(q, self.lookup_q_vals[0], self.lookup_q_vals[-1])
        kv_clamped = np.clip(kv, self.lookup_kv_vals[0], self.lookup_kv_vals[-1])
        qi = np.clip(np.searchsorted(self.lookup_q_vals, q_clamped), 1, len(self.lookup_q_vals) - 1)
        q0, q1 = self.lookup_q_vals[qi - 1], self.lookup_q_vals[qi]
        tq = (q_clamped - q0) / max(q1 - q0, 1e-9)
        ki = np.clip(np.searchsorted(self.lookup_kv_vals, kv_clamped), 1, len(self.lookup_kv_vals) - 1)
        kv0, kv1 = self.lookup_kv_vals[ki - 1], self.lookup_kv_vals[ki]
        tkv = (kv_clamped - kv0) / max(kv1 - kv0, 1e-9)
        v00 = self.lookup_grid[qi - 1, ki - 1]
        v01 = self.lookup_grid[qi - 1, ki]
        v10 = self.lookup_grid[qi, ki - 1]
        v11 = self.lookup_grid[qi, ki]
        v0 = v00 * (1 - tkv) + v01 * tkv
        v1 = v10 * (1 - tkv) + v11 * tkv
        return float(v0 * (1 - tq) + v1 * tq)

    def predict_single(self, q_len, kv_len, total_cost):
        if total_cost <= self.THRESHOLD and len(self.lookup_q_vals) > 0:
            return self._lookup_small(q_len, kv_len)
        else:
            return self.alpha * q_len * kv_len + self.beta * q_len + self.gamma

    def predict_batch(self, q_lens, kv_lens):
        total_cost = sum(q_len * kv_len for q_len, kv_len in zip(q_lens, kv_lens))
        return sum(self.predict_single(q, kv, total_cost) for q, kv in zip(q_lens, kv_lens))


# ============================================================================
# IndexerBalancer
# ============================================================================
class IndexerBalancer:
    """Partition tokens into contiguous segments, one per rank, preserving order.

    Cost model (per-token):
      - For each request, predict its total latency via the predictor.
      - If q_len > CHUNK_SIZE: split into chunks of CHUNK_SIZE tokens.
        Each chunk's cost = predict_single(CHUNK_SIZE, kv_len) / CHUNK_SIZE per token.
        The last chunk may be smaller: cost = predict_single(rem, kv_len) / rem per token.
      - If q_len <= CHUNK_SIZE: cost = predict_single(q_len, kv_len) / q_len per token.

    After computing per-token cost for all tokens across all requests,
    we compute prefix sum over tokens and find cut points that split
    the total cost evenly into rank_size segments.
    """

    CHUNK_SIZE = 1600  # segment size for cost calculation

    def __init__(self, rank_size, predictor):
        self.rank_size = rank_size
        self.predictor = predictor

    def _compute_token_costs(self, q_lens, kv_lens):
        """Compute per-token cost for all tokens across all requests.

        Each request (q, kv) is split into chunks of CHUNK_SIZE tokens.
        Chunk i has chunk_q = min(CHUNK_SIZE, remaining_q),
                    chunk_kv = max(kv - C * i, 1).
        chunk_cost = predict_single(chunk_q, chunk_kv) / chunk_q per token.

        Returns:
            token_costs: list[float] - cost per token, length = sum(q_lens)
        """
        token_costs = []
        C = self.CHUNK_SIZE

        total_qkv = sum(q * kv for q, kv in zip(q_lens, kv_lens))
        for _req_idx, (q, kv) in enumerate(zip(q_lens, kv_lens)):
            assert q > 0
            # Split into chunks, compute per-token cost for each chunk
            remaining = q
            chunk_kv = kv - q
            while remaining > 0:
                chunk_q = min(C, remaining)
                chunk_kv = chunk_kv + chunk_q
                chunk_cost = self.predictor.predict_single(chunk_q, chunk_kv, total_qkv)
                per_token = chunk_cost / chunk_q
                token_costs.extend([per_token] * chunk_q)
                remaining -= chunk_q

        return token_costs

    def need_balance(self, q_lens, kv_lens):
        return True
        # sum = 0
        # for q, kv in zip(q_lens, kv_lens):
        #     sum += q * kv
        # return sum > 32 * 1024 * 1024

    def balance(self, q_lens, kv_lens):
        """Return sp_num_tokens: list of length rank_size, each element is the
        number of tokens assigned to that rank.

        Tokens are partitioned contiguously (preserving request order),
        with cut points chosen to equalize predicted latency across ranks.
        """
        n_tokens = sum(q_lens)
        R = self.rank_size

        if n_tokens == 0:
            return [0] * R

        # Per-token cost
        token_costs = self._compute_token_costs(q_lens, kv_lens)

        # Total cost and target per rank
        total_cost = sum(token_costs)
        cost_per_rank = total_cost / R

        # Single pass: scan tokens, record cut point whenever cumulative cost crosses next target
        cut_points = [0]
        cost = 0.0
        next_cost_target = cost_per_rank
        for pos in range(n_tokens):
            cost += token_costs[pos]
            if cost >= next_cost_target and len(cut_points) < R:
                cut_points.append(pos)
                next_cost_target += cost_per_rank
        cut_points.append(n_tokens)

        while len(cut_points) < R + 1:
            cut_points.append(n_tokens)

        # Convert cut_points to sp_num_tokens (per-rank token counts)
        sp_num_tokens = [cut_points[r + 1] - cut_points[r] for r in range(R)]
        return sp_num_tokens


def get_all2all_params_for_index_balance(
    old_splits: list[int],
    new_splits: list[int],
    rank: int,
    element_size: int = 1,
) -> tuple[list[int], list[int]]:
    """Compute input_split_sizes and output_split_sizes for the given *rank*
    for ``torch.distributed.all_to_all_single`` when re-sharding a contiguous
    token tensor from *old_splits* partitioning to *new_splits* partitioning.

    Both *old_splits* and *new_splits* are length-``world_size`` lists whose
    entries sum to the same total number of tokens.  They represent two
    different contiguous partitions of the same global token sequence.

    Parameters
    ----------
    old_splits : list[int]
        ``attn_sp_token_nums`` – current per-rank token counts.
    new_splits : list[int]
        ``ib_attn_sp_token_nums`` – target per-rank token counts.
    rank : int
        The rank to compute split sizes for.
    element_size : int, optional
        Multiplier for each split entry (e.g. ``n_heads * head_dim`` when
        each token carries multiple elements).  Default is 1.

    Returns
    -------
    input_split_sizes : list[int]
        ``input_split_sizes[i]`` = number of elements this rank sends to rank *i*.
    output_split_sizes : list[int]
        ``output_split_sizes[i]`` = number of elements this rank receives from rank *i*.
    """
    R = len(old_splits)

    # Cumulative boundaries: old_cu[r] = sum(old_splits[:r])
    old_cu = [0]
    for x in old_splits:
        old_cu.append(old_cu[-1] + x)
    new_cu = [0]
    for x in new_splits:
        new_cu.append(new_cu[-1] + x)

    old_start, old_end = old_cu[rank], old_cu[rank + 1]
    new_start, new_end = new_cu[rank], new_cu[rank + 1]

    input_split_sizes = []
    output_split_sizes = []
    for i in range(R):
        # overlap between this rank's old range and rank i's new range
        overlap_in = max(0, min(old_end, new_cu[i + 1]) - max(old_start, new_cu[i]))
        input_split_sizes.append(overlap_in * element_size)
        # overlap between rank i's old range and this rank's new range
        overlap_out = max(0, min(old_cu[i + 1], new_end) - max(old_cu[i], new_start))
        output_split_sizes.append(overlap_out * element_size)

    return input_split_sizes, output_split_sizes
