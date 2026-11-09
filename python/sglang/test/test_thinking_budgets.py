import unittest
from typing import List, Optional, Tuple

import torch

from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

class TestThinkingBudgets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def test_case_1(self):
        think_end_ids = torch.tensor([[110, 119, 120], [110, 119, 120]])
        thinking_budgets = torch.tensor([10, 20])
        seq_lens = torch.tensor([0, 0])
        last_token_ids = torch.full_like(think_end_ids, -1)
        sampling_info = SamplingBatchInfo(
            temperatures = None,
            repetition_penalties=None,
            top_ps=None,
            top_ks=None,
            min_ps=None,
            is_all_greedy=False,
            need_min_p_sampling=False,
            vocab_size=0,
            thinking_budgets=thinking_budgets,
            last_token_ids=last_token_ids,
            think_end_ids=think_end_ids)

        for i in range(1, 30):
            seq_lens += 1
            next_token_ids = torch.randint(100, (2,))
            sampling_info.apply_thinking_budgets(seq_lens, next_token_ids)
            self.assertTrue((sampling_info.last_token_ids[:,-1:].squeeze(-1) == next_token_ids).all())
            if i == 12:
                self.assertTrue((sampling_info.thinking_budgets == torch.tensor([-1, 20])).all())
            if i == 22:
                self.assertTrue((sampling_info.thinking_budgets == torch.tensor([-1, -1])).all())


    # spec 路径: predicts 为 [PB*S] 扁平布局; k = thinking_budgets - seq_lens - 1
    # 是窗口内落在 thinking_budgets 位置的槽位

    def _make_spec_info(self, think_end_ids, thinking_budgets):
        return SamplingBatchInfo(
            temperatures=None,
            repetition_penalties=None,
            top_ps=None,
            top_ks=None,
            min_ps=None,
            is_all_greedy=False,
            need_min_p_sampling=False,
            vocab_size=0,
            thinking_budgets=thinking_budgets,
            think_end_ids=think_end_ids)

    def test_spec_case_1_inject_at_budget_position(self):
        """k 落在 [0, acc] → 把 end token 写进 k 并把 accept_length 截到 k"""
        draft_token_num = 4
        sampling_info = self._make_spec_info(
            torch.tensor([[110]], dtype=torch.int32),
            torch.tensor([10], dtype=torch.int32),
        )
        accepts = torch.tensor([3], dtype=torch.int32)
        predicts = torch.tensor([7, 8, 9, 10], dtype=torch.int32)
        # k = 10 - 7 - 1 = 2
        sampling_info.apply_thinking_budgets_spec(
            torch.tensor([7], dtype=torch.int32), predicts, accepts, draft_token_num
        )
        self.assertEqual(predicts.tolist(), [7, 8, 110, 10])
        self.assertEqual(accepts.tolist(), [2])
        self.assertEqual(sampling_info.thinking_budgets.tolist(), [-1])

    def test_spec_case_2_natural_end_is_honored_and_latched(self):
        """模型在窗口内自己吐了 end token → 不注入不截断, 且 budget 闩锁为 -1"""
        draft_token_num = 4
        sampling_info = self._make_spec_info(
            torch.tensor([[110]], dtype=torch.int32),
            torch.tensor([10], dtype=torch.int32),
        )
        accepts = torch.tensor([3], dtype=torch.int32)
        predicts = torch.tensor([7, 110, 9, 10], dtype=torch.int32)
        sampling_info.apply_thinking_budgets_spec(
            torch.tensor([7], dtype=torch.int32), predicts, accepts, draft_token_num
        )
        self.assertEqual(predicts.tolist(), [7, 110, 9, 10])
        self.assertEqual(accepts.tolist(), [3])
        self.assertEqual(sampling_info.thinking_budgets.tolist(), [-1])

        # 闩锁生效: 后续窗口即使 k 落进 [0, acc] 也不得再注入(原 bug 就是这里二次注入)
        predicts_later = torch.tensor([11, 12, 13, 14], dtype=torch.int32)
        accepts_later = torch.tensor([3], dtype=torch.int32)
        sampling_info.apply_thinking_budgets_spec(
            torch.tensor([9], dtype=torch.int32),
            predicts_later,
            accepts_later,
            draft_token_num,
        )
        self.assertEqual(predicts_later.tolist(), [11, 12, 13, 14])
        self.assertEqual(accepts_later.tolist(), [3])

    def test_spec_case_3_budget_not_reached(self):
        """k > acc → 本步不动, budget 保留等后续窗口"""
        draft_token_num = 4
        sampling_info = self._make_spec_info(
            torch.tensor([[110]], dtype=torch.int32),
            torch.tensor([10], dtype=torch.int32),
        )
        accepts = torch.tensor([2], dtype=torch.int32)
        predicts = torch.tensor([7, 8, 9, 10], dtype=torch.int32)
        # k = 10 - 6 - 1 = 3 > acc = 2
        sampling_info.apply_thinking_budgets_spec(
            torch.tensor([6], dtype=torch.int32), predicts, accepts, draft_token_num
        )
        self.assertEqual(predicts.tolist(), [7, 8, 9, 10])
        self.assertEqual(accepts.tolist(), [2])
        self.assertEqual(sampling_info.thinking_budgets.tolist(), [10])

class TestThinkingBudgetsSpecWindows(unittest.TestCase):
    """多窗口仿真: 同一 SamplingBatchInfo 跨窗口复用(等价 scheduler 侧持久对象),
    每窗口提交 predicts[0..acc] 拼输出流、seq_lens += acc+1, 与真实提交节奏一致。
    不设 CUDA guard, CPU 即可跑。"""

    S = 4  # draft_token_num
    END = 110

    def _simulate(
        self, natural_step: Optional[int], steps: int
    ) -> Tuple[List[int], List[int], SamplingBatchInfo]:
        si = SamplingBatchInfo(
            temperatures=None,
            repetition_penalties=None,
            top_ps=None,
            top_ks=None,
            min_ps=None,
            is_all_greedy=False,
            need_min_p_sampling=False,
            vocab_size=0,
            thinking_budgets=torch.tensor([20], dtype=torch.int32),
            think_end_ids=torch.tensor([[self.END]], dtype=torch.int32),
        )
        # 首 token 绝对位置 = seq_lens+1 = 10; budget=20 → cap 在第 3 个窗口内
        seq_lens = torch.tensor([9], dtype=torch.int32)
        stream: List[int] = []
        commit_lens: List[int] = []
        for step in range(steps):
            predicts = torch.ones(self.S, dtype=torch.int32)
            if step == natural_step:
                predicts[1] = self.END  # 模型在窗口内自收尾
            accepts = torch.tensor([3], dtype=torch.int32)
            si.apply_thinking_budgets_spec(seq_lens, predicts, accepts, self.S)
            acc = int(accepts.item())  # 测试代码允许 host 同步
            stream.extend(predicts[: acc + 1].tolist())
            commit_lens.append(acc + 1)
            seq_lens = seq_lens + acc + 1
        return stream, commit_lens, si

    def test_natural_end_not_ignored(self):
        """模型提前自收尾后, k 后续落进 [0, acc] 也不得再注入第二个 end token

        旧代码在本用例 count==2: step2 的 k=2∈[0,3] 时在答案正文里再插一个。
        """
        stream, commit_lens, si = self._simulate(natural_step=1, steps=5)
        self.assertEqual(stream.count(self.END), 1)
        # 自收尾后各窗口提交量不受截断
        self.assertEqual(commit_lens, [4, 4, 4, 4, 4])
        # 闩锁跨窗口保持
        self.assertEqual(si.thinking_budgets.tolist(), [-1])

    def test_cap_still_enforced_without_natural_end(self):
        """无自收尾时 cap 照常生效: 恰好一个 end token, 绝对位置 == thinking_budgets"""
        stream, commit_lens, si = self._simulate(natural_step=None, steps=5)
        self.assertEqual(stream.count(self.END), 1)
        # stream 下标 i ↔ 绝对位置 10+i
        self.assertEqual(stream.index(self.END) + 10, 20)
        # 注入窗口 accept_length 截到 k=2
        self.assertEqual(commit_lens, [4, 4, 3, 4, 4])
        self.assertEqual(si.thinking_budgets.tolist(), [-1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
