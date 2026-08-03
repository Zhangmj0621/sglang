"""CPU-only regression tests for stale overlapped prefill results."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

# Must precede sglang imports on machines without a working Triton runtime.
try:
    import torch._inductor.runtime.triton_heuristics  # noqa: F401
except Exception:
    pass

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)


def _req(epoch=0, *, chunked=1):
    return SimpleNamespace(
        retraction_count=epoch,
        is_retracted=False,
        is_chunked=chunked,
        output_ids=[],
        embedding=None,
        return_logprob=False,
        return_hidden_states=False,
        stream=False,
        grammar=None,
        time_stats=SimpleNamespace(
            set_prefill_finished_time=Mock(),
            set_last_chunked_prefill_finish_time=Mock(),
            set_prefill_transfer_queue_entry_time=Mock(),
            set_completion_time=Mock(),
        ),
        finished=Mock(return_value=False),
        check_finished=Mock(),
    )


def _batch(req, epoch):
    return SimpleNamespace(
        reqs=[req],
        launch_retraction_counts=(epoch,),
        return_logprob=False,
        decoding_reqs=None,
        spec_info=None,
        prefill_stats=SimpleNamespace(),
        dp_cooperation_info=None,
    )


def _result(value=123):
    return SimpleNamespace(
        logits_output=None,
        next_token_ids=torch.tensor([value], dtype=torch.int64),
        embeddings=torch.tensor([[float(value)]], dtype=torch.float32),
        extend_input_len_per_req=None,
        extend_logprob_start_len_per_req=None,
        copy_done=None,
        can_run_cuda_graph=False,
    )


def _standard_scheduler(*, generation):
    return SimpleNamespace(
        is_generation=generation,
        stream_output=Mock(),
        report_prefill_stats=Mock(),
        _maybe_update_reasoning_tokens=Mock(),
        tree_cache=SimpleNamespace(cache_unfinished_req=Mock()),
        enable_hisparse=False,
    )


def _pd_scheduler():
    return SimpleNamespace(
        tree_cache=SimpleNamespace(cache_unfinished_req=Mock()),
        disagg_prefill_inflight_queue=[],
        spec_algorithm=SimpleNamespace(is_eagle=lambda: False),
        enable_overlap=True,
        send_kv_chunk=Mock(),
        report_prefill_stats=Mock(),
    )


def _alignment_batch(reqs, epochs):
    size = len(reqs)
    return ScheduleBatch(
        reqs=reqs,
        launch_retraction_counts=epochs,
        model_config=SimpleNamespace(is_encoder_decoder=False),
        device="cpu",
        req_pool_indices=torch.arange(size),
        seq_lens=torch.arange(size),
        seq_lens_cpu=torch.arange(size),
        orig_seq_lens=torch.arange(size),
        seq_lens_sum=sum(range(size)),
        sampling_info=Mock(),
    )


def test_schedule_batch_captures_and_copies_immutable_launch_epoch():
    req = _req(epoch=0)
    batch = ScheduleBatch(reqs=[req])

    assert batch.launch_retraction_counts == (0,)
    req.retraction_count = 1

    copied = batch.copy()
    assert copied.reqs == [req]
    assert copied.launch_retraction_counts == (0,)


def test_schedule_batch_filter_and_merge_keep_launch_epochs_aligned():
    reqs = [_req(epoch=i) for i in range(4)]
    left = _alignment_batch(reqs[:3], (10, 11, 12))
    left.filter_batch(keep_indices=[2, 0])

    assert left.reqs == [reqs[2], reqs[0]]
    assert left.launch_retraction_counts == (12, 10)

    right = _alignment_batch([reqs[3]], (13,))
    left.merge_batch(right)
    assert left.reqs == [reqs[2], reqs[0], reqs[3]]
    assert left.launch_retraction_counts == (12, 10, 13)


def test_standard_generation_skips_old_epoch_after_readmission():
    req = _req(epoch=0, chunked=1)
    old_batch = _batch(req, epoch=0)
    req.retraction_count = 1
    req.is_retracted = False  # a newer admission now owns the mutable Req
    req.is_chunked = 1
    scheduler = _standard_scheduler(generation=True)

    SchedulerOutputProcessorMixin.process_batch_result_prefill(
        scheduler, old_batch, _result()
    )

    assert req.output_ids == []
    assert req.is_chunked == 1
    req.time_stats.set_last_chunked_prefill_finish_time.assert_not_called()
    scheduler.tree_cache.cache_unfinished_req.assert_not_called()
    scheduler.stream_output.assert_called_once_with([], False, None)

    scheduler.stream_output.reset_mock()
    new_batch = _batch(req, epoch=1)
    SchedulerOutputProcessorMixin.process_batch_result_prefill(
        scheduler, new_batch, _result()
    )
    assert req.is_chunked == 0
    req.time_stats.set_last_chunked_prefill_finish_time.assert_called_once_with()
    scheduler.stream_output.assert_called_once_with([req], False, req)


def test_embedding_skips_old_epoch_after_readmission():
    req = _req(epoch=0, chunked=1)
    old_batch = _batch(req, epoch=0)
    req.retraction_count = 1
    req.is_retracted = False
    req.is_chunked = 1
    scheduler = _standard_scheduler(generation=False)

    with patch.object(
        __import__(
            "sglang.srt.managers.scheduler_output_processor_mixin",
            fromlist=["envs"],
        ).envs.SGLANG_EMBEDDINGS_SPARSE_HEAD,
        "is_set",
        return_value=False,
    ):
        SchedulerOutputProcessorMixin.process_batch_result_prefill(
            scheduler, old_batch, _result()
        )

    assert req.embedding is None
    assert req.output_ids == []
    assert req.is_chunked == 1
    scheduler.tree_cache.cache_unfinished_req.assert_not_called()
    scheduler.stream_output.assert_called_once_with([], False, None)


def test_pd_prefill_skips_old_epoch_but_new_epoch_processes_normally():
    req = _req(epoch=0, chunked=1)
    old_batch = _batch(req, epoch=0)
    req.retraction_count = 1
    req.is_retracted = False
    req.is_chunked = 1
    req.tmp_end_idx = 7
    scheduler = _pd_scheduler()

    SchedulerDisaggregationPrefillMixin.process_batch_result_disagg_prefill(
        scheduler, old_batch, _result()
    )

    assert req.output_ids == []
    assert req.is_chunked == 1
    scheduler.tree_cache.cache_unfinished_req.assert_not_called()
    scheduler.send_kv_chunk.assert_not_called()
    assert scheduler.disagg_prefill_inflight_queue == []

    new_batch = _batch(req, epoch=1)
    SchedulerDisaggregationPrefillMixin.process_batch_result_disagg_prefill(
        scheduler, new_batch, _result()
    )
    assert req.is_chunked == 0
    scheduler.send_kv_chunk.assert_called_once_with(
        req, last_chunk=False, end_idx=req.tmp_end_idx
    )
