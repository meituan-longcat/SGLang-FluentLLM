#!/usr/bin/env python3
"""Build the warmed B16/S1=4 three-lane 512-group Perfetto timeline."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


TICKS_PER_US = 50.0
ROLES = ("AIV0", "AIV1", "AIC")
MARK_RE = re.compile(r"SFAD512 lane=(\d+) code=(\d+) cycle=(\d+)")
def read_invocations(path):
    lanes = defaultdict(list)
    for line in path.read_text().splitlines():
        match = MARK_RE.search(line)
        if match:
            lane, code, cycle = map(int, match.groups())
            lanes[lane].append((code, cycle))
    result = {}
    for lane, points in lanes.items():
        invocations = []
        current = []
        for point in points:
            if point[0] == 90 and current:
                invocations.append(current)
                current = []
            current.append(point)
        if current:
            invocations.append(current)
        result[lane] = invocations
    return result


def last_complete(invocations):
    result = {}
    for lane, calls in invocations.items():
        complete = [call for call in calls
                    if 90 in dict(call) and 100 in dict(call) and
                    399 in dict(call)]
        if complete:
            result[lane] = complete[-1]
    return result


def interval(name, lane, begin, end, origin, category, detail):
    return {
        "name": name,
        "cat": category,
        "ph": "X",
        "pid": 1,
        "tid": lane,
        "ts": round((begin - origin) / TICKS_PER_US, 3),
        "dur": round((end - begin) / TICKS_PER_US, 3),
        "args": {
            "start_cycle": begin,
            "end_cycle": end,
            "duration_us": round((end - begin) / TICKS_PER_US, 3),
            "detail": detail,
        },
    }


def add_flow(events, flow_id, name, source_lane, source, target_lane,
             target, origin, item):
    if source > target:
        return
    args = {"group_item": item, "dependency": name}
    events.extend((
        {"name": name, "cat": "dependency", "ph": "s", "id": flow_id,
         "pid": 1, "tid": source_lane,
         "ts": round((source - origin) / TICKS_PER_US, 3), "args": args},
        {"name": name, "cat": "dependency", "ph": "f", "bp": "e",
         "id": flow_id, "pid": 1, "tid": target_lane,
         "ts": round((target - origin) / TICKS_PER_US, 3), "args": args},
    ))


def add_gaps(events, lane, begin, end, origin):
    covered = sorted(
        (event["args"]["start_cycle"], event["args"]["end_cycle"])
        for event in events
        if event.get("ph") == "X" and event.get("tid") == lane
    )
    cursor = begin
    for start, finish in covered:
        if cursor < start:
            events.append(interval("Control / dispatch", lane, cursor, start,
                                   origin, "control",
                                   "No measured data stage owns this interval."))
        cursor = max(cursor, finish)
    if cursor < end:
        events.append(interval("Control / drain", lane, cursor, end, origin,
                               "control",
                               "No measured data stage owns this interval."))


def add_stage_excluding_waits(events, name, lane, begin, end, waits, origin,
                              category, detail):
    """Emit one exclusive lane: compute spans and measured wait spans."""
    cursor = begin
    segment = 0
    for wait_begin, wait_end, wait_name, wait_category, wait_detail in waits:
        if cursor < wait_begin:
            events.append(interval(
                f"{name} [{segment}]", lane, cursor, wait_begin, origin,
                category, detail))
            segment += 1
        events.append(interval(wait_name, lane, wait_begin, wait_end, origin,
                               wait_category, wait_detail))
        cursor = max(cursor, wait_end)
    if cursor < end:
        events.append(interval(f"{name} [{segment}]", lane, cursor, end,
                               origin, category, detail))


def item_name(item, groups_per_task):
    slot = item % groups_per_task
    return f"token {item // groups_per_task}, group {slot}"


def chunks_in_group(item, groups_per_task, chunks):
    group = item % groups_per_task
    return min(4, chunks - group * 4)


def soft_codes(item, total_items, trace_layout):
    if trace_layout == "lookahead2":
        return 60000 + item * 2, 60000 + item * 2 + 1
    if item + 1 < total_items:
        return 1000 + (item + 1) * 20 + 4, 1000 + (item + 1) * 20 + 5
    return 8002, 8003


def merge_codes(item, total_items, trace_layout):
    if trace_layout == "lookahead2":
        return 70000 + item * 2, 70000 + item * 2 + 1
    if item + 2 < total_items:
        return 1000 + (item + 2) * 20 + 2, 1000 + (item + 2) * 20 + 3
    if item + 1 < total_items:
        return 8000, 8001
    return 8004, 8005


def mm2_codes(item, total_items, trace_layout):
    if trace_layout == "lookahead2":
        return 80000 + item * 2, 80000 + item * 2 + 1
    if item + 1 < total_items:
        return 2000 + (item + 1) * 20 + 2, 2000 + (item + 1) * 20 + 3
    return 8000, 8001


def build(stage_points, span_points, selected_core, stage_source, chunks,
          configuration=None, trace_layout="stable"):
    if set(stage_points) != {0, 1, 2}:
        raise RuntimeError("selected-core log must contain lanes 0, 1 and 2")
    marks = {ROLES[lane]: dict(points)
             for lane, points in stage_points.items()}
    origin = min(values[90] for values in marks.values())
    events = [{
        "name": "process_name", "ph": "M", "pid": 1, "tid": 0,
        "args": {"name": "SFAD 512-group / max-token core"},
    }]
    for index, role in enumerate(ROLES):
        events.extend((
            {"name": "thread_name", "ph": "M", "pid": 1, "tid": role,
             "args": {"name": f"core{selected_core}/{role}"}},
            {"name": "thread_sort_index", "ph": "M", "pid": 1,
             "tid": role, "args": {"sort_index": index}},
        ))
        lane_marks = marks[role]
        if all(code in lane_marks for code in (91, 92, 93)):
            events.append(interval(
                f"{role}: role initialization", role,
                lane_marks[90], lane_marks[91], origin, "init",
                "TPipe buffers/events initialization before start barrier."))
            if role == "AIC":
                events.extend((
                    interval(
                        "AIC: wait AIV0+AIV1 ready", role,
                        lane_marks[91], lane_marks[92], origin, "wait_start",
                        "Wait until both vector sub-blocks reach the barrier."),
                    interval(
                        "AIC: release AIV0+AIV1", role,
                        lane_marks[92], lane_marks[93], origin, "start_sync",
                        "One cross-core event releases both AIV lanes."),
                ))
            else:
                events.extend((
                    interval(
                        f"{role}: publish ready", role,
                        lane_marks[91], lane_marks[92], origin, "start_sync",
                        "Publish this vector sub-block's barrier arrival."),
                    interval(
                        f"{role}: wait synchronized release", role,
                        lane_marks[92], lane_marks[93], origin, "wait_start",
                        "Wait for the AIC coordinator to release both AIVs."),
                ))
            if lane_marks[93] < lane_marks[100]:
                events.append(interval(
                    f"{role}: barrier exit", role,
                    lane_marks[93], lane_marks[100], origin, "control",
                    "Control path between barrier release and pipeline entry."))
        else:
            events.append(interval(
                f"{role}: role initialization", role,
                lane_marks[90], lane_marks[100], origin, "init",
                "TPipe buffers/events initialization before pipeline entry."))

    task_count = sum(1 for code in marks["AIC"]
                     if 30000 <= code < 31000 and code % 10 == 0)
    total_items = sum(1 for code in marks["AIC"]
                      if 2000 <= code < 8000 and (code - 2000) % 20 == 0)
    if task_count == 0 or total_items % task_count != 0:
        raise RuntimeError("group items cannot be mapped evenly to tasks")
    groups_per_task = total_items // task_count
    direct_output = groups_per_task == 1

    for role in ROLES[:2]:
        lane_marks = marks[role]
        for task in range(task_count):
            base = 30000 + task * 10
            events.extend((
                interval(f"token {task}: TaskInfo", role,
                         lane_marks[base], lane_marks[base + 1], origin,
                         "scalar", "Find batch and causal threshold."),
                interval(f"token {task}: ChunkCount", role,
                         lane_marks[base + 1], lane_marks[base + 2], origin,
                         "scalar", "Read one sparse index per 128-token chunk."),
                interval(f"token {task}: Publish task", role,
                         lane_marks[base + 2], lane_marks[base + 3], origin,
                         "mte3", "Publish active chunk count to AIC."),
            ))
        for item in range(total_items):
            base = 1000 + item * 20
            label = item_name(item, groups_per_task)
            group_chunks = chunks_in_group(item, groups_per_task, chunks)
            gather_name = f"{label}: Gather {group_chunks}x128 KV"
            gather_detail = "Random GM->UB gather and packed-KV GM publish."
            gather_chunks = []
            for chunk in range(group_chunks):
                chunk_begin = base + 6 + chunk * 2
                if chunk_begin + 1 in lane_marks:
                    begin = lane_marks[chunk_begin]
                    end = lane_marks[chunk_begin + 1]
                    publish_code = 90000 + item * 16 + chunk * 2
                    if (publish_code in lane_marks and
                            publish_code + 1 in lane_marks):
                        publish_begin = lane_marks[publish_code]
                        publish_end = lane_marks[publish_code + 1]
                        if not begin <= publish_begin <= publish_end <= end:
                            raise RuntimeError(
                                f"invalid KV-ready interval for {role}/{label}/"
                                f"chunk{chunk}")
                        if begin < publish_begin:
                            gather_chunks.append((
                                begin, publish_begin,
                                f"{label}: Gather/pack chunk {chunk} "
                                "(128 tokens)", "gather",
                                "One AIV half gathers 64 KV rows, packs them "
                                "and publishes row count to GM."))
                        gather_chunks.append((
                            publish_begin, publish_end,
                            f"{label}: publish KV-ready chunk {chunk}",
                            "publish_kv",
                            "CrossCoreSetFlag(SYNC_KV); any packed-slot "
                            "backpressure is measured here."))
                        if publish_end < end:
                            gather_chunks.append((
                                publish_end, end,
                                f"{label}: Gather return chunk {chunk}",
                                "gather_control",
                                "Measured function return after KV-ready "
                                "publication."))
                    else:
                        gather_chunks.append((
                            begin, end,
                            f"{label}: Gather+publish KV-ready chunk {chunk} "
                            "(128 tokens)", "gather",
                            "Legacy measurement boundary: includes gather, "
                            "pack, row count and KV-ready publication."))
            add_stage_excluding_waits(
                events, f"{gather_name} control", role,
                lane_marks[base], lane_marks[base + 1], gather_chunks, origin,
                "gather_control", gather_detail)
            soft_begin, soft_end = soft_codes(item, total_items, trace_layout)
            softmax_name = f"{label}: {group_chunks * 128}-token Softmax"
            softmax_detail = "SoftmaxFlashV2 and publish probability."
            if direct_output:
                softmax_name += " + FP32 normalize"
                softmax_detail = (
                    "SoftmaxFlashV2, normalize probability in FP32 "
                    "and publish it for direct MM2 output.")
            wait_mm1_code = 100000 + item * 2
            mm1_wait = [(
                lane_marks[wait_mm1_code], lane_marks[wait_mm1_code + 1],
                f"{label}: wait MM1 completion", "vector_wait_mm1_result",
                "Cross-core wait until AIC MM1 Fixpipe output is complete.")]
            add_stage_excluding_waits(
                events, softmax_name, role,
                lane_marks[soft_begin], lane_marks[soft_end], mm1_wait, origin,
                "vector", softmax_detail)
            merge_begin, merge_end = merge_codes(item, total_items,
                                                 trace_layout)
            merge_name = (f"{label}: Wait direct MM2 completion" if direct_output
                          else f"{label}: FP32 update O")
            merge_category = (
                "direct_output_completion" if direct_output else "merge")
            merge_detail = (
                "Consume MM2 completion; output was written directly by Fixpipe."
                if direct_output else
                "Online FP32 O/LSE update; final group stores output.")
            wait_mm2_code = 110000 + item * 2
            mm2_wait = [(
                lane_marks[wait_mm2_code], lane_marks[wait_mm2_code + 1],
                f"{label}: wait MM2 completion", "vector_wait_mm2_result",
                "Cross-core wait until AIC MM2 Fixpipe output is complete.")]
            add_stage_excluding_waits(
                events, merge_name, role,
                lane_marks[merge_begin], lane_marks[merge_end], mm2_wait,
                origin, merge_category, merge_detail)
        # Rejected Gather/Softmax-overlap experiments use these optional
        # markers.  Keep them as overlays on the same three logical lanes:
        # Scalar issue and MTE2 flight are intentionally concurrent with the
        # aggregate Gather/Softmax intervals drawn above.
        for item in range(total_items):
            label = item_name(item, groups_per_task)
            issue_begin = 65000 + item * 2
            consume_begin = 66000 + item * 2
            prepare_begin = 67000 + item * 2
            if issue_begin in lane_marks and issue_begin + 1 in lane_marks:
                events.append(interval(
                    f"{label}: issue future random Gather", role,
                    lane_marks[issue_begin], lane_marks[issue_begin + 1],
                    origin, "prefetch_issue",
                    "Scalar resolves sparse/page indices and issues future "
                    "KV DataCopy commands while PIPE_V executes Softmax."))
            if (issue_begin + 1 in lane_marks and
                    consume_begin in lane_marks and
                    lane_marks[issue_begin + 1] <= lane_marks[consume_begin]):
                events.append(interval(
                    f"{label}: prefetched MTE2 in flight", role,
                    lane_marks[issue_begin + 1], lane_marks[consume_begin],
                    origin, "prefetch_mte2",
                    "Future KV copies remain in flight until their Gather "
                    "consumer reaches the MTE2-to-MTE3 dependency."))
            if consume_begin in lane_marks and consume_begin + 1 in lane_marks:
                events.append(interval(
                    f"{label}: consume prefetch wait", role,
                    lane_marks[consume_begin], lane_marks[consume_begin + 1],
                    origin, "prefetch_wait",
                    "Measured MTE2-to-MTE3 wait when the future Gather is "
                    "consumed."))
            if prepare_begin in lane_marks and prepare_begin + 1 in lane_marks:
                events.append(interval(
                    f"{label}: prepare/publish future task", role,
                    lane_marks[prepare_begin], lane_marks[prepare_begin + 1],
                    origin, "prefetch_prepare",
                    "TaskInfo, valid ChunkCount and AIV-to-AIC task "
                    "publication for the future token."))
        add_gaps(events, role, lane_marks[90], lane_marks[399], origin)

    role = "AIC"
    lane_marks = marks[role]
    for task in range(task_count):
        base = 30000 + task * 10
        events.extend((
            interval(f"token {task}: Wait task", role,
                     lane_marks[base], lane_marks[base + 1], origin,
                     "wait", "Wait for both AIV task descriptors."),
            interval(f"token {task}: Load query", role,
                     lane_marks[base + 1], lane_marks[base + 2], origin,
                     "mte2", "Issue Query and QueryRope GM->L1."),
        ))
    for item in range(total_items):
        base = 2000 + item * 20
        label = item_name(item, groups_per_task)
        group_chunks = chunks_in_group(item, groups_per_task, chunks)
        mm1_detail = "Load packed KV and issue FP32-logit MM1 operations."
        kv_waits = []
        for chunk in range(group_chunks):
            wait_code = 40000 + item * 16 + chunk * 2
            kv_waits.append((
                lane_marks[wait_code], lane_marks[wait_code + 1],
                f"{label}: wait_kv chunk {chunk}", "wait_kv",
                "Cross-core wait until both AIV halves publish packed KV."))
        add_stage_excluding_waits(
            events, f"{label}: MM1 load/issue", role,
            lane_marks[base], lane_marks[base + 1], kv_waits, origin,
            "mm1", mm1_detail)
        mm2_begin, mm2_end = mm2_codes(item, total_items, trace_layout)
        wait_prob_code = 50000 + item * 2
        prob_wait = [(
            lane_marks[wait_prob_code], lane_marks[wait_prob_code + 1],
            f"{label}: wait_prob", "wait_prob",
            "Cross-core wait until both AIV halves publish probability.")]
        mm2_name = f"{label}: MM2 + Fix"
        mm2_detail = (
            "Load Value/probability, accumulate chunks and Fixpipe directly "
            "to final FP32 output." if direct_output else
            "Load Value/probability, accumulate chunks and write FP32 partial O.")
        add_stage_excluding_waits(
            events, mm2_name, role, lane_marks[mm2_begin],
            lane_marks[mm2_end], prob_wait, origin, "mm2", mm2_detail)
    add_gaps(events, role, lane_marks[90], lane_marks[399], origin)

    for item in range(total_items):
        aic_mm1_end = marks["AIC"][2000 + item * 20 + 1]
        kv_ready = marks["AIC"][40000 + item * 16 +
                                (chunks_in_group(item, groups_per_task,
                                                 chunks) - 1) * 2 + 1]
        soft_begin, soft_end = soft_codes(item, total_items, trace_layout)
        mm2_begin, mm2_end = mm2_codes(item, total_items, trace_layout)
        merge_begin, merge_end = merge_codes(item, total_items, trace_layout)
        for role_index, role in enumerate(ROLES[:2]):
            gather_end = marks[role][1000 + item * 20 + 1]
            add_flow(events, 1000 + item * 10 + role_index,
                     "packed KV -> MM1", role, gather_end, "AIC",
                     kv_ready, origin, item)
            add_flow(events, 2000 + item * 10 + role_index,
                     "MM1 logits -> Softmax", "AIC", aic_mm1_end, role,
                     marks[role][soft_begin], origin, item)
            add_flow(events, 3000 + item * 10 + role_index,
                     "probability -> MM2", role, marks[role][soft_end], "AIC",
                     marks["AIC"][50000 + item * 2 + 1], origin, item)
            final_dependency = ("direct output -> completion" if direct_output
                                else "MM2 output -> update O")
            add_flow(events, 4000 + item * 10 + role_index,
                     final_dependency, "AIC", marks["AIC"][mm2_end],
                     role, marks[role][merge_end], origin, item)

    spans = []
    selected_only = set(span_points) == {0, 1, 2}
    for lane, points in sorted(span_points.items()):
        values = dict(points)
        spans.append({
            "core": selected_core if selected_only else lane // 3,
            "role": ROLES[lane % 3],
            "entry_cycle": values[90],
            "start_cycle": values[100],
            "end_cycle": values[399],
            "duration_us": round((values[399] - values[90]) /
                                 TICKS_PER_US, 3),
        })
    global_begin = min(row["entry_cycle"] for row in spans)
    global_end = max(row["end_cycle"] for row in spans)
    stage_spans = {
        role: round((marks[role][399] - marks[role][90]) / TICKS_PER_US, 3)
        for role in ROLES
    }
    entry_offsets = {
        role: round((marks[role][90] - origin) / TICKS_PER_US, 3)
        for role in ROLES
    }
    role_init = {
        role: round((marks[role][100] - marks[role][90]) / TICKS_PER_US, 3)
        for role in ROLES
    }
    aic_stage_totals = {
        category: round(sum(
            event["dur"] for event in events
            if event.get("ph") == "X" and event.get("tid") == "AIC" and
            event.get("cat") == category), 3)
        for category in ("wait_kv", "mm1", "wait_prob", "mm2")
    }
    # AIC scalar waits are not exclusive hardware-idle intervals: previously
    # issued MTE1/MM/Fix work can continue while CrossCoreWaitFlag blocks the
    # scalar issue stream.  Use the completion-synchronized enclosing spans
    # for true stage wall time; keep the wait totals only as observations of
    # how long scalar issue was blocked on AIV input readiness.
    aic_stage_wall_totals = {
        "mm1": round(sum(
            (marks["AIC"][2000 + item * 20 + 1] -
             marks["AIC"][2000 + item * 20]) / TICKS_PER_US
            for item in range(total_items)), 3),
        "mm2": round(sum(
            (marks["AIC"][mm2_codes(item, total_items, trace_layout)[1]] -
             marks["AIC"][mm2_codes(item, total_items, trace_layout)[0]]) /
            TICKS_PER_US
            for item in range(total_items)), 3),
    }
    vector_wait_totals = {
        stage: {
            role: round(sum(
                event["dur"] for event in events
                if event.get("ph") == "X" and event.get("tid") == role and
                event.get("cat") == category), 3)
            for role in ROLES[:2]
        }
        for stage, category in (
            ("mm1", "vector_wait_mm1_result"),
            ("mm2", "vector_wait_mm2_result"),
        )
    }
    if direct_output:
        pipeline = (
            "Cube chunk=128; single-group FP32-normalized Softmax; MM2 "
            "Fixpipe writes final output")
    else:
        pipeline = (
            "Cube chunk=128; Vector group=512; FP32 online update O/LSE")
    metadata = {
        "configuration": configuration or (
            f"B16/S1=4, {chunks} chunks, totalQ=64, BF16, "
            "seed17, 24 AIC/48 AIV"),
        "pipeline": pipeline,
        "warmup": "four invocations in one process; only invocation 4 is plotted",
        "selected_core": selected_core,
        "selected_core_selection": (
            "deterministic core with the maximum assigned token count; "
            "never selected by observed latency"),
        "stage_source": stage_source,
        "trace_layout": trace_layout,
        "ticks_per_us": TICKS_PER_US,
        "selected_core_combined_span_us": round(
            (global_end - global_begin) / TICKS_PER_US, 3),
        "selected_core_stage_spans_us": stage_spans,
        "selected_core_entry_offsets_us": entry_offsets,
        "selected_core_role_init_us": role_init,
        "selected_core_aic_stage_totals_us": aic_stage_totals,
        "selected_core_aic_completion_wall_totals_us":
            aic_stage_wall_totals,
        "selected_core_vector_wait_mm2_result_us":
            vector_wait_totals["mm2"],
        "selected_core_vector_wait_result_us": vector_wait_totals,
        "measurement_note": (
            "The plotted pass records only the deterministic max-token core. "
            "Lane entry skew is measured and is not normalized away. AIC MM1 "
            "and MM2 intervals end after trace-only FIX_S synchronization; "
            "AIV MM1/MM2 cross-core waits are measured separately. AIC scalar "
            "wait totals can overlap previously issued MTE1/MM/Fix execution "
            "and must not be added to or subtracted from hardware compute time; "
            "use completion_wall_totals for MM1/MM2 comparison."),
        "selected_core_lane_spans": spans,
    }
    return {"displayTimeUnit": "us", "traceEvents": events,
            "metadata": metadata}, spans, stage_points


def write_summary(path, spans, selected_core):
    lines = ["core\trole\tentry_cycle\tpipeline_cycle\tend_cycle\t"
             "duration_us\tselected_core"]
    lines.extend(
        f"{row['core']}\t{row['role']}\t{row['entry_cycle']}\t"
        f"{row['start_cycle']}\t"
        f"{row['end_cycle']}\t{row['duration_us']:.3f}\t"
        f"{int(row['core'] == selected_core)}"
        for row in spans
    )
    path.write_text("\n".join(lines) + "\n")


def write_timestamps(path, stage_points):
    lines = ["role\tcode\tcycle"]
    for lane in range(3):
        lines.extend(f"{ROLES[lane]}\t{code}\t{cycle}"
                     for code, cycle in stage_points[lane])
    path.write_text("\n".join(lines) + "\n")


def write_vector_wait_mm2(path, trace):
    waits = [
        event for event in trace["traceEvents"]
        if event.get("ph") == "X" and
        event.get("cat") == "vector_wait_mm2_result"
    ]
    lines = ["lane\tstage\tstart_us\tduration_us"]
    lines.extend(
        f"{event['tid']}\t{event['name']}\t{event['ts']:.3f}\t"
        f"{event['dur']:.3f}"
        for event in waits
    )
    for role in ROLES[:2]:
        durations = [event["dur"] for event in waits
                     if event["tid"] == role]
        total = sum(durations)
        average = total / len(durations) if durations else 0.0
        lines.append(
            f"{role}\tTOTAL({len(durations)})\tNA\t{total:.3f}")
        lines.append(
            f"{role}\tAVERAGE\tNA\t{average:.3f}")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage_log", type=Path)
    parser.add_argument("--core", type=int, required=True)
    parser.add_argument("--chunks", type=int, choices=(2, 16), required=True)
    parser.add_argument("--configuration")
    parser.add_argument("--trace-layout", choices=("stable", "lookahead2"),
                        default="stable")
    parser.add_argument("--output", type=Path,
                        default=Path("512_3lane.timeline"))
    args = parser.parse_args()
    stage = last_complete(read_invocations(args.stage_log))
    spans = stage
    trace, span_rows, stage_points = build(
        stage, spans, args.core, args.stage_log.name, args.chunks,
        args.configuration, args.trace_layout)
    args.output.write_text(json.dumps(trace, indent=2) + "\n")
    stem = args.output.with_suffix("")
    write_summary(stem.with_name(stem.name + "_spans.tsv"), span_rows,
                  args.core)
    write_timestamps(stem.with_name(stem.name + "_timestamps.tsv"),
                     stage_points)
    write_vector_wait_mm2(
        stem.with_name(stem.name + "_vector_wait_mm2.tsv"), trace)


if __name__ == "__main__":
    main()
