import json
import os
import sys

from specgen.evaluation import ncu
from specgen.evaluation.jobs import EvalJob
from specgen.evaluation.ncu import build_profile, parse_details_rules, parse_raw_launches, profile_command, short_kernel_name
from specgen.generation.ncu_format import format_ncu_profile, kernel_lines, ncu_summary_cell

GEMM = "void gemm_kernel<128, 128>(const float *, float *, int)"
COLUMNS = ["gpu__time_duration.sum", "sm__throughput.avg.pct_of_peak_sustained_elapsed",
           "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
           "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
           "dram__throughput.avg.pct_of_peak_sustained_elapsed", "dram__bytes.sum.per_second",
           "sm__warps_active.avg.pct_of_peak_sustained_active", "sm__maximum_warps_per_active_cycle_pct",
           "launch__occupancy_limit_registers", "launch__occupancy_limit_shared_mem", "launch__occupancy_limit_warps",
           "launch__registers_per_thread", "launch__shared_mem_per_block_dynamic", "launch__block_size",
           "launch__grid_size", "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
           "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"]
UNITS = ["nsecond", "%", "%", "%", "%", "byte/second", "%", "%", "block", "block", "block", "register/thread",
         "byte", "", "", "%", "%"]
CONVERT_A = ["18000", "35", "55", "0", "54.9", "4.5e12", "52", "100", "8", "32", "4", "32", "0", "256", "8192", "1.0", "30.0"]
GEMM_ROW = ["385000", "62", "15", "70", "4.1", "1.25e11", "12", "12.5", "1", "1", "4", "157", "65536", "256", "1024",
            "9.0", "7.7"]
CONVERT_B = ["17000", "35", "55", "0", "55.1", "4.6e12", "52", "100", "8", "32", "4", "32", "0", "256", "8192", "1.2", "31.0"]


def csv_line(cells):
    return ",".join(f'"{c}"' for c in cells)


# the raw page: header, then the units, then one row per profiled launch (two kernels here)
RAW = "\n".join([csv_line(["ID", "Kernel Name", *COLUMNS]), csv_line(["", "", *UNITS]),
                 csv_line(["0", "convert(float *)", *CONVERT_A]), csv_line(["1", GEMM, *GEMM_ROW]),
                 csv_line(["2", "convert(float *)", *CONVERT_B])])
RAW_NOTHING = "\n".join([csv_line(["ID", "Kernel Name", *COLUMNS]), csv_line(["", "", *UNITS])])
DETAILS_HEADER = ["ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Context", "Stream", "Block Size",
                  "Grid Size", "Device", "CC", "Section Name", "Metric Name", "Metric Unit", "Metric Value", "Rule Name",
                  "Rule Type", "Rule Description", "Estimated Speedup Type", "Estimated Speedup"]


def rule_row(launch, kernel, section, rule, kind, text, stype="", speedup=""):
    return csv_line([launch, "1", "python3", "host", kernel, "1", "7", "(256, 1, 1)", "(64, 1, 1)", "0", "10.3",
                     section, "", "", "", rule, kind, text, stype, speedup])


DETAILS = "\n".join([
    csv_line(DETAILS_HEADER),
    rule_row("1", GEMM, "SpeedOfLight", "SOLBottleneck", "OPT", "Compute and memory are low."),
    rule_row("1", GEMM, "Occupancy", "TheoreticalOccupancy", "OPT",
             "Theoretical occupancy (12.5%) is limited by registers. See the Kernel Profiling Guide.", "global", "43.9"),
    rule_row("1", GEMM, "SchedulerStats", "IssueSlotUtilization", "OPT", "Issues every 4.6 cycles.", "local", "43.9"),
    rule_row("1", GEMM, "WarpStateStats", "CPIStall", "INF", "Check the Warp Stall Sampling table."),
    rule_row("0", "convert(float *)", "MemoryWorkloadAnalysis_Tables", "MemoryCacheAccessPattern", "OPT",
             "Loads are not coalesced.", "global", "5.0"),
])


def test_raw_page_is_read_per_launch_and_grouped_by_kernel():
    launches, units = parse_raw_launches(RAW)
    assert [launch["id"] for launch in launches] == ["0", "1", "2"]  # the units row is not a launch
    assert units["dram__bytes.sum.per_second"] == "byte/second" and units["gpu__time_duration.sum"] == "nsecond"
    profile = build_profile(launches, units, parse_details_rules(DETAILS))
    gemm, convert = profile["kernels"]
    assert (gemm["name"], gemm["launches"], gemm["share"]) == ("gemm_kernel<128, 128>", 1, 0.9167)
    assert (convert["name"], convert["launches"], convert["share"]) == ("convert", 2, 0.0833)
    # every kernel keeps its own numbers: nothing from the converts leaks into the GEMM
    assert gemm["metrics"]["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"] == 70.0
    assert gemm["metrics"]["dram__throughput.avg.pct_of_peak_sustained_elapsed"] == 4.1
    assert convert["metrics"]["dram__throughput.avg.pct_of_peak_sustained_elapsed"] == 55.0  # its two launches
    assert [r["rule"] for r in gemm["rules"]] == ["SOLBottleneck", "TheoreticalOccupancy", "IssueSlotUtilization", "CPIStall"]
    assert [r["rule"] for r in convert["rules"]] == ["MemoryCacheAccessPattern"]
    assert profile["time_ns"] == 420000.0
    assert parse_raw_launches("not a csv") == ([], {}) and parse_details_rules("") == {}
    assert short_kernel_name("void k<(int)4, std::array<char *, 2>>(int, T2)") == "k<(int)4, std::array<char *, 2>>"


def test_profile_command_selects_one_forward():
    common = dict(sections=["SpeedOfLight"], metrics=["gpu__time_duration.sum"], forward=3, launch_cap=64,
                  kernel_path="k.py", task_path="t.py", seed=0, iterations=5)
    cmd = profile_command("ncu", "r.ncu-rep", nvtx=True, **common)
    assert cmd[cmd.index("--nvtx-include") + 1] == "forward3/" and "--nvtx" in cmd and "--launch-skip" not in " ".join(cmd)
    assert "--launch-count=64" in cmd and cmd[-4:] == ["k.py", "t.py", "5", "0"]
    fallback = profile_command("ncu", "r.ncu-rep", nvtx=False, **common)
    assert fallback[fallback.index("--kernel-id") + 1] == "::regex:.*:3" and "mangled" in fallback


def test_ncu_profile_in_the_prompt():
    profile = build_profile(*parse_raw_launches(RAW), parse_details_rules(DETAILS))
    lines = format_ncu_profile(profile, 7000).splitlines()
    assert lines[0] == ("Nsight Compute profile of one forward pass (2 kernels; shares of the forward's GPU time; "
                        "ncu runs with locked clocks):")
    assert lines[1] == "- `gemm_kernel<128, 128>` (92%):"
    assert lines[2] == "  - throughput (% of peak): SM 62%, memory 15%, tensor pipe 70%"
    assert lines[3] == "  - memory (% of peak): DRAM 4.1%, DRAM bandwidth 125 GB/s"
    assert lines[4] == ("  - occupancy: achieved 12%, theoretical 12%, limited by registers and shared memory, "
                        "157 registers/thread, 64.0 KB shared memory/block, 256 threads/block, 1024 blocks")
    assert lines[5] == "  - warp stalls (% of active warp cycles): barrier 9.0, long scoreboard 7.7"
    assert lines[6:10] == ["  - ncu analysis:", "    - [OPT] SOLBottleneck: Compute and memory are low.",
                           "    - [OPT, est. speedup 43.9%] TheoreticalOccupancy: Theoretical occupancy (12.5%) is "
                           "limited by registers.",
                           "    - [OPT, est. local speedup 43.9%] IssueSlotUtilization: Issues every 4.6 cycles."]
    assert lines[10] == "- `convert` (8%):" and "DRAM bandwidth 4.55 TB/s" in lines[12]
    assert "ms" not in "\n".join(lines) and " us" not in "\n".join(lines)  # no ncu-timed durations
    single = build_profile(*parse_raw_launches(RAW.replace('"convert(float *)"', f'"{GEMM}"')), {})
    assert format_ncu_profile(single, 7000).splitlines()[:2] == [
        "Nsight Compute profile of one forward pass (1 kernel; ncu runs with locked clocks):",
        "- `gemm_kernel<128, 128>`:"]
    assert len(format_ncu_profile(profile, 300)) <= 300
    cell = ncu_summary_cell(profile)
    assert cell == ("2 kernels, the main one 92%, SM 62%, tensor 70%, DRAM 4.1%, occupancy 12% "
                    "(limit: registers, shared memory), 157 regs; advice: TheoreticalOccupancy +43.9%, "
                    "IssueSlotUtilization +43.9%")
    assert ncu_summary_cell(None) == "-"
    tcgen05 = {"sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": 95.6,
               "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active": 95.8}
    assert kernel_lines({"metrics": tcgen05}) == ["throughput (% of peak): tensor pipe 96% (tcgen05)"]


FAKE_NCU = """#!{python}
import json, pathlib, sys, time
argv = sys.argv[1:]
pathlib.Path({log!r}).open("a").write(json.dumps(argv) + "\\n")
if "--import" in argv:
    mode = pathlib.Path(argv[argv.index("--import") + 1]).read_text()
    page = argv[argv.index("--page") + 1]
    print({raw!r} if page == "raw" and mode != "nothing" else {nothing!r} if page == "raw" else {details!r})
    sys.exit(0)
if {slow} and argv.count("--section") > 3:
    time.sleep(30)
mode = "nothing" if {nvtx_captures_nothing} and "--nvtx" in argv else "launches"
pathlib.Path(argv[argv.index("-o") + 1]).write_text(mode)
"""


def fake_ncu(tmp_path, *, slow=False, nvtx_captures_nothing=False):
    path = tmp_path / "ncu"
    path.write_text(FAKE_NCU.format(python=sys.executable, log=str(tmp_path / "calls.jsonl"), raw=RAW,
                                    nothing=RAW_NOTHING, details=DETAILS, slow=slow,
                                    nvtx_captures_nothing=nvtx_captures_nothing))
    os.chmod(path, 0o755)
    return path


def ncu_job(tmp_path, binary, **params):
    return EvalJob(kind="ncu", task_kind="kernelbench", task_path="t.py", kernel_path="k.py", work_dir=str(tmp_path),
                   params={"ncu_binary": str(binary), "metrics": ["gpu__time_duration.sum"],
                           "sections": ["SpeedOfLight", "Occupancy", "LaunchStats", "WarpStateStats", "SourceCounters"],
                           **params})


def calls(tmp_path):
    return [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]


def test_run_profiles_one_forward_and_keeps_kernels_apart(tmp_path):
    result = ncu.run(ncu_job(tmp_path, fake_ncu(tmp_path)))
    assert result["ok"] and result["status"] == "success" and result["selection"] == "nvtx" and not result["degraded"]
    assert result["kernels"] == 2 and result["rules"] == 5
    assert [k["name"] for k in result["profile"]["kernels"]] == ["gemm_kernel<128, 128>", "convert"]
    assert result["metrics"]["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"] == 70.0  # the GEMM's
    profile_call, raw_call = calls(tmp_path)[0], calls(tmp_path)[1]
    assert "--nvtx" in profile_call and profile_call.count("--section") == 5
    assert raw_call[raw_call.index("--print-units") + 1] == "base"


def test_without_nvtx_ranges_each_kernel_is_profiled_once(tmp_path):
    result = ncu.run(ncu_job(tmp_path, fake_ncu(tmp_path, nvtx_captures_nothing=True)))
    assert result["ok"] and result["selection"] == "kernel-id" and result["kernels"] == 2
    profiles = [c for c in calls(tmp_path) if "--import" not in c]
    assert "--nvtx" in profiles[0] and "--kernel-id" in profiles[1]


def test_a_slow_profile_is_killed_and_repeated_with_the_minimal_sections(tmp_path):
    result = ncu.run(ncu_job(tmp_path, fake_ncu(tmp_path, slow=True), timeout_s=2))
    assert result["ok"] and result["degraded"]
    assert [a["timed_out"] for a in result["attempts"]] == [True, False]
    assert result["attempts"][0]["seconds"] < 10  # killed, not waited for
    minimal = [c for c in calls(tmp_path) if "--import" not in c][1]
    assert [minimal[i + 1] for i, a in enumerate(minimal) if a == "--section"] == list(ncu.MINIMAL_SECTIONS)


def test_missing_counter_permission_is_reported(tmp_path):
    fake = tmp_path / "ncu"
    fake.write_text("#!/bin/sh\necho '==ERROR== ERR_NVGPUCTRPERM - The user does not have permission'\nexit 1\n")
    os.chmod(fake, 0o755)
    result = ncu.run(ncu_job(tmp_path, fake))
    assert result["status"] == "ncu_unavailable" and not result["ok"]
