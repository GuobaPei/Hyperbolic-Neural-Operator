import atexit
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


def _is_enabled() -> bool:
    return os.environ.get("HNO_CLAMP_STATS", "0") == "1"


@dataclass
class _TagStats:
    total: int = 0
    clamped: int = 0
    min_arg: float = float("inf")
    eps_min: float = float("inf")
    eps_max: float = float("-inf")


@dataclass
class _ClampStats:
    total: int = 0
    clamped: int = 0
    min_arg: float = float("inf")
    min_margin: float = float("inf")
    samples: list = field(default_factory=list)  # list[torch.Tensor] on CPU
    by_tag: Dict[str, _TagStats] = field(default_factory=dict)


_STATS = _ClampStats()


def _max_per_call() -> int:
    try:
        return max(0, int(os.environ.get("HNO_CLAMP_STATS_MAX_PER_CALL", "1024")))
    except Exception:
        return 1024


def _max_samples() -> int:
    try:
        return max(0, int(os.environ.get("HNO_CLAMP_STATS_MAX_SAMPLES", "200000")))
    except Exception:
        return 200000


def record_acosh_clamp(arg: torch.Tensor, eps: float, tag: str = "default") -> None:
    if not _is_enabled():
        return

    eps_f = float(eps)
    threshold = 1.0 + eps_f
    with torch.no_grad():
        arg_detached = arg.detach()
        total = int(arg_detached.numel())
        clamped = int((arg_detached < threshold).sum().item())
        min_arg = float(arg_detached.min().item())
        min_margin = float((arg_detached - 1.0).min().item())

        _STATS.total += total
        _STATS.clamped += clamped
        _STATS.min_arg = min(_STATS.min_arg, min_arg)
        _STATS.min_margin = min(_STATS.min_margin, min_margin)

        tag_stats = _STATS.by_tag.setdefault(tag, _TagStats())
        tag_stats.total += total
        tag_stats.clamped += clamped
        tag_stats.min_arg = min(tag_stats.min_arg, min_arg)
        tag_stats.eps_min = min(tag_stats.eps_min, eps_f)
        tag_stats.eps_max = max(tag_stats.eps_max, eps_f)

        k = _max_per_call()
        if k > 0:
            flat = (arg_detached - 1.0).reshape(-1)
            if flat.numel() <= k:
                sample = flat
            else:
                idx = torch.randint(0, flat.numel(), (k,), device=flat.device)
                sample = flat[idx]
            _STATS.samples.append(sample.detach().float().cpu())


def _summary_dict() -> dict:
    if _STATS.total <= 0:
        return {
            "enabled": _is_enabled(),
            "total": 0,
            "clamped": 0,
            "clamp_rate": None,
        }

    out = {
        "enabled": _is_enabled(),
        "run_id": os.environ.get("HNO_CLAMP_STATS_RUN_ID", ""),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": int(_STATS.total),
        "clamped": int(_STATS.clamped),
        "clamp_rate": float(_STATS.clamped) / float(_STATS.total),
        "min_arg": float(_STATS.min_arg),
        "min_margin": float(_STATS.min_margin),
        "by_tag": {},
    }

    max_s = _max_samples()
    if max_s > 0 and _STATS.samples:
        samples = torch.cat(_STATS.samples, dim=0)
        if samples.numel() > max_s:
            idx = torch.randint(0, samples.numel(), (max_s,))
            samples = samples[idx]
        qs = torch.tensor([0.0, 0.001, 0.01, 0.05, 0.5], dtype=torch.float32)
        qv = torch.quantile(samples, qs).tolist()
        out["margin_quantiles"] = {
            "p0": float(qv[0]),
            "p0.1": float(qv[1]),
            "p1": float(qv[2]),
            "p5": float(qv[3]),
            "p50": float(qv[4]),
        }

    for tag, stats in _STATS.by_tag.items():
        out["by_tag"][tag] = {
            "total": int(stats.total),
            "clamped": int(stats.clamped),
            "clamp_rate": float(stats.clamped) / float(stats.total) if stats.total else None,
            "min_arg": float(stats.min_arg),
            "eps_min": float(stats.eps_min) if stats.eps_min != float("inf") else None,
            "eps_max": float(stats.eps_max) if stats.eps_max != float("-inf") else None,
        }

    return out


def dump_json(path: str) -> Optional[dict]:
    if not _is_enabled():
        return None

    summary = _summary_dict()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def _atexit_dump() -> None:
    if not _is_enabled():
        return

    out_path = os.environ.get("HNO_CLAMP_STATS_OUT", "").strip()
    if out_path:
        try:
            dump_json(out_path)
        except Exception as e:
            print(f"[HNO_CLAMP_STATS][WARN] failed to dump json: {e}")
    try:
        s = _summary_dict()
        if s.get("total", 0) > 0:
            print(
                "[HNO_CLAMP_STATS] clamp_rate="
                f"{s['clamp_rate']:.6e} ({s['clamped']}/{s['total']}), "
                f"min(arg-1)={s.get('min_margin', None)}"
           )
    except Exception:
        pass


atexit.register(_atexit_dump)

