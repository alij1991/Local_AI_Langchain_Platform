from __future__ import annotations


def format_bytes_human(size_bytes: int | float | None) -> str | None:
    if size_bytes is None:
        return None
    try:
        n = float(size_bytes)
    except Exception:
        return None
    if n < 0:
        return None
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while n >= 1024 and idx < len(units) - 1:
        n /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(n)} {units[idx]}"
    if n >= 100:
        return f"{n:.0f} {units[idx]}"
    if n >= 10:
        return f"{n:.1f} {units[idx]}"
    return f"{n:.2f} {units[idx]}"
