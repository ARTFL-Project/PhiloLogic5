"""Limits of the processes the loader starts."""

import resource


def raise_open_files_limit():
    """Raise the limit on the files this process, and the processes it starts, can have open, as far as allowed:
    merges of sorted files keep a lot of them open. Returns the new limit (resource.RLIM_INFINITY if unlimited)."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    for limit in (hard, 1 << 20, 10240):  # macOS doesn't allow a process an unlimited (or too high) limit
        if soft == resource.RLIM_INFINITY or (limit != resource.RLIM_INFINITY and soft >= limit):
            break  # already that high
        if hard != resource.RLIM_INFINITY and limit > hard:
            continue
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))
            break
        except (ValueError, OSError):
            continue
    return resource.getrlimit(resource.RLIMIT_NOFILE)[0]
