"""Plan cache invalidation only when surviving requests keep their physical slots."""


class _NoIdentity:
    """Tag for a slot with no instance id; never equal to a real rid."""


def has_unknown_identity(identities):
    """True when some slot has no instance id and is being treated as new."""
    return any(ident[0] is _NoIdentity for ident in identities)


def request_identities(rids, req_uids):
    """Pair each rid with its request-instance uid.

    A client can reuse a rid. Missing uids receive fresh sentinel objects
    that never match a previous slot, forcing a cold gather rather than
    reusing an unknown request's KV.
    """
    if req_uids is None:
        return [(_NoIdentity, object()) for _ in rids]
    return [
        (rid, uid) if uid is not None else (_NoIdentity, object())
        for rid, uid in zip(rids, req_uids)
    ]


def build_req_id_status(rids, prev_rids, capacity):
    """Map destination slots to the previous slots of the same instances.

    Use identities from request_identities, never recycled pool indices.
    Nonnegative values name surviving source slots; -1 means a new request.
    Reject duplicate identities because their source would be ambiguous.
    """
    if capacity < 1 or len(rids) > capacity:
        raise ValueError("Invalid hot-cache batch capacity")
    if len(set(rids)) != len(rids):
        raise ValueError("Duplicate request ids in one batch")
    previous = list(prev_rids or [])
    if len(set(previous)) != len(previous):
        raise ValueError("Duplicate request ids in the previous batch")
    where = {rid: slot for slot, rid in enumerate(previous) if slot < capacity}
    return [where.get(rid, -1) for rid in rids]




def slots_to_invalidate(source_by_destination, capacity):
    """Return slots that cannot reuse their existing hot rows.

    Only a survivor in the same physical slot keeps its cache. Invalidate
    new, unknown, moved and padding slots; moving KV requires a separately
    validated InitKvOffload path.
    """
    keep = {
        destination
        for destination, source in enumerate(source_by_destination)
        if source == destination
    }
    return [slot for slot in range(capacity) if slot not in keep]
