

import json
import os
import re

# ── UUID detection ────────────────────────────────────────────────────────────

_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
    re.IGNORECASE,
)


def _is_uuid(value: str) -> bool:
    return isinstance(value, str) and bool(_UUID_RE.match(value))


# ── Natural-order sort key ─────────────────────────────────────────────────────

def _natural_key(s: str):
    """Returns a sort key so that numeric sub-strings sort numerically."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


# ── Phase 1: collect every UUID in the tree ───────────────────────────────────

def _collect_uuids(obj, id_set: set, ref_set: set) -> None:
    """
    Walk the JSON tree and populate:
    - id_set  : UUIDs that appear as the value of an "id" key
    - ref_set : UUIDs that appear as values in any other position
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "id" and _is_uuid(v):
                id_set.add(v)
            elif isinstance(v, str) and _is_uuid(v):
                ref_set.add(v)
            if isinstance(v, (dict, list)):
                _collect_uuids(v, id_set, ref_set)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, str) and _is_uuid(item):
                ref_set.add(item)
            elif isinstance(item, (dict, list)):
                _collect_uuids(item, id_set, ref_set)


# ── Phase 2: build UUID → ref-N mapping ──────────────────────────────────────

def _build_uuid_map(ref_set: set) -> dict:
    """
    Every UUID in ref_set (i.e. referenced somewhere outside an 'id' field)
    gets a deterministic symbolic name ref-1, ref-2, …
    UUIDs that appear ONLY as 'id' field values (not in ref_set) are NOT
    added here so their 'id' entries will be dropped entirely.
    """
    return {uid: f"ref-{i}" for i, uid in enumerate(sorted(ref_set), start=1)}


# ── Phase 3: transform tree ───────────────────────────────────────────────────

def _transform(obj, uuid_map: dict):
    """
    Recursively:
    - 'id' fields whose UUID is referenced  → keep, replace value with ref-N
    - 'id' fields whose UUID is unreferenced → drop the field entirely
    - Any other field whose value is a UUID  → replace with ref-N
    - All dict keys sorted in natural order
    """
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if k == "id" and _is_uuid(v):
                if v in uuid_map:          # referenced ID – keep with symbolic name
                    new_dict[k] = uuid_map[v]
                # else: unreferenced ID – drop the field entirely
            elif isinstance(v, str) and _is_uuid(v):
                new_dict[k] = uuid_map.get(v, v)   # replace with ref-N
            elif isinstance(v, (dict, list)):
                new_dict[k] = _transform(v, uuid_map)
            else:
                new_dict[k] = v
        return dict(sorted(new_dict.items(), key=lambda x: _natural_key(x[0])))

    elif isinstance(obj, list):
        result = []
        for item in obj:
            if isinstance(item, str) and _is_uuid(item):
                result.append(uuid_map.get(item, item))
            else:
                result.append(_transform(item, uuid_map))
        return result

    return obj


# ── File-level processing ─────────────────────────────────────────────────────

def process_file(filepath: str, source_dir: str, output_dir: str) -> None:
    with open(filepath, encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            print(f"  SKIP  {filepath}  (invalid JSON: {exc})")
            return

    if not isinstance(data, dict) or "realm" not in data:
        print(f"  SKIP  {filepath}  (no 'realm' key)")
        return

    print(f"  OK    {filepath}")

    id_set: set = set()
    ref_set: set = set()
    _collect_uuids(data, id_set, ref_set)

    uuid_map = _build_uuid_map(ref_set)
    transformed = _transform(data, uuid_map)

    # Preserve sub-directory structure under output_dir
    rel = os.path.relpath(filepath, source_dir)
    out_path = os.path.join(output_dir, rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(transformed, fh, indent=2, ensure_ascii=False)

    removed  = len(id_set - ref_set)
    replaced = len(id_set & ref_set)
    orphan   = len(ref_set - id_set)
    print(f"        ids removed={removed}  ids replaced={replaced}  orphan refs={orphan}")
    print(f"        → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

source_realm_file = "C:\\sandeep\\daily\\20260323"




def main() -> None:
    source_dir = source_realm_file
    output_dir = os.path.join(source_dir, "cleaned")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Source : {source_dir}")
    print(f"Output : {output_dir}")
    print()

    count = 0
    for dirpath, _dirs, filenames in os.walk(source_dir):
        # Skip the output directory to avoid re-processing cleaned files
        if os.path.abspath(dirpath).startswith(os.path.abspath(output_dir)):
            continue
        for filename in filenames:
            if filename.lower().endswith(".json"):
                process_file(os.path.join(dirpath, filename), source_dir, output_dir)
                count += 1

    print(f"\nDone – {count} JSON file(s) examined.")


if __name__ == "__main__":
    main()
