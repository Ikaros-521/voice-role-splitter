"""Role tables (cast files): the contract behind `vrs init --roles-file` and `run.yaml: roles`.

The first element of each entry is not a display string - it is the canonical role id that
ends up in roles.csv, becomes the exported folder name, and decides cluster merging (two
clusters given the same value are the same character). So it is validated here, at init
time, instead of failing later as a broken path or a silently duplicated role.

Three input shapes are accepted, so a hand-written file and an exported one both work:

    roles: {伊卡洛斯: "イカロス（早見沙織）"}     # mapping, shortest to write
    roles: [[伊卡洛斯, "イカロス（早見沙織）"]]    # pairs, what run.yaml stores
    roles: [伊卡洛斯]                            # value doubles as the label
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

# characters that must not appear in a role id: path separators, Windows-reserved, control
UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
# vrs/roles.py routes these to the _未确定 pile, so a role may not claim one
RESERVED = {"未定", "需复核", "unresolved"}
MAX_LEN = 60


class CastError(ValueError):
    """message is written to be shown to a user, not logged"""


def parse(entry, pos: int) -> list:
    """accept [value, label], {value: label} and the bare string "value" """
    if isinstance(entry, (list, tuple)) and len(entry) == 2:
        return [str(entry[0]), str(entry[1])]
    if isinstance(entry, dict) and len(entry) == 1:
        (v, l), = entry.items()
        return [str(v), str(l)]
    if isinstance(entry, str) and entry.strip():
        return [entry, entry]
    raise CastError(f'entry #{pos} is not a role: {entry!r} '
                    f'(expected [value, label], {{value: label}} or a plain string)')


def normalize(raw, where: str = "roles") -> list:
    """[[value, label], ...] from a mapping, a list, or one already-normalised list"""
    if isinstance(raw, dict):
        entries = [[str(k), ("" if v is None else str(v))] for k, v in raw.items()]
    elif isinstance(raw, (list, tuple)):
        entries = [parse(e, i) for i, e in enumerate(raw, 1)]
    else:
        raise CastError(f'{where}: expected a mapping or a list of roles, got '
                        f'{type(raw).__name__}')
    if not entries:
        raise CastError(f'{where}: empty - the annotation page needs at least one role '
                        f'(it always adds "— 未定 —" itself)')
    return validate(entries)


def validate(entries: list) -> list:
    seen: dict = {}
    for pos, e in enumerate(entries, 1):
        if len(e) != 2:
            raise CastError(f"entry #{pos} must be [value, label], got {e!r}")
        raw, label = str(e[0]), str(e[1]).strip()
        value = raw.strip()          # surrounding spaces are harmless: they never reach disk
        if not value:
            raise CastError(f"entry #{pos} has an empty value; the value is the role name "
                            f"written to roles.csv and used as the export folder name")
        if value.startswith(".") or value.endswith("."):
            raise CastError(f'value {value!r} (entry #{pos}) starts or ends with ".": Windows '
                            f"refuses such folder names, so the export would silently drop "
                            f"this role")
        if value in RESERVED:
            raise CastError(f'value {value!r} (entry #{pos}) is reserved: {sorted(RESERVED)} '
                            f'are how the pipeline marks "not decided yet", so every file you '
                            f'name with it would land in _未确定 instead of a role folder')
        if UNSAFE.search(value):
            raise CastError(f'value {value!r} (entry #{pos}) cannot be a folder name: it '
                            f'contains one of \\ / : * ? " < > | - keep that character in '
                            f'the label instead')
        if len(value) > MAX_LEN:
            raise CastError(f'value {value!r} (entry #{pos}) is longer than {MAX_LEN} '
                            f'characters; roles.csv and the folder tree get the full value')
        key = value.casefold()
        if key in seen:
            raise CastError(f'role {value!r} (entry #{pos}) duplicates {seen[key]!r}: role '
                            f'ids are merged case-insensitively, so both would become one '
                            f'character')
        seen[key] = value
        entries[pos - 1] = [value, label or value]
    return entries


def load_cast(path: str | Path) -> list:
    """read a .yaml/.yml/.json role table into [[value, label], ...], validated"""
    p = Path(path)
    if not p.exists():
        raise CastError(f'cast file not found: {p}')
    text = p.read_text("utf-8")
    try:
        data = json.loads(text) if p.suffix.lower() == ".json" else yaml.safe_load(text)
    except Exception as exc:
        raise CastError(f'{p}: not parseable as YAML/JSON ({exc})') from exc
    if data is None:
        raise CastError(f"{p}: file is empty")
    raw = data if isinstance(data, list) else _roles_key(data, p)
    return normalize(raw, where=str(p))


def _roles_key(data, p: Path):
    if not isinstance(data, dict):
        raise CastError(f'{p}: expected a mapping with roles: or a list, got '
                        f'{type(data).__name__}')
    for k in ("roles", "Roles", "cast"):
        if k in data:
            return data[k]
    raise CastError(f'{p}: no top-level "roles:" key (and the file is not a bare list of '
                    f'roles)')


def dump_cast(entries: list) -> str:
    """a cast file in the shape load_cast reads back, one line per role, commented"""
    body = yaml.safe_dump({"roles": [[v, l] for v, l in entries]}, allow_unicode=True,
                          default_flow_style=False, sort_keys=False)
    return body
