"""Cast files (role tables): accepted shapes, the folder-safety guard, and the run.yaml loop."""
from pathlib import Path

import pytest

from vrs.cast import CastError, dump_cast, load_cast, normalize
from vrs.cli import main
from vrs.config import Config, ConfigError


def write(tmp_path, text, name="cast.yaml"):
    p = tmp_path / name
    p.write_text(text, "utf-8")
    return p


def audio(tmp_path, n=3):
    import numpy as np
    import soundfile as sf
    d = tmp_path / "audio"
    d.mkdir(exist_ok=True)
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    for i in range(n):
        sf.write(str(d / f"v{i}.wav"), (0.2 * np.sin(2 * np.pi * (150 + 40 * i) * t)).astype(np.float32), 16000)
    return d


def test_mapping_pairs_and_bare_strings_all_read_the_same_way():
    m = normalize({"爱丽丝": "アリス（CV）"})
    p = normalize([["爱丽丝", "アリス（CV）"]])
    b = normalize(["爱丽丝"])
    assert m == p
    assert b == [["爱丽丝", "爱丽丝"]], "a bare string uses itself as the label"


def test_empty_label_falls_back_to_the_value():
    assert normalize({"爱丽丝": ""}) == [["爱丽丝", "爱丽丝"]]


def test_value_and_label_get_trimmed():
    assert normalize([[" 爱丽丝 ", " アリス "]]) == [["爱丽丝", "アリス"]]


@pytest.mark.parametrize("bad", [
    "爱/丽丝", 'a"b', "a:b", "a*b", "a?b", "a<b", "a|b", "a\\b",
])
def test_path_hostile_values_are_rejected_at_init(bad):
    with pytest.raises(CastError, match="folder name"):
        normalize({bad: "label"})


def test_dotted_names_rejected_but_surrounding_space_is_trimmed():
    for bad in (".hidden", "名字.", ".."):
        with pytest.raises(CastError, match='"."'):
            normalize({bad: "label"})
    assert normalize({" 爱丽丝 ": "label"}) == [["爱丽丝", "label"]]


def test_overlong_value_rejected():
    with pytest.raises(CastError, match="60"):
        normalize({"角" * 61: "label"})


def test_case_insensitive_duplicate_merges_are_rejected():
    with pytest.raises(CastError, match="duplicates"):
        normalize([["Alice", "a"], ["alice", "b"]])


def test_reserved_names_would_be_routed_to_the_unresolved_pile():
    for bad in ("未定", "需复核", "unresolved"):
        with pytest.raises(CastError, match="reserved"):
            normalize({bad: "label"})


def test_entry_that_is_not_a_role_reports_the_position():
    with pytest.raises(CastError, match="entry #2"):
        normalize(["爱丽丝", ["a", "b", "c"]])


def test_empty_table_is_an_error_not_a_silent_dropdown():
    with pytest.raises(CastError):
        normalize([])
    with pytest.raises(CastError):
        normalize({})


def test_load_cast_reads_yaml_and_json(tmp_path):
    y = write(tmp_path, 'roles:\n  爱丽丝: "アリス"\n  鲍勃: Bob\n')
    assert load_cast(y) == [["爱丽丝", "アリス"], ["鲍勃", "Bob"]]
    j = write(tmp_path, '{"roles": [["爱丽丝", "アリス"]]}', name="cast.json")
    assert load_cast(j) == [["爱丽丝", "アリス"]]
    bare = write(tmp_path, "- 爱丽丝\n- 鲍勃\n", name="bare.yaml")
    assert load_cast(bare) == [["爱丽丝", "爱丽丝"], ["鲍勃", "鲍勃"]]


def test_load_cast_rejects_a_file_without_roles_key(tmp_path):
    p = write(tmp_path, "people:\n  a: b\n")
    with pytest.raises(CastError, match="no top-level"):
        load_cast(p)


def test_load_cast_rejects_missing_and_unparseable(tmp_path):
    with pytest.raises(CastError, match="not found"):
        load_cast(tmp_path / "nope.yaml")
    p = write(tmp_path, "roles: [")
    with pytest.raises(CastError, match="parseable"):
        load_cast(p)


def test_dump_cast_round_trips(tmp_path):
    entries = normalize({"爱丽丝": "アリス", "鲍勃": "Bob"})
    p = write(tmp_path, dump_cast(entries))
    assert load_cast(p) == entries


def set_roles(run_dir, value):
    """hand-edit run.yaml the way a person would, keeping the rest of the file intact"""
    import yaml
    f = Path(run_dir) / "run.yaml"
    data = yaml.safe_load(f.read_text("utf-8"))
    data["roles"] = value
    f.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), "utf-8")


def test_run_yaml_accepts_the_mapping_and_the_pair_form(tmp_path):
    run = tmp_path / "run"
    Config(audio_dir=str(audio(tmp_path))).save(str(run))
    want = [["爱丽丝", "アリス"], ["鲍勃", "鲍勃"]]
    set_roles(run, {"爱丽丝": "アリス", "鲍勃": ""})
    assert Config.load(str(run)).roles == want
    set_roles(run, [["爱丽丝", "アリス"], ["鲍勃", "鲍勃"]])
    assert Config.load(str(run)).roles == want


def test_run_yaml_rejects_a_role_that_cannot_be_a_folder(tmp_path):
    run = tmp_path / "run"
    Config(audio_dir=str(audio(tmp_path))).save(str(run))
    set_roles(run, {"天/使": "label"})
    with pytest.raises(ConfigError, match="folder name"):
        Config.load(str(run))


def test_saved_run_yaml_uses_the_readable_mapping_form(tmp_path):
    run = tmp_path / "run"
    Config(audio_dir=str(audio(tmp_path)), roles=[["爱丽丝", "アリス"]]).save(str(run))
    assert "爱丽丝: アリス" in (run / "run.yaml").read_text("utf-8")


def test_cli_init_consumes_a_cast_file(tmp_path, capsys):
    a = audio(tmp_path)
    cast = write(tmp_path, 'roles:\n  爱丽丝: "アリス（CV）"\n  旁白或系统语音: ナレーション\n')
    code = main(["init", "--audio-dir", str(a), "--run-dir", str(tmp_path / "r"),
                 "--roles-file", str(cast)])
    assert code == 0
    assert "2 in the page dropdown" in capsys.readouterr().out
    assert Config.load(str(tmp_path / "r")).roles == [["爱丽丝", "アリス（CV）"],
                                                      ["旁白或系统语音", "ナレーション"]]


def test_cli_init_fails_cleanly_on_a_bad_cast_file(tmp_path, capsys):
    a = audio(tmp_path)
    bad = write(tmp_path, 'roles:\n  Alice: x\n  alice: y\n')
    assert main(["init", "--audio-dir", str(a), "--run-dir", str(tmp_path / "r2"),
                 "--roles-file", str(bad)]) == 2
    assert "cast file error" in capsys.readouterr().err


def test_cli_init_without_a_cast_file_keeps_the_default_dropdown(tmp_path):
    a = audio(tmp_path)
    assert main(["init", "--audio-dir", str(a), "--run-dir", str(tmp_path / "r")]) == 0


def test_cli_init_validates_roles_given_through_set(tmp_path, capsys):
    a = audio(tmp_path)
    code = main(["init", "--audio-dir", str(a), "--run-dir", str(tmp_path / "r"),
                 "--set", "roles=[[爱丽丝, アリス], 鲍勃]"])
    assert code == 0
    assert Config.load(str(tmp_path / "r")).roles == [["爱丽丝", "アリス"], ["鲍勃", "鲍勃"]]
    bad = main(["init", "--audio-dir", str(a), "--run-dir", str(tmp_path / "r2"),
                "--set", 'roles=[["a/b", "x"]]'])
    assert bad == 2
    assert "cast file error" in capsys.readouterr().err
