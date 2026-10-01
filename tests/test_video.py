import json

import pytest

from vision_loadgen import db
from vision_loadgen.config import ConfigError, FramesConfig
from vision_loadgen.sources import HEADER_NAME, MANIFEST_NAME, CorpusSource
from vision_loadgen.video import DEFAULT_REMOTE_CORPUS_ROOT, copy_hint, corpus_from_video
from vision_loadgen.video_camera import PORT, PREFIX, camera_name, clone_overrides

@pytest.fixture
def video(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
    for index in range(60):  # 2 s at 30 fps
        frame = np.full((48, 64, 3), index * 4 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_from_video_samples_at_target_fps(tmp_path, video):
    seen = []
    corpus = corpus_from_video(video, "lobby", tmp_path / "corpora", fps=5, progress=lambda done, total: seen.append(done))
    directory = tmp_path / "corpora" / "lobby"
    assert corpus.frames == 10 and corpus.fps == 5.0 and corpus.source_fps == 30.0
    assert (corpus.width, corpus.height) == (64, 48)
    assert corpus.image_root == f"{DEFAULT_REMOTE_CORPUS_ROOT}/lobby"
    lines = [json.loads(line) for line in (directory / MANIFEST_NAME).read_text().splitlines()]
    assert [line["image_path"] for line in lines[:2]] == ["000000.jpg", "000001.jpg"]
    assert [line["frame_index"] for line in lines[:3]] == [0, 6, 12]
    assert len(list(directory.glob("*.jpg"))) == 10
    assert json.loads((directory / HEADER_NAME).read_text())["frames"] == 10
    assert seen[-1] == 10
    with pytest.raises(FileExistsError):
        corpus_from_video(video, "lobby", tmp_path / "corpora", fps=5)


def test_from_video_max_seconds_and_bad_input(tmp_path, video):
    corpus = corpus_from_video(video, "short", tmp_path, fps=10, max_seconds=1)
    assert corpus.frames == 10  # 30 frames, every 3rd
    with pytest.raises(ConfigError):
        corpus_from_video(tmp_path / "missing.mp4", "x", tmp_path)
    with pytest.raises(ConfigError):
        corpus_from_video(video, "../escape", tmp_path)


def test_copy_hint():
    class Corpus:
        directory = "corpora/lobby"
        image_root = "/app/events/loadgen_corpus/lobby"
    assert copy_hint(Corpus, "admin1@host:/srv/events/loadgen_corpus/") == \
        'scp -r "corpora/lobby" admin1@host:/srv/events/loadgen_corpus/'
    assert "<modules host>" in copy_hint(Corpus)


def _corpus(tmp_path, entries, header=None):
    directory = tmp_path / "corpora" / "c"
    directory.mkdir(parents=True)
    (directory / MANIFEST_NAME).write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    if header is not None:
        (directory / HEADER_NAME).write_text(json.dumps(header))
    return FramesConfig(corpus_dir=str(tmp_path / "corpora")), directory


def test_corpus_relative_paths_resolve_to_remote_root_without_checking(tmp_path):
    frames, _ = _corpus(tmp_path, [{"image_path": "000000.jpg"}, {"image_path": "000001.jpg"}],
                        {"image_root": "/app/events/loadgen_corpus/c", "fps": 5.0})
    source = CorpusSource(frames, "c")
    assert not source.verified and source.fps == 5.0 and source.frames == 2
    assert source.frame_for(0)["image_path"] == "/app/events/loadgen_corpus/c/000000.jpg"
    override = CorpusSource(frames, "c", image_root="/data/other")
    assert override.frame_for(1)["image_path"].startswith("/data/other/")


def test_corpus_relative_paths_default_to_local_folder_and_are_checked(tmp_path):
    frames, directory = _corpus(tmp_path, [{"image_path": "000000.jpg"}])
    with pytest.raises(FileNotFoundError):
        CorpusSource(frames, "c")
    (directory / "000000.jpg").write_bytes(b"jpeg")
    source = CorpusSource(frames, "c")
    assert source.verified and source.frame_for(0)["image_path"] == str(directory / "000000.jpg")


def test_legacy_absolute_paths_unchanged(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpeg")
    frames, _ = _corpus(tmp_path, [{"image_path": str(image)}])
    assert CorpusSource(frames, "c").frame_for(3)["image_path"] == str(image)


def test_video_camera_name_and_overrides():
    assert camera_name("Lobby video #2") == f"{PREFIX}lobby-video-2"
    assert camera_name(f"{PREFIX}x") == f"{PREFIX}x"
    with pytest.raises(ConfigError):
        camera_name("!!!")
    columns = [
        db.Column("id", "uuid"), db.Column("name", "text"), db.Column("rtsp_url", "text", not_null=True),
        db.Column("ip", "text"), db.Column("port", "integer"), db.Column("is_active", "boolean"),
        db.Column("description", "text"), db.Column("region_id", "uuid"), db.Column("created_at", "timestamptz"),
        db.Column("deleted_at", "timestamptz"),
    ]
    overrides = clone_overrides(columns, "loadgen-video-lobby", "desc", "127.0.0.1")
    assert overrides["rtsp_url"] == "" and overrides["is_active"] is False and overrides["port"] == PORT
    assert overrides["name"] == "loadgen-video-lobby" and "region_id" not in overrides
    assert isinstance(overrides["id"], db.Raw) and overrides["deleted_at"] is None
    sql, params = db.build_clone_sql("cameras", columns, "id", "id", "abc", overrides)
    assert "src.\"region_id\"" in sql and params[-1] == "abc"
