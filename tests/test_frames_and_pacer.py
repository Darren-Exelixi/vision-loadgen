from collections import Counter

from vision_loadgen.config import FramesConfig
from vision_loadgen.frames import LOADGEN_TAG, build_message, is_synthetic
from vision_loadgen.pacer import Pacer

SOURCE = {"camera_id": "real", "timestamp": 1.0, "image_path": "/data/frames/a.jpg",
          "frame_date": "2026-09-29", "frame_hour": "09", "extra": "dropped"}


def test_build_message_live_keeps_producer_dates():
    message = build_message(SOURCE, "syn-1", 1790000000.0, "run", FramesConfig(), passthrough_dates=True)
    assert message == {"camera_id": "syn-1", "timestamp": 1790000000.0, "image_path": "/data/frames/a.jpg",
                       "frame_date": "2026-09-29", "frame_hour": "09", LOADGEN_TAG: "run"}


def test_build_message_corpus_recomputes_dates_in_timezone():
    frames = FramesConfig(timezone="Asia/Dubai")
    message = build_message(SOURCE, "syn-1", 1790000000.0, "run", frames, passthrough_dates=False)
    # 1790000000 = 2026-09-21 14:13:20 UTC = 18:13 in Dubai
    assert (message["frame_date"], message["frame_hour"]) == ("2026-09-21", "18")


def test_is_synthetic_by_tag_or_id():
    assert is_synthetic({"camera_id": "x", LOADGEN_TAG: "run"}, set())
    assert is_synthetic({"camera_id": "syn"}, {"syn"})
    assert not is_synthetic({"camera_id": "real"}, {"syn"})


def test_pacer_sends_each_camera_at_fps_without_bursts():
    pacer = Pacer(fps=5)
    pacer.set_active(4, now=0.0)
    sends = Counter()
    per_tick = []
    for step in range(1, 1001):
        due = pacer.due(step * 0.01)
        per_tick.append(len(due))
        sends.update(due)
    assert set(sends) == {0, 1, 2, 3}
    assert all(49 <= count <= 51 for count in sends.values())
    assert max(per_tick) <= 2


def test_pacer_skips_missed_slots_instead_of_bursting():
    pacer = Pacer(fps=10)
    pacer.set_active(1, now=0.0)
    assert pacer.due(0.0) == [0]
    assert pacer.due(5.0) == [0]
    assert pacer.due(5.05) == []


def test_pacer_shrinks_and_grows():
    pacer = Pacer(fps=1)
    pacer.set_active(3, now=0.0)
    pacer.set_active(1, now=0.0)
    assert set(pacer.due(2.0)) == {0}
    pacer.set_active(2, now=2.0)
    assert set(pacer.due(4.0)) == {0, 1}


def test_pacer_pause_then_resume_spreads_again():
    pacer = Pacer(fps=1)
    pacer.set_active(4, now=0.0)
    pacer.set_active(0, now=0.5)
    assert pacer.next_due() is None
    pacer.set_active(4, now=100.0)
    assert pacer.due(100.0) == [0]
    assert pacer.due(100.3) == [1]
