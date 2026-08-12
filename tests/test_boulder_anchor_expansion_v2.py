from __future__ import annotations
import pandas as pd
from scripts.plan_boulder_anchor_expansion_v2 import build_queue

def test_queues_only_verified_noninitial_semifinals_and_finals():
    frame = pd.DataFrame([
        {"event_id": 1478, "category_round_id": 1, "event": "Bern", "gender": "Men", "round": "Semi-final", "video_id": "abc123", "official_channel": "World Climbing", "metadata_status": "Verified by public YouTube oEmbed", "official_youtube_url": "https://youtube.test/a", "duration_seconds": 100},
        {"event_id": 1479, "category_round_id": 2, "event": "Madrid", "gender": "Men", "round": "Final", "video_id": "abc124", "official_channel": "World Climbing", "metadata_status": "Verified by public YouTube oEmbed", "official_youtube_url": "https://youtube.test/b", "duration_seconds": 100},
        {"event_id": 1484, "category_round_id": 3, "event": "Chamonix", "gender": "Women", "round": "Qualification", "video_id": "abc125", "official_channel": "World Climbing", "metadata_status": "Verified by public YouTube oEmbed", "official_youtube_url": "https://youtube.test/c", "duration_seconds": 100},
    ])
    queue = build_queue(frame)
    assert queue["event_id"].tolist() == [1478]
    assert queue.iloc[0].batch_id == "boulder-anchor-expansion-v2-event-1478"
    assert not queue.iloc[0].execution_authorized
    assert not queue.iloc[0].external_transmission_authorized
