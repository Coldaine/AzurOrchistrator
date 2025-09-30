from __future__ import annotations

import numpy as np

from azl_bot.core.hashing import FrameHasher


def test_is_stable_reaches_threshold():
        hasher = FrameHasher(similarity_threshold=0.99)
        img = np.random.randint(0, 256, (16, 16), dtype=np.uint8)

        # First call establishes baseline
        assert hasher.is_stable(img, required_matches=3) is False
        # Second and third are still not enough
        assert hasher.is_stable(img, required_matches=3) is False
        assert hasher.is_stable(img, required_matches=3) is True


def test_is_stable_resets_on_change():
        hasher = FrameHasher(similarity_threshold=0.95)
        img1 = np.zeros((16, 16), dtype=np.uint8)
        img2 = np.ones((16, 16), dtype=np.uint8) * 255

        assert hasher.is_stable(img1, required_matches=2) is False
        # Change should reset
        assert hasher.is_stable(img2, required_matches=2) is False
        # Need two in a row after reset
        assert hasher.is_stable(img2, required_matches=2) is True