        """Test stability detection."""
        img = np.random.randint(0, 256, (8, 8), dtype=np.uint8)

        # First call establishes baseline (always returns True for change)
        stable1 = self.hasher.is_stable(img, required_matches=3)
        assert stable1 is False

        # Second call should still not be stable (only 1 stable frame so far)
        stable2 = self.hasher.is_stable(img, required_matches=3)
        assert stable2 is False

        # Third call should still not be stable (only 2 stable frames so far)
        stable3 = self.hasher.is_stable(img, required_matches=3)
        assert stable3 is False

        # Fourth call should now be stable (3 stable frames)
        stable4 = self.hasher.is_stable(img, required_matches=3)
        assert stable4 is True