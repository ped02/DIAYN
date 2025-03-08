import unittest
from DIAYN.replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    def test_buffer_capacity(self):
        buffer_size = 5
        buffer = ReplayBuffer(buffer_size)

        # Add more than buffer_size elements
        for i in range(10):
            buffer.add(i)

        self.assertEqual(len(buffer.buffer), buffer_size)

        # FIFO behavior
        self.assertEqual(list(buffer.buffer), list(range(5, 10)))

    def test_reproducible_sampling(self):
        buffer_size = 10

        buffer = ReplayBuffer(buffer_size, seed=42)
        for i in range(buffer_size):
            buffer.add(i)

        sample_1 = buffer.sample(5)

        # Create a new buffer with same seed
        buffer = ReplayBuffer(buffer_size, seed=42)
        for i in range(buffer_size):
            buffer.add(i)
        sample_2 = buffer.sample(5)

        self.assertEqual(sample_1, sample_2)  # Check repeatability with seed

    def test_rng_state_save_load(self):
        buffer_size = 10
        buffer = ReplayBuffer(buffer_size)

        for i in range(buffer_size):
            buffer.add(i)

        # Sample some to randomize seed a little
        for i in range(7):
            buffer.sample()

        # Get state before
        rng_state = buffer.get_rng_state()

        # Sample original buffer
        sample_1 = buffer.sample(5)

        # Setup new buffer and set state
        buffer_2 = ReplayBuffer(buffer_size)

        for i in range(buffer_size):
            buffer_2.add(i)

        buffer_2.set_rng_state(rng_state)

        # Sample new buffer with same state
        sample_2 = buffer_2.sample(5)

        self.assertEqual(sample_1, sample_2)


if __name__ == '__main__':
    unittest.main()
