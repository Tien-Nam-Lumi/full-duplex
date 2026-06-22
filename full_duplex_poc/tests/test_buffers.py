import numpy as np

from full_duplex_poc.buffers import RollingAudioBuffer


def test_rolling_buffer_trims_to_limit():
    buf = RollingAudioBuffer(max_samples=5)
    buf.append(np.array([1, 2, 3], dtype=np.float32))
    buf.append(np.array([4, 5, 6], dtype=np.float32))

    assert buf.size == 5
    np.testing.assert_array_equal(buf.to_numpy(), np.array([2, 3, 4, 5, 6], dtype=np.float32))


def test_tail_and_consume_preserve_order():
    buf = RollingAudioBuffer(max_samples=10)
    buf.extend(
        [
            np.array([1, 2], dtype=np.float32),
            np.array([3, 4, 5], dtype=np.float32),
        ]
    )

    np.testing.assert_array_equal(buf.tail(3), np.array([3, 4, 5], dtype=np.float32))
    np.testing.assert_array_equal(buf.consume(4), np.array([1, 2, 3, 4], dtype=np.float32))
    np.testing.assert_array_equal(buf.to_numpy(), np.array([5], dtype=np.float32))

