import numpy as np
from signalweave import streaming

def test_ring_buffer():
    rb = streaming.RingBuffer(10)
    rb.write(np.arange(5))
    assert len(rb.read()) == 5
    rb.write(np.arange(10))
    assert rb.is_full
    assert len(rb.read()) == 10
