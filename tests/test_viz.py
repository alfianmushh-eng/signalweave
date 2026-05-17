import matplotlib; matplotlib.use('Agg')
import numpy as np
from signalweave import viz

def test_plot_waveform():
    x = np.sin(np.linspace(0, 2*np.pi, 100))
    fig = viz.plot_waveform(x)
    assert fig is not None
