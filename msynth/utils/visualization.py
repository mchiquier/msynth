'''
Helper methods for visualization.
Created by Basile Van Hoorick, Feb 2022.
'''

from __init__ import *

# Library imports.
import IPython.display as ipd
import matplotlib.ticker as mticker


def play(x, Fs):
    ipd.display(ipd.Audio(x, rate=Fs))


def get_tp_matrix_plot_image(tp_matrix, title, timecell_ms, pitch_bounds, colormap='bone'):
    (P, T) = tp_matrix.shape
    assert P == 128
    num_pitches = pitch_bounds[1] - pitch_bounds[0] + 1
    freq_min = 440 * 2 ** ((pitch_bounds[0] - 69) / 12)
    freq_max = 440 * 2 ** ((pitch_bounds[1] - 69) / 12)
    duration_ms = T * timecell_ms

    fig = plt.figure()
    x = np.linspace(timecell_ms / 2.0, duration_ms - timecell_ms / 2.0, T) / 1000.0
    y = np.logspace(np.log10(freq_min), np.log10(freq_max), num_pitches)
    X, Y = np.meshgrid(x, y)

    im = plt.pcolormesh(X, Y, tp_matrix[pitch_bounds[0]:pitch_bounds[1] + 1],
                        cmap=colormap, shading='nearest')
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.yscale('log')
    ax = plt.gca()
    ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    plt.xlim(0, duration_ms / 1000)
    plt.ylim(freq_min * 2 ** (-0.5 / 12), freq_max * 2 ** (0.5 / 12))
    plt.grid(True)

    # https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = data.copy() / 255.0
    plt.close()

    return data


def gen_waveform_plot_image(xs, labels, Fs, zoom=1):
    fig = plt.figure()

    if not isinstance(xs, list):
        xs = [xs]
        labels = [labels]

    for (x, label) in zip(xs, labels):
        if zoom > 1:
            x = x[int(len(x) / 2 - len(x) / (zoom * 2)):int(len(x) / 2 + len(x) / (zoom * 2))]
        t = np.arange(len(x)) / Fs
        plt.plot(t, x, label=label)

    plt.title('Waveforms')
    plt.xlabel('Time [s]')
    plt.ylabel('Audio Signal')
    plt.xlim(0, len(x) / Fs)
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.legend()

    # https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = data.copy() / 255.0
    plt.close()

    return data


def get_stft_plot_image(f, t, Zxx, title, colormap='magma'):
    fig = plt.figure()

    plt.pcolormesh(t, f / 1000.0, np.abs(Zxx), vmin=0, vmax=0.04,
                   cmap=colormap, shading='gouraud')
    plt.title(title)
    plt.ylabel('Frequency [kHz]')
    plt.xlabel('Time [s]')
    plt.xlim(0, np.max(t))
    plt.ylim(0, np.max(f) / 1000.0 / 4.0)
    plt.grid(True)

    # https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = data.copy() / 255.0
    plt.close()

    return data