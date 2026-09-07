"""
WHALE-style FPS Audio Enhancer — standalone prototype
======================================================

What this does
---------------
Captures whatever is playing on your PC (system/game audio) via WASAPI
loopback, runs it through a small real-time DSP chain that:

  1. Compresses/limits loud transient peaks (gunshots, explosions) with a
     fast-attack / medium-release compressor.
  2. Boosts a footstep-relevant frequency band with a peaking EQ so
     footsteps stay clear and audible.

...then plays the processed audio out to whichever output device you pick
in the GUI (e.g. your headphones), *without* needing Equalizer APO.

This is a PROTOTYPE meant to prove the DSP concept and device-selection
flow. It is NOT tuned for competitive-grade low latency — expect roughly
20-60ms of added latency depending on your buffer size and audio driver.
A production version (like the real WHALE X AUDIO app) would normally be
written in C++ with WASAPI directly (or JUCE) for much lower latency and
a signed installer.

Requirements (Windows only)
----------------------------
    pip install pyaudiowpatch numpy PySimpleGUI

  - pyaudiowpatch: a PyAudio fork with WASAPI loopback support (this is
    what lets us "record" whatever is currently playing on the system).
  - numpy: fast array math for the DSP.
  - PySimpleGUI: quick desktop GUI for device selection + sliders.
    (If you don't want this dependency, see the `NO_GUI` fallback block
    at the bottom of the file, which just prompts in the console.)

How it works, briefly
----------------------
  System audio (loopback) --> capture buffer --> DSP chain --> output
                                                      |
                                            [Compressor] -> [Footstep EQ]

Tune the constants below (THRESHOLD_DB, RATIO, EQ_FREQ, EQ_GAIN_DB, etc.)
to taste, or expose them as GUI sliders (already wired up).
"""

import sys
import time
import threading
import numpy as np

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    print("ERROR: pyaudiowpatch not installed. Run: pip install pyaudiowpatch")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Audio / DSP settings (defaults — the GUI lets you change these live)
# ---------------------------------------------------------------------------
CHUNK = 512                 # frames per buffer (lower = less latency, more CPU)
SAMPLE_FORMAT = pyaudio.paFloat32

# Compressor (tames loud gunshot-type peaks)
THRESHOLD_DB = -18.0        # level above which compression kicks in
RATIO = 6.0                 # how strongly peaks above threshold get squashed
ATTACK_MS = 3.0             # how fast it reacts to a sudden loud sound
RELEASE_MS = 120.0          # how fast it recovers afterwards

# Footstep EQ (peaking boost around footstep-relevant frequencies)
EQ_FREQ = 2200.0            # Hz — center of the boost (footstep "tap" clarity)
EQ_GAIN_DB = 6.0            # how much to boost that band
EQ_Q = 1.2                  # bandwidth of the boost (higher = narrower)


# ---------------------------------------------------------------------------
# DSP building blocks
# ---------------------------------------------------------------------------
class PeakingEQ:
    """A simple biquad peaking EQ filter (RBJ cookbook formulas)."""

    def __init__(self, sample_rate, freq, gain_db, q):
        self.sample_rate = sample_rate
        self.set_params(freq, gain_db, q)
        # filter state (stereo: 2 channels of history)
        self.x1 = np.zeros(2)
        self.x2 = np.zeros(2)
        self.y1 = np.zeros(2)
        self.y2 = np.zeros(2)

    def set_params(self, freq, gain_db, q):
        A = 10 ** (gain_db / 40.0)
        w0 = 2 * np.pi * freq / self.sample_rate
        alpha = np.sin(w0) / (2 * q)
        cos_w0 = np.cos(w0)

        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A

        self.b0, self.b1, self.b2 = b0 / a0, b1 / a0, b2 / a0
        self.a1, self.a2 = a1 / a0, a2 / a0

    def process(self, samples):
        """samples: shape (n, 2) float32 in [-1, 1]. Processed in place-ish."""
        out = np.empty_like(samples)
        for ch in range(samples.shape[1]):
            x = samples[:, ch]
            y = np.empty_like(x)
            x1, x2, y1, y2 = self.x1[ch], self.x2[ch], self.y1[ch], self.y2[ch]
            for n in range(len(x)):
                x0 = x[n]
                y0 = (self.b0 * x0 + self.b1 * x1 + self.b2 * x2
                      - self.a1 * y1 - self.a2 * y2)
                y[n] = y0
                x2, x1 = x1, x0
                y2, y1 = y1, y0
            self.x1[ch], self.x2[ch], self.y1[ch], self.y2[ch] = x1, x2, y1, y2
            out[:, ch] = y
        return out


class Compressor:
    """Feed-forward RMS-ish peak compressor with attack/release smoothing."""

    def __init__(self, sample_rate, threshold_db, ratio, attack_ms, release_ms):
        self.sr = sample_rate
        self.set_params(threshold_db, ratio, attack_ms, release_ms)
        self.envelope = 0.0

    def set_params(self, threshold_db, ratio, attack_ms, release_ms):
        self.threshold = 10 ** (threshold_db / 20.0)
        self.ratio = ratio
        self.attack_coeff = np.exp(-1.0 / (self.sr * attack_ms / 1000.0))
        self.release_coeff = np.exp(-1.0 / (self.sr * release_ms / 1000.0))

    def process(self, samples):
        # Use max across channels per-sample as the detection signal.
        detect = np.max(np.abs(samples), axis=1)
        gain = np.empty_like(detect)
        env = self.envelope

        for n in range(len(detect)):
            level = detect[n]
            coeff = self.attack_coeff if level > env else self.release_coeff
            env = coeff * env + (1 - coeff) * level

            if env > self.threshold and env > 0:
                over_db = 20 * np.log10(env / self.threshold)
                reduction_db = over_db * (1 - 1.0 / self.ratio)
                gain[n] = 10 ** (-reduction_db / 20.0)
            else:
                gain[n] = 1.0

        self.envelope = env
        return samples * gain[:, None]


# ---------------------------------------------------------------------------
# Audio engine: loopback capture -> DSP -> playback to chosen device
# ---------------------------------------------------------------------------
class AudioEngine:
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.running = False
        self.thread = None

        self.compressor = None
        self.eq = None
        self.sample_rate = 48000
        self.channels = 2

        # Live-tweakable params (GUI writes to these; DSP thread reads them)
        self.threshold_db = THRESHOLD_DB
        self.ratio = RATIO
        self.attack_ms = ATTACK_MS
        self.release_ms = RELEASE_MS
        self.eq_gain_db = EQ_GAIN_DB
        self.eq_freq = EQ_FREQ
        self.eq_q = EQ_Q
        self.enabled = True

    # -- device discovery -----------------------------------------------
    def list_output_devices(self):
        """Real playback devices you can send processed audio to (headphones etc.)."""
        devices = []
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if info["maxOutputChannels"] > 0:
                devices.append((i, info["name"]))
        return devices

    def get_default_loopback_device(self):
        """The system's default speaker, opened in WASAPI loopback ('record what plays')."""
        wasapi_info = self.pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = self.pa.get_device_info_by_index(
            wasapi_info["defaultOutputDevice"]
        )
        if not default_speakers["isLoopbackDevice"]:
            for loopback in self.pa.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    return loopback
        return default_speakers

    # -- lifecycle ---------------------------------------------------------
    def start(self, output_device_index):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(
            target=self._run, args=(output_device_index,), daemon=True
        )
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def update_params(self, threshold_db=None, ratio=None, attack_ms=None,
                       release_ms=None, eq_gain_db=None, eq_freq=None,
                       eq_q=None, enabled=None):
        if threshold_db is not None: self.threshold_db = threshold_db
        if ratio is not None: self.ratio = ratio
        if attack_ms is not None: self.attack_ms = attack_ms
        if release_ms is not None: self.release_ms = release_ms
        if eq_gain_db is not None: self.eq_gain_db = eq_gain_db
        if eq_freq is not None: self.eq_freq = eq_freq
        if eq_q is not None: self.eq_q = eq_q
        if enabled is not None: self.enabled = enabled

        if self.compressor:
            self.compressor.set_params(
                self.threshold_db, self.ratio, self.attack_ms, self.release_ms
            )
        if self.eq:
            self.eq.set_params(self.eq_freq, self.eq_gain_db, self.eq_q)

    # -- main audio loop -----------------------------------------------
    def _run(self, output_device_index):
        loopback = self.get_default_loopback_device()
        self.sample_rate = int(loopback["defaultSampleRate"])
        self.channels = min(2, loopback["maxInputChannels"])

        out_info = self.pa.get_device_info_by_index(output_device_index)

        self.compressor = Compressor(
            self.sample_rate, self.threshold_db, self.ratio,
            self.attack_ms, self.release_ms
        )
        self.eq = PeakingEQ(self.sample_rate, self.eq_freq, self.eq_gain_db, self.eq_q)

        in_stream = self.pa.open(
            format=SAMPLE_FORMAT,
            channels=self.channels,
            rate=self.sample_rate,
            frames_per_buffer=CHUNK,
            input=True,
            input_device_index=loopback["index"],
        )
        out_stream = self.pa.open(
            format=SAMPLE_FORMAT,
            channels=self.channels,
            rate=self.sample_rate,
            frames_per_buffer=CHUNK,
            output=True,
            output_device_index=output_device_index,
        )

        print(f"Capturing: {loopback['name']}  ->  Output: {out_info['name']}")
        print(f"Sample rate: {self.sample_rate} Hz, channels: {self.channels}")

        try:
            while self.running:
                data = in_stream.read(CHUNK, exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.float32).reshape(-1, self.channels)

                if self.enabled:
                    samples = self.compressor.process(samples)
                    samples = self.eq.process(samples)
                    samples = np.clip(samples, -1.0, 1.0)

                out_stream.write(samples.astype(np.float32).tobytes())
        finally:
            in_stream.stop_stream(); in_stream.close()
            out_stream.stop_stream(); out_stream.close()

    def close(self):
        self.stop()
        self.pa.terminate()


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
SETTINGS_FILE = "whale_audio_settings.json"

PARAM_KEYS = [
    "-THRESH-", "-RATIO-", "-ATTACK-", "-RELEASE-",
    "-EQGAIN-", "-EQFREQ-", "-EQQ-", "-ENABLED-",
]


def save_settings(values, path=SETTINGS_FILE):
    import json
    data = {k: values[k] for k in PARAM_KEYS if k in values}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_settings(path=SETTINGS_FILE):
    import json, os
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_gui():
    import PySimpleGUI as sg

    engine = AudioEngine()
    devices = engine.list_output_devices()
    device_names = [f"{idx}: {name}" for idx, name in devices]

    saved = load_settings() or {}
    v = lambda key, default: saved.get(key, default)  # noqa: E731

    layout = [
        [sg.Text("เลือกอุปกรณ์เสียงออก (หูฟัง):")],
        [sg.Combo(device_names, key="-DEVICE-", size=(50, 1),
                   default_value=device_names[0] if device_names else "")],
        [sg.Checkbox("เปิดใช้งานประมวลผลเสียง", default=v("-ENABLED-", True),
                      key="-ENABLED-", enable_events=True)],
        [sg.HorizontalSeparator()],
        [sg.Text("ลดเสียงปืน (Compressor)")],
        [sg.Text("Threshold (dB)", size=(14, 1)),
         sg.Slider((-40, 0), v("-THRESH-", THRESHOLD_DB), orientation="h",
                    key="-THRESH-", enable_events=True, size=(30, 15))],
        [sg.Text("Ratio", size=(14, 1)),
         sg.Slider((1, 20), v("-RATIO-", RATIO), orientation="h",
                    key="-RATIO-", enable_events=True, size=(30, 15))],
        [sg.Text("Attack (ms)", size=(14, 1)),
         sg.Slider((0.5, 30), v("-ATTACK-", ATTACK_MS), orientation="h",
                    key="-ATTACK-", enable_events=True, size=(30, 15),
                    resolution=0.5)],
        [sg.Text("Release (ms)", size=(14, 1)),
         sg.Slider((20, 500), v("-RELEASE-", RELEASE_MS), orientation="h",
                    key="-RELEASE-", enable_events=True, size=(30, 15))],
        [sg.HorizontalSeparator()],
        [sg.Text("เพิ่มความชัดเสียงเท้า (Footstep EQ)")],
        [sg.Text("Boost (dB)", size=(14, 1)),
         sg.Slider((0, 15), v("-EQGAIN-", EQ_GAIN_DB), orientation="h",
                    key="-EQGAIN-", enable_events=True, size=(30, 15))],
        [sg.Text("Frequency (Hz)", size=(14, 1)),
         sg.Slider((500, 6000), v("-EQFREQ-", EQ_FREQ), orientation="h",
                    key="-EQFREQ-", enable_events=True, size=(30, 15))],
        [sg.Text("Q (ความแคบ)", size=(14, 1)),
         sg.Slider((0.3, 5), v("-EQQ-", EQ_Q), orientation="h",
                    key="-EQQ-", enable_events=True, size=(30, 15),
                    resolution=0.1)],
        [sg.HorizontalSeparator()],
        [sg.Button("เริ่มทำงาน", key="-START-"),
         sg.Button("หยุด", key="-STOP-"),
         sg.Button("บันทึกค่านี้เป็นค่าเริ่มต้น", key="-SAVE-"),
         sg.Button("ปิดโปรแกรม", key="-EXIT-")],
        [sg.Text("", key="-STATUS-", size=(60, 1))],
    ]

    window = sg.Window("WHALE-style Audio Enhancer (Prototype)", layout)

    # Apply any settings loaded from disk right away, even before Start.
    if saved:
        engine.update_params(
            threshold_db=saved.get("-THRESH-"),
            ratio=saved.get("-RATIO-"),
            attack_ms=saved.get("-ATTACK-"),
            release_ms=saved.get("-RELEASE-"),
            eq_gain_db=saved.get("-EQGAIN-"),
            eq_freq=saved.get("-EQFREQ-"),
            eq_q=saved.get("-EQQ-"),
            enabled=saved.get("-ENABLED-"),
        )

    while True:
        event, values = window.read(timeout=100)
        if event in (sg.WINDOW_CLOSED, "-EXIT-"):
            break

        if event == "-START-":
            if not device_names:
                window["-STATUS-"].update("ไม่พบอุปกรณ์เสียงออก")
                continue
            idx = int(values["-DEVICE-"].split(":")[0])
            engine.start(idx)
            window["-STATUS-"].update("กำลังทำงาน...")

        if event == "-STOP-":
            engine.stop()
            window["-STATUS-"].update("หยุดแล้ว")

        if event == "-SAVE-":
            save_settings(values)
            window["-STATUS-"].update("บันทึกค่าเริ่มต้นแล้ว — ครั้งหน้าจะเปิดมาพร้อมค่านี้")

        if event in ("-ENABLED-", "-THRESH-", "-RATIO-", "-ATTACK-",
                     "-RELEASE-", "-EQGAIN-", "-EQFREQ-", "-EQQ-"):
            engine.update_params(
                threshold_db=values["-THRESH-"],
                ratio=values["-RATIO-"],
                attack_ms=values["-ATTACK-"],
                release_ms=values["-RELEASE-"],
                eq_gain_db=values["-EQGAIN-"],
                eq_freq=values["-EQFREQ-"],
                eq_q=values["-EQQ-"],
                enabled=values["-ENABLED-"],
            )

    engine.close()
    window.close()


# ---------------------------------------------------------------------------
# Console fallback (if you don't want to install PySimpleGUI)
# ---------------------------------------------------------------------------
def run_console():
    engine = AudioEngine()
    devices = engine.list_output_devices()
    print("\nอุปกรณ์เสียงออกที่ใช้ได้:")
    for idx, name in devices:
        print(f"  [{idx}] {name}")
    choice = int(input("\nเลือก device index สำหรับหูฟัง: "))
    engine.start(choice)
    print("กำลังทำงาน... กด Ctrl+C เพื่อหยุด")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    engine.close()


if __name__ == "__main__":
    try:
        import PySimpleGUI  # noqa: F401
        run_gui()
    except ImportError:
        print("PySimpleGUI ไม่ได้ติดตั้ง — รันแบบ console แทน")
        print("(ติดตั้งด้วย: pip install PySimpleGUI  เพื่อใช้ GUI)")
        run_console()
