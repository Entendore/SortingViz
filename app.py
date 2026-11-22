#!/usr/bin/env python3
"""
Sorting Algorithm Visualizer — PySide6 Desktop Application
Showcases 15 sorting algorithms with animated bar visualizations,
step-by-step mode, adjustable speed/size, live statistics, and
real-time audio synthesis that maps bar values to musical tones.
"""

import sys
import random
import math
import numpy as np
import sounddevice as sd
from enum import Enum
from dataclasses import dataclass, field
from typing import Generator, Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QSlider,
    QGroupBox, QFrame, QSplitter, QSizePolicy, QCheckBox,
)
from PySide6.QtCore import Qt, QTimer, QRectF, QPointF, QEasingCurve
from PySide6.QtGui import (
    QPainter, QColor, QBrush, QPen, QFont, QLinearGradient,
    QPainterPath, QRadialGradient, QPalette, QConicalGradient,
)

# ---------------------------------------------------------------------------
# Audio Engine — real-time tone synthesis
# ---------------------------------------------------------------------------

SAMPLE_RATE = 44100


class AudioEngine:
    """Plays short sine-wave tones mapped to array values.

    Frequencies follow a pentatonic scale so that random accesses
    never sound dissonant.  A low-pass filtered decay envelope keeps
    the sound pleasant even at high speeds.
    """

    # C pentatonic across 4 octaves (C3 - C7)
    PENTATONIC = [130.81, 146.83, 164.81, 196.00, 220.00,
                  261.63, 293.66, 329.63, 392.00, 440.00,
                  523.25, 587.33, 659.26, 783.99, 880.00,
                  1046.50, 1174.66, 1318.51, 1567.98, 1760.00]

    def __init__(self):
        self.enabled = True
        self.volume = 0.35
        self._stream: Optional[sd.OutputStream] = None
        self._min_freq = self.PENTATONIC[0]
        self._max_freq = self.PENTATONIC[-1]

    # lifecycle --------------------------------------------------------

    def start(self):
        if self._stream is None:
            self._stream = sd.OutputStream(
                samplerate=SAMPLE_RATE, channels=1,
                dtype="float32", blocksize=512, latency="low",
            )
            self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # public -----------------------------------------------------------

    def play_tone(self, value: int, max_value: int = 100,
                  duration_ms: float = 60):
        """Play a tone whose frequency is mapped from *value*."""
        if not self.enabled or self._stream is None:
            return
        freq = self._value_to_freq(value, max_value)
        self._emit(freq, duration_ms)

    def play_compare(self, val_a: int, val_b: int, max_value: int = 100):
        """Play the higher of two compared values."""
        self.play_tone(max(val_a, val_b), max_value, duration_ms=50)

    def play_swap(self, val_a: int, val_b: int, max_value: int = 100):
        """Play both values as a quick two-tone chirp."""
        if not self.enabled or self._stream is None:
            return
        freq_a = self._value_to_freq(val_a, max_value)
        freq_b = self._value_to_freq(val_b, max_value)
        self._emit(freq_a, 30)
        self._emit(freq_b, 30)

    # internals --------------------------------------------------------

    def _value_to_freq(self, value: int, max_value: int) -> float:
        t = max(0.0, min(1.0, value / max(max_value, 1)))
        idx_f = t * (len(self.PENTATONIC) - 1)
        lo = int(idx_f)
        hi = min(lo + 1, len(self.PENTATONIC) - 1)
        frac = idx_f - lo
        return self.PENTATONIC[lo] * (1 - frac) + self.PENTATONIC[hi] * frac

    def _emit(self, freq: float, duration_ms: float):
        if self._stream is None:
            return
        n_samples = int(SAMPLE_RATE * duration_ms / 1000)
        if n_samples < 2:
            return
        t = np.linspace(0, duration_ms / 1000, n_samples, dtype=np.float32)
        # sine wave + subtle 2nd harmonic for richness
        wave = np.sin(2 * np.pi * freq * t)
        wave += 0.15 * np.sin(2 * np.pi * freq * 2 * t)
        # exponential decay envelope
        envelope = np.exp(-t * 40.0)
        wave *= envelope
        # apply volume
        wave *= self.volume
        # clip & write
        np.clip(wave, -1.0, 1.0, out=wave)
        try:
            self._stream.write(wave.reshape(-1, 1).astype(np.float32))
        except Exception:
            pass


# Global audio engine instance
_audio = AudioEngine()

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class BarState(Enum):
    DEFAULT   = "default"
    COMPARING = "comparing"
    SWAPPING  = "swapping"
    SORTED    = "sorted"
    PIVOT     = "pivot"
    ACTIVE    = "active"


@dataclass
class SortStep:
    array: list[int]
    comparing: list[int] = field(default_factory=list)
    swapping: list[int]  = field(default_factory=list)
    sorted: list[int]    = field(default_factory=list)
    pivot: list[int]     = field(default_factory=list)
    active: list[int]    = field(default_factory=list)


@dataclass
class AlgorithmInfo:
    name: str
    time_best: str
    time_avg: str
    time_worst: str
    space: str
    stable: bool
    description: str
    category: str = "Comparison"  # Comparison / Distribution / Hybrid / Novelty

# ---------------------------------------------------------------------------
# Algorithm Catalogue - 15 sorting algorithms
# ---------------------------------------------------------------------------

ALGORITHMS: dict[str, AlgorithmInfo] = {
    # -- Classic Comparison Sorts ----------------------------------------
    "bubble": AlgorithmInfo(
        name="Bubble Sort",
        time_best="O(n)", time_avg="O(n\u00B2)", time_worst="O(n\u00B2)",
        space="O(1)", stable=True,
        category="Comparison",
        description="Repeatedly compares adjacent elements and swaps them "
                    "if they are in the wrong order. Simple but inefficient "
                    "for large datasets. The name comes from larger elements "
                    "'bubbling' to the top of the array.",
    ),
    "selection": AlgorithmInfo(
        name="Selection Sort",
        time_best="O(n\u00B2)", time_avg="O(n\u00B2)", time_worst="O(n\u00B2)",
        space="O(1)", stable=False,
        category="Comparison",
        description="Finds the minimum element from the unsorted region and "
                    "places it at the beginning. Makes the minimum number of "
                    "swaps (at most n-1) but always scans the full unsorted "
                    "portion, making it inefficient regardless of input.",
    ),
    "insertion": AlgorithmInfo(
        name="Insertion Sort",
        time_best="O(n)", time_avg="O(n\u00B2)", time_worst="O(n\u00B2)",
        space="O(1)", stable=True,
        category="Comparison",
        description="Builds the sorted array one item at a time by inserting "
                    "each element into its correct position among the "
                    "previously sorted elements. Extremely efficient for "
                    "small or nearly-sorted arrays - used as the base case "
                    "in Timsort.",
    ),
    "cocktail": AlgorithmInfo(
        name="Cocktail Shaker Sort",
        time_best="O(n)", time_avg="O(n\u00B2)", time_worst="O(n\u00B2)",
        space="O(1)", stable=True,
        category="Comparison",
        description="A bidirectional variant of Bubble Sort that traverses "
                    "the array alternately from left-to-right and "
                    "right-to-left. This helps 'turtles' (small values near "
                    "the end) move to their correct position faster.",
    ),
    "gnome": AlgorithmInfo(
        name="Gnome Sort",
        time_best="O(n)", time_avg="O(n\u00B2)", time_worst="O(n\u00B2)",
        space="O(1)", stable=True,
        category="Comparison",
        description="Inspired by a garden gnome sorting a line of flower "
                    "pots. It works by moving forward when elements are in "
                    "order, and swapping + stepping backward when they are "
                    "not. Conceptually similar to Insertion Sort but uses "
                    "swaps instead of shifts.",
    ),
    "odd_even": AlgorithmInfo(
        name="Odd-Even Sort",
        time_best="O(n log n)", time_avg="O(n\u00B2)", time_worst="O(n\u00B2)",
        space="O(1)", stable=True,
        category="Comparison",
        description="Also known as Brick Sort. Compares all pairs of "
                    "elements at odd indices, then even indices, and "
                    "repeats until sorted. Originally designed for "
                    "parallel processing - odd and even phases can "
                    "run simultaneously on different processors.",
    ),

    # -- Efficient Comparison Sorts -------------------------------------
    "shell": AlgorithmInfo(
        name="Shell Sort",
        time_best="O(n log n)", time_avg="O(n^1.25)", time_worst="O(n\u00B2)",
        space="O(1)", stable=False,
        category="Efficient",
        description="A generalization of Insertion Sort that allows the "
                    "exchange of far-apart elements. Uses a decreasing "
                    "gap sequence (here: Ciura's sequence) to partially "
                    "sort the array, making the final insertion pass fast. "
                    "Performance depends heavily on the gap sequence chosen.",
    ),
    "comb": AlgorithmInfo(
        name="Comb Sort",
        time_best="O(n log n)", time_avg="O(n\u00B2 / 2^p)", time_worst="O(n\u00B2)",
        space="O(1)", stable=False,
        category="Efficient",
        description="Improves on Bubble Sort by eliminating turtles "
                    "(small values at the end). Uses a gap that shrinks "
                    "by a shrink factor (typically 1.3) each pass. The "
                    "name comes from the resemblance to combing through "
                    "hair - removing tangles from different distances.",
    ),
    "merge": AlgorithmInfo(
        name="Merge Sort",
        time_best="O(n log n)", time_avg="O(n log n)", time_worst="O(n log n)",
        space="O(n)", stable=True,
        category="Efficient",
        description="A divide-and-conquer algorithm that splits the array "
                    "in half, recursively sorts each half, then merges them "
                    "back together. Guarantees O(n log n) performance in "
                    "all cases but requires O(n) extra space for the merge "
                    "step.",
    ),
    "quick": AlgorithmInfo(
        name="Quick Sort",
        time_best="O(n log n)", time_avg="O(n log n)", time_worst="O(n\u00B2)",
        space="O(log n)", stable=False,
        category="Efficient",
        description="Picks a pivot element and partitions the array so "
                    "that elements smaller than the pivot come before it "
                    "and larger elements come after. Recursively sorts the "
                    "two partitions. Extremely fast in practice and the "
                    "basis for most standard library sorts.",
    ),
    "heap": AlgorithmInfo(
        name="Heap Sort",
        time_best="O(n log n)", time_avg="O(n log n)", time_worst="O(n log n)",
        space="O(1)", stable=False,
        category="Efficient",
        description="Builds a max-heap from the array, then repeatedly "
                    "extracts the maximum element and places it at the end. "
                    "Provides guaranteed O(n log n) performance with only "
                    "O(1) extra space, but poor cache performance makes it "
                    "slower than Quick Sort in practice.",
    ),
    "cycle": AlgorithmInfo(
        name="Cycle Sort",
        time_best="O(n\u00B2)", time_avg="O(n\u00B2)", time_worst="O(n\u00B2)",
        space="O(1)", stable=False,
        category="Efficient",
        description="Minimizes the total number of writes to the array. "
                    "For each element, it counts how many elements are "
                    "smaller to find its correct position, then rotates "
                    "the cycle into place. Theoretical minimum writes but "
                    "high comparison count makes it impractical for general use.",
    ),

    # -- Non-Comparison / Distribution Sorts ----------------------------
    "counting": AlgorithmInfo(
        name="Counting Sort",
        time_best="O(n + k)", time_avg="O(n + k)", time_worst="O(n + k)",
        space="O(k)", stable=True,
        category="Distribution",
        description="Counts the occurrences of each distinct value, then "
                    "uses those counts to determine positions. Not a "
                    "comparison sort - it exploits the fact that values are "
                    "integers within a known range. Extremely fast when the "
                    "range k is O(n).",
    ),
    "radix": AlgorithmInfo(
        name="Radix Sort (LSD)",
        time_best="O(d*(n+b))", time_avg="O(d*(n+b))", time_worst="O(d*(n+b))",
        space="O(n + b)", stable=True,
        category="Distribution",
        description="Sorts integers by processing individual digits from "
                    "least significant to most significant (LSD variant), "
                    "using a stable subroutine (here: counting sort per "
                    "digit). Performance depends on the number of digits d "
                    "and the base b. Not a comparison sort.",
    ),

    # -- Novelty Sort ---------------------------------------------------
    "bogo": AlgorithmInfo(
        name="Bogo Sort",
        time_best="O(n)", time_avg="O(n * n!)", time_worst="O(infinity)",
        space="O(1)", stable=False,
        category="Novelty",
        description="Randomly shuffles the array and checks if it is "
                    "sorted. Repeats until sorted. Theoretically the "
                    "worst possible sorting algorithm - the expected "
                    "number of shuffles is n! (factorial). Included for "
                    "entertainment and educational purposes only. "
                    "Array size is limited to 10 for sanity.",
    ),
}

# ---------------------------------------------------------------------------
# Color palette (dark theme)
# ---------------------------------------------------------------------------

COLORS = {
    BarState.DEFAULT:   QColor(120, 120, 135),
    BarState.COMPARING: QColor(245, 158, 11),
    BarState.SWAPPING:  QColor(239, 68, 68),
    BarState.SORTED:    QColor(34, 197, 94),
    BarState.PIVOT:     QColor(139, 92, 246),
    BarState.ACTIVE:    QColor(6, 182, 212),
}

BG_DARK       = QColor(24, 24, 27)
BG_CARD       = QColor(36, 36, 42)
BG_WIDGET     = QColor(49, 49, 58)
FG_PRIMARY    = QColor(244, 244, 245)
FG_SECONDARY  = QColor(161, 161, 170)
FG_MUTED      = QColor(113, 113, 122)
BORDER        = QColor(63, 63, 70)
ACCENT        = QColor(99, 102, 241)
ACCENT_HOVER  = QColor(129, 140, 248)
GREEN_BTN     = QColor(34, 197, 94)
RED_BTN       = QColor(239, 68, 68)
YELLOW_BTN    = QColor(245, 158, 11)
SOUND_COLOR   = QColor(168, 85, 247)

# ---------------------------------------------------------------------------
# Sorting generators  (yield SortStep for each visual frame)
# ---------------------------------------------------------------------------

def _is_sorted(a: list[int]) -> bool:
    return all(a[i] <= a[i+1] for i in range(len(a) - 1))

# -- Classic O(n^2) -----------------------------------------------------

def bubble_sort(arr: list[int]) -> Generator[SortStep, None, None]:
    a = arr[:]; n = len(a); sorted_idx: list[int] = []
    for i in range(n - 1):
        swapped = False
        for j in range(n - i - 1):
            yield SortStep(a[:], [j, j+1], [], sorted_idx[:], [], [])
            if a[j] > a[j+1]:
                a[j], a[j+1] = a[j+1], a[j]
                swapped = True
                yield SortStep(a[:], [], [j, j+1], sorted_idx[:], [], [])
        sorted_idx.append(n - 1 - i)
        if not swapped:
            sorted_idx = list(range(n)); break
    if 0 not in sorted_idx: sorted_idx.append(0)
    yield SortStep(a[:], [], [], sorted_idx, [], [])


def selection_sort(arr: list[int]) -> Generator[SortStep, None, None]:
    a = arr[:]; n = len(a); sorted_idx: list[int] = []
    for i in range(n - 1):
        min_j = i
        for j in range(i + 1, n):
            yield SortStep(a[:], [min_j, j], [], sorted_idx[:], [min_j], [])
            if a[j] < a[min_j]: min_j = j
        if min_j != i:
            a[i], a[min_j] = a[min_j], a[i]
            yield SortStep(a[:], [], [i, min_j], sorted_idx[:], [], [])
        sorted_idx.append(i)
    sorted_idx.append(n - 1)
    yield SortStep(a[:], [], [], sorted_idx, [], [])


def insertion_sort(arr: list[int]) -> Generator[SortStep, None, None]:
    a = arr[:]; n = len(a); sorted_idx: list[int] = [0]
    for i in range(1, n):
        key = a[i]; j = i - 1
        yield SortStep(a[:], [i], [], sorted_idx[:], [], [i])
        while j >= 0 and a[j] > key:
            yield SortStep(a[:], [j, j+1], [], sorted_idx[:], [], [j+1])
            a[j+1] = a[j]
            yield SortStep(a[:], [], [j, j+1], sorted_idx[:], [], [])
            j -= 1
        a[j+1] = key
        sorted_idx.append(i)
    yield SortStep(a[:], [], [], list(range(n)), [], [])


def cocktail_shaker_sort(arr: list[int]) -> Generator[SortStep, None, None]:
    a = arr[:]; n = len(a)
    sorted_idx: list[int] = []
    start, end = 0, n - 1
    swapped = True
    while swapped and start < end:
        swapped = False
        # forward pass
        for i in range(start, end):
            yield SortStep(a[:], [i, i+1], [], sorted_idx[:], [], [])
            if a[i] > a[i+1]:
                a[i], a[i+1] = a[i+1], a[i]
                swapped = True
                yield SortStep(a[:], [], [i, i+1], sorted_idx[:], [], [])
        sorted_idx.append(end)
        end -= 1
        if not swapped: break
        swapped = False
        # backward pass
        for i in range(end, start, -1):
            yield SortStep(a[:], [i-1, i], [], sorted_idx[:], [], [])
            if a[i-1] > a[i]:
                a[i-1], a[i] = a[i], a[i-1]
                swapped = True
                yield SortStep(a[:], [], [i-1, i], sorted_idx[:], [], [])
        sorted_idx.append(start)
        start += 1
    for i in range(n):
        if i not in sorted_idx: sorted_idx.append(i)
    yield SortStep(a[:], [], [], sorted_idx, [], [])


def gnome_sort(arr: list[int]) -> Generator[SortStep, None, None]:
    a = arr[:]; n = len(a); i = 0; sorted_idx: list[int] = []
    while i < n:
        if i == 0 or a[i] >= a[i - 1]:
            sorted_idx.append(i)
            yield SortStep(a[:], [], [], sorted_idx[:], [], [i])
            i += 1
        else:
            yield SortStep(a[:], [i-1, i], [], sorted_idx[:], [], [])
            a[i], a[i-1] = a[i-1], a[i]
            yield SortStep(a[:], [], [i-1, i], sorted_idx[:], [], [])
            i -= 1
    yield SortStep(a[:], [], [], list(range(n)), [], [])


def odd_even_sort(arr: list[int]) -> Generator[SortStep, None, None]:
    a = arr[:]; n = len(a); sorted_flag = False
    while not sorted_flag:
        sorted_flag = True
        # odd phase
        for i in range(1, n - 1, 2):
            yield SortStep(a[:], [i, i+1], [], [], [], [i])
            if a[i] > a[i+1]:
                a[i], a[i+1] = a[i+1], a[i]
                sorted_flag = False
                yield SortStep(a[:], [], [i, i+1], [], [], [])
        # even phase
        for i in range(0, n - 1, 2):
            yield SortStep(a[:], [i, i+1], [], [], [], [i])
            if a[i] > a[i+1]:
                a[i], a[i+1] = a[i+1], a[i]
                sorted_flag = False
                yield SortStep(a[:], [], [i, i+1], [], [], [])
    yield SortStep(a[:], [], [], list(range(n)), [], [])

# -- Efficient Sorts ----------------------------------------------------

def shell_sort(arr: list[int]) -> Generator[SortStep, None, None]:
    a = arr[:]; n = len(a)
    # Ciura gap sequence (2001)
    gaps = [701, 301, 132, 57, 23, 10, 4, 1]
    for gap in gaps:
        if gap > n: continue
        for i in range(gap, n):
            temp = a[i]; j = i
            yield SortStep(a[:], [j, j-gap], [], [], [], [i])
            while j >= gap and a[j - gap] > temp:
                yield SortStep(a[:], [j, j-gap], [], [], [], [])
                a[j] = a[j - gap]
                yield SortStep(a[:], [], [j, j-gap], [], [], [])
                j -= gap
            a[j] = temp
    yield SortStep(a[:], [], [], list(range(n)), [], [])


def comb_sort(arr: list[int]) -> Generator[SortStep, None, None]:
    a = arr[:]; n = len(a)
    gap = n; shrink = 1.3; sorted_flag = False
    while not sorted_flag:
        gap = max(1, int(gap / shrink))
        if gap <= 1: sorted_flag = True
        i = 0
        while i + gap < n:
            yield SortStep(a[:], [i, i+gap], [], [], [], [i])
            if a[i] > a[i+gap]:
                a[i], a[i+gap] = a[i+gap], a[i]
                sorted_flag = False
                yield SortStep(a[:], [], [i, i+gap], [], [], [])
            i += 1
    yield SortStep(a[:], [], [], list(range(n)), [], [])


def merge_sort(arr: list[int]) -> Generator[SortStep, None, None]:
    a = arr[:]; n = len(a)

    def _merge(l, m, r):
        left = a[l:m+1]; right = a[m+1:r+1]
        i = j = 0; k = l
        while i < len(left) and j < len(right):
            yield SortStep(a[:], [l+i, m+1+j], [], [], [], [k])
            if left[i] <= right[j]:
                a[k] = left[i]; i += 1
            else:
                a[k] = right[j]; j += 1
            yield SortStep(a[:], [], [k], [], [], [])
            k += 1
        while i < len(left):
            a[k] = left[i]; i += 1
            yield SortStep(a[:], [], [k], [], [], [])
            k += 1
        while j < len(right):
            a[k] = right[j]; j += 1
            yield SortStep(a[:], [], [k], [], [], [])
            k += 1

    def _sort(l, r):
        if l >= r: return
        m = (l + r) // 2
        yield from _sort(l, m)
        yield from _sort(m+1, r)
        yield from _merge(l, m, r)

    yield from _sort(0, n - 1)
    yield SortStep(a[:], [], [], list(range(n)), [], [])


def quick_sort(arr: list[int]) -> Generator[SortStep, None, None]:
    a = arr[:]; n = len(a); done: set[int] = set()

    def _partition(lo, hi):
        pivot = a[hi]; i = lo - 1
        yield SortStep(a[:], [], [], list(done), [hi], [])
        for j in range(lo, hi):
            yield SortStep(a[:], [j, hi], [], list(done), [hi], [])
            if a[j] < pivot:
                i += 1
                if i != j:
                    a[i], a[j] = a[j], a[i]
                    yield SortStep(a[:], [], [i, j], list(done), [hi], [])
        a[i+1], a[hi] = a[hi], a[i+1]
        yield SortStep(a[:], [], [i+1, hi], list(done), [], [])
        return i + 1

    def _qs(lo, hi):
        if lo < hi:
            pi = yield from _partition(lo, hi)
            done.add(pi)
            yield SortStep(a[:], [], [], list(done), [], [])
            yield from _qs(lo, pi - 1)
            yield from _qs(pi + 1, hi)
        elif lo == hi:
            done.add(lo)

    yield from _qs(0, n - 1)
    yield SortStep(a[:], [], [], list(range(n)), [], [])


def heap_sort(arr: list[int]) -> Generator[SortStep, None, None]:
    a = arr[:]; n = len(a); sorted_idx: list[int] = []

    def _heapify(size, root):
        largest = root; l = 2*root+1; r = 2*root+2
        if l < size:
            yield SortStep(a[:], [l, largest], [], sorted_idx[:], [], [])
            if a[l] > a[largest]: largest = l
        if r < size:
            yield SortStep(a[:], [r, largest], [], sorted_idx[:], [], [])
            if a[r] > a[largest]: largest = r
        if largest != root:
            a[root], a[largest] = a[largest], a[root]
            yield SortStep(a[:], [], [root, largest], sorted_idx[:], [], [])
            yield from _heapify(size, largest)

    for i in range(n // 2 - 1, -1, -1):
        yield from _heapify(n, i)
    for i in range(n - 1, 0, -1):
        a[0], a[i] = a[i], a[0]
        sorted_idx.append(i)
        yield SortStep(a[:], [], [0, i], sorted_idx[:], [], [])
        yield from _heapify(i, 0)
    sorted_idx.append(0)
    yield SortStep(a[:], [], [], sorted_idx, [], [])


def cycle_sort(arr: list[int]) -> Generator[SortStep, None, None]:
    a = arr[:]; n = len(a)
    for cycle_start in range(n - 1):
        item = a[cycle_start]
        pos = cycle_start
        for i in range(cycle_start + 1, n):
            yield SortStep(a[:], [pos, i], [], [], [], [cycle_start])
            if a[i] < item: pos += 1
        if pos == cycle_start: continue
        while item == a[pos]:
            pos += 1
        a[pos], item = item, a[pos]
        yield SortStep(a[:], [], [pos, cycle_start], [], [], [])
        while pos != cycle_start:
            pos = cycle_start
            for i in range(cycle_start + 1, n):
                yield SortStep(a[:], [pos, i], [], [], [], [cycle_start])
                if a[i] < item: pos += 1
            while item == a[pos]: pos += 1
            a[pos], item = item, a[pos]
            yield SortStep(a[:], [], [pos, cycle_start], [], [], [])
    yield SortStep(a[:], [], [], list(range(n)), [], [])

# -- Distribution Sorts -------------------------------------------------

def counting_sort(arr: list[int]) -> Generator[SortStep, None, None]:
    a = arr[:]; n = len(a)
    if n == 0:
        yield SortStep(a, [], [], [], [], []); return
    lo, hi = min(a), max(a)
    k = hi - lo + 1
    count = [0] * k
    for i, v in enumerate(a):
        count[v - lo] += 1
        yield SortStep(a[:], [i], [], [], [], [i])
    idx = 0
    for val in range(lo, hi + 1):
        for _ in range(count[val - lo]):
            a[idx] = val
            yield SortStep(a[:], [], [idx], [], [], [])
            idx += 1
    yield SortStep(a[:], [], [], list(range(n)), [], [])


def radix_sort(arr: list[int]) -> Generator[SortStep, None, None]:
    a = arr[:]; n = len(a)
    if n == 0:
        yield SortStep(a, [], [], [], [], []); return
    max_val = max(a)
    exp = 1
    sorted_idx: list[int] = []
    while max_val // exp > 0:
        output = [0] * n
        count = [0] * 10
        for i in range(n):
            digit = (a[i] // exp) % 10
            count[digit] += 1
        for i in range(1, 10): count[i] += count[i-1]
        for i in range(n - 1, -1, -1):
            digit = (a[i] // exp) % 10
            count[digit] -= 1
            output[count[digit]] = a[i]
        for i in range(n):
            a[i] = output[i]
            yield SortStep(a[:], [], [i], sorted_idx[:], [], [i])
        exp *= 10
    yield SortStep(a[:], [], [], list(range(n)), [], [])

# -- Novelty Sort -------------------------------------------------------

def bogo_sort(arr: list[int]) -> Generator[SortStep, None, None]:
    a = arr[:]
    while not _is_sorted(a):
        random.shuffle(a)
        yield SortStep(a[:], [], [], [], [], [])
    yield SortStep(a[:], [], [], list(range(len(a))), [], [])


GENERATORS: dict[str, callable] = {
    "bubble":    bubble_sort,
    "selection": selection_sort,
    "insertion": insertion_sort,
    "cocktail":  cocktail_shaker_sort,
    "gnome":     gnome_sort,
    "odd_even":  odd_even_sort,
    "shell":     shell_sort,
    "comb":      comb_sort,
    "merge":     merge_sort,
    "quick":     quick_sort,
    "heap":      heap_sort,
    "cycle":     cycle_sort,
    "counting":  counting_sort,
    "radix":     radix_sort,
    "bogo":      bogo_sort,
}

# Category order for display
CATEGORIES = ["Comparison", "Efficient", "Distribution", "Novelty"]

def _algo_category(key: str) -> str:
    info = ALGORITHMS.get(key)
    if not info: return "Comparison"
    return info.category


# ---------------------------------------------------------------------------
# Color utilities for value-based gradients
# ---------------------------------------------------------------------------

def _hsv_to_qcolor(h: float, s: float, v: float) -> QColor:
    """h in [0,360), s/v in [0,1]. Returns QColor."""
    c = QColor.fromHsvF(h / 360.0, s, v)
    return c


def _value_color(value: int, max_value: int) -> QColor:
    """Map a bar value to a vivid rainbow gradient color.
    Low values -> blue/purple, mid -> green/cyan, high -> orange/red."""
    t = value / max(max_value, 1)
    # Hue sweeps from 240 (blue) through 180 (cyan), 120 (green),
    # 60 (yellow), 0/360 (red)
    hue = (1.0 - t) * 240.0
    return _hsv_to_qcolor(hue, 0.75, 0.85)


def _lerp_color(c1: QColor, c2: QColor, t: float) -> QColor:
    """Linearly interpolate between two QColors."""
    return QColor(
        int(c1.red()   + (c2.red()   - c1.red())   * t),
        int(c1.green() + (c2.green() - c1.green()) * t),
        int(c1.blue()  + (c2.blue()  - c1.blue())  * t),
        int(c1.alpha() + (c2.alpha() - c1.alpha()) * t),
    )


# ---------------------------------------------------------------------------
# Custom painted bar chart widget  (heavily enhanced)
# ---------------------------------------------------------------------------

class BarChartWidget(QWidget):
    """Renders the array as colored bars with smooth animation,
    value-based rainbow gradient, comparison arcs, glow effects,
    bottom reflections, grid lines, and completion wave.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.array: list[int] = []
        self.bar_states: list[BarState] = []
        self.comparing_indices: list[int] = []
        self.swapping_indices: list[int] = []
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

        # Smooth interpolation state
        self._target_heights: list[float] = []
        self._display_heights: list[float] = []
        self._lerp_speed = 0.35  # 0=instant, 1=never moves

        # Completion wave
        self._wave_active = False
        self._wave_pos = -1  # -1 = inactive
        self._wave_timer = QTimer(self)
        self._wave_timer.setInterval(18)
        self._wave_timer.timeout.connect(self._wave_tick)

        # 60fps render timer for smooth interpolation
        self._render_timer = QTimer(self)
        self._render_timer.setInterval(16)  # ~60fps
        self._render_timer.timeout.connect(self._interpolate_tick)
        self._render_timer.start()

    def set_data(self, array: list[int], states: list[BarState],
                 comparing: list[int] = [],
                 swapping: list[int] = []):
        self.array = array
        self.bar_states = states
        self.comparing_indices = comparing
        self.swapping_indices = swapping

        # Rebuild target heights if array length changed
        n = len(array)
        max_val = max(array) if array else 1
        while len(self._target_heights) < n:
            self._target_heights.append(0.0)
            self._display_heights.append(0.0)
        self._target_heights = self._target_heights[:n]
        self._display_heights = self._display_heights[:n]

        for i, val in enumerate(array):
            self._target_heights[i] = val / max_val

        self.update()

    def start_completion_wave(self):
        """Trigger a rainbow wave sweep from left to right."""
        self._wave_active = True
        self._wave_pos = -1
        self._wave_timer.start()

    # -- smooth animation -------------------------------------------------

    def _interpolate_tick(self):
        moved = False
        for i in range(len(self._display_heights)):
            diff = self._target_heights[i] - self._display_heights[i]
            if abs(diff) > 0.001:
                self._display_heights[i] += diff * (1.0 - self._lerp_speed)
                moved = True
            else:
                self._display_heights[i] = self._target_heights[i]
        if moved:
            self.update()

    def _wave_tick(self):
        if not self._wave_active:
            self._wave_timer.stop()
            return
        self._wave_pos += 1
        if self._wave_pos >= len(self.array):
            self._wave_active = False
            self._wave_timer.stop()
            # Set all to sorted
            self.bar_states = [BarState.SORTED] * len(self.array)
        self.update()

    # -- layout helpers ---------------------------------------------------

    def _layout(self):
        """Return (w, h, n, gap, bar_w, top_pad, bot_pad, draw_h, baseline_y, max_val)."""
        w = self.width(); h = self.height()
        n = len(self.array)
        max_val = max(self.array) if self.array else 1
        gap = max(1, min(3, w // n // 6))
        bar_w = max(2, (w - gap * (n + 1)) / n)
        top_pad = 28          # room for value labels
        reflection_h = 30    # reflection area
        bot_pad = 10
        draw_h = h - top_pad - bot_pad - reflection_h
        baseline_y = top_pad + draw_h
        return w, h, n, gap, bar_w, top_pad, bot_pad, draw_h, baseline_y, max_val

    def _bar_rect(self, i, layout):
        """Return (x, y, bar_w, bar_h) for bar i using display heights."""
        w, h, n, gap, bar_w, top_pad, bot_pad, draw_h, baseline_y, max_val = layout
        frac = self._display_heights[i] if i < len(self._display_heights) else 0.0
        bar_h = max(0.0, frac * draw_h)
        x = gap + i * (bar_w + gap)
        y = baseline_y - bar_h
        return x, y, bar_w, bar_h

    # -- painting ---------------------------------------------------------

    def paintEvent(self, event):                          # noqa: N802
        if not self.array:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        layout = self._layout()
        w, h, n, gap, bar_w, top_pad, bot_pad, draw_h, baseline_y, max_val = layout

        self._draw_grid(p, layout)
        self._draw_bars(p, layout)
        self._draw_arcs(p, layout)
        self._draw_reflections(p, layout)

        p.end()

    def _draw_grid(self, p: QPainter, layout):
        """Draw subtle horizontal grid lines and axis labels."""
        w, h, n, gap, bar_w, top_pad, bot_pad, draw_h, baseline_y, max_val = layout
        grid_color = QColor(255, 255, 255, 18)
        label_color = QColor(255, 255, 255, 50)
        font = QFont("Consolas", 7)
        p.setFont(font)
        p.setPen(QPen(grid_color, 1, Qt.PenStyle.DotLine))
        num_lines = 5
        for i in range(1, num_lines + 1):
            y = baseline_y - (draw_h * i / num_lines)
            p.drawLine(int(gap), int(y), int(w - gap), int(y))
            val = int(max_val * i / num_lines)
            p.setPen(QPen(label_color))
            p.drawText(int(gap + 2), int(y - 2), str(val))
            p.setPen(QPen(grid_color, 1, Qt.PenStyle.DotLine))
        # baseline
        p.setPen(QPen(QColor(255, 255, 255, 40), 1))
        p.drawLine(int(gap), int(baseline_y), int(w - gap), int(baseline_y))

    def _bar_color(self, i: int, layout) -> QColor:
        """Determine the color for bar i, considering state and wave."""
        state = self.bar_states[i] if i < len(self.bar_states) else BarState.DEFAULT
        w, h, n, gap, bar_w, top_pad, bot_pad, draw_h, baseline_y, max_val = layout

        # During completion wave, color bars by wave position
        if self._wave_active:
            wave_width = max(4, n // 6)
            dist = i - self._wave_pos
            if dist < 0:
                # Already passed: sorted green
                return QColor(34, 197, 94)
            elif dist <= wave_width:
                # In the wave: rainbow!
                t = 1.0 - (dist / wave_width)
                hue = t * 300  # sweep through rainbow
                return _hsv_to_qcolor(hue, 0.9, 1.0)
            else:
                # Not yet reached: value-based color
                return _value_color(self.array[i], max_val)

        # State-based colors with value-based rainbow for DEFAULT
        if state == BarState.DEFAULT:
            return _value_color(self.array[i], max_val)
        return COLORS[state]

    def _draw_bars(self, p: QPainter, layout):
        """Draw all bars with gradients, glow, and value labels."""
        w, h, n, gap, bar_w, top_pad, bot_pad, draw_h, baseline_y, max_val = layout
        show_labels = bar_w >= 18 and n <= 50

        # Collect glow bars to draw glows on a separate pass (behind bars)
        glow_bars = []

        for i in range(n):
            x, y, bw, bh = self._bar_rect(i, layout)
            if bh < 1:
                continue
            color = self._bar_color(i, layout)
            state = self.bar_states[i] if i < len(self.bar_states) else BarState.DEFAULT

            # Rounded-top bar path
            path = QPainterPath()
            radius = min(bw / 2, 5)
            if bh < radius * 2:
                radius = bh / 2
            path.moveTo(x, baseline_y)
            path.lineTo(x, y + radius)
            path.quadTo(x, y, x + radius, y)
            path.lineTo(x + bw - radius, y)
            path.quadTo(x + bw, y, x + bw, y + radius)
            path.lineTo(x + bw, baseline_y)
            path.closeSubpath()

            # Gradient fill: vertical gradient for 3D depth
            grad = QLinearGradient(x, y, x + bw, baseline_y)
            grad.setColorAt(0.0, color.lighter(140))
            grad.setColorAt(0.3, color.lighter(115))
            grad.setColorAt(1.0, color.darker(120))
            p.setBrush(QBrush(grad))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPath(path)

            # Thin bright edge on the left side for a 3D bevel
            if bw > 4:
                edge_grad = QLinearGradient(x, y, x + bw * 0.3, y)
                edge_color = QColor(255, 255, 255, 35)
                edge_grad.setColorAt(0.0, edge_color)
                edge_color2 = QColor(255, 255, 255, 0)
                edge_grad.setColorAt(1.0, edge_color2)
                p.setBrush(QBrush(edge_grad))
                p.setPen(Qt.PenStyle.NoPen)
                bevel_path = QPainterPath()
                bevel_path.moveTo(x, baseline_y)
                bevel_path.lineTo(x, y + radius)
                bevel_path.quadTo(x, y, x + radius, y)
                bevel_path.lineTo(x + bw * 0.3, y)
                bevel_path.lineTo(x + bw * 0.3, baseline_y)
                bevel_path.closeSubpath()
                p.drawPath(bevel_path)

            # Collect active bars for glow pass
            if state in (BarState.COMPARING, BarState.SWAPPING,
                         BarState.PIVOT, BarState.ACTIVE):
                glow_bars.append((x, y, bw, bh, color))

            # Value label
            if show_labels and self.array[i] > 0:
                p.setPen(QPen(QColor(255, 255, 255, 200)))
                label_font = QFont("Consolas", max(7, min(9, int(bw * 0.45))))
                p.setFont(label_font)
                label = str(self.array[i])
                fm = p.fontMetrics()
                lw = fm.horizontalAdvance(label)
                lx = x + (bw - lw) / 2
                ly = y - 4
                p.drawText(int(lx), int(ly), label)

        # Glow pass (drawn on top of all bars)
        for x, y, bw, bh, color in glow_bars:
            self._draw_glow(p, x, y, bw, bh, color)

    def _draw_glow(self, p: QPainter, x: float, y: float,
                   bw: float, bh: float, color: QColor):
        """Draw a soft, large glow behind an active bar."""
        cx = x + bw / 2
        cy = y + bh / 2
        glow_radius = bw * 3.0

        # Outer glow
        glow = QRadialGradient(QPointF(cx, cy), glow_radius)
        gc = QColor(color); gc.setAlpha(45)
        glow.setColorAt(0.0, gc)
        gc2 = QColor(color); gc2.setAlpha(0)
        glow.setColorAt(0.4, gc2)
        p.setBrush(QBrush(glow))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QRectF(cx - glow_radius, cy - glow_radius,
                              glow_radius * 2, glow_radius * 2))

        # Inner bright core glow
        core = QRadialGradient(QPointF(cx, y), bw * 1.2)
        bright = QColor(color); bright.setAlpha(70)
        core.setColorAt(0.0, bright)
        bright2 = QColor(color); bright2.setAlpha(0)
        core.setColorAt(1.0, bright2)
        p.setBrush(QBrush(core))
        p.drawEllipse(QRectF(cx - bw * 1.2, y - bw * 1.2,
                              bw * 2.4, bw * 2.4))

    def _draw_arcs(self, p: QPainter, layout):
        """Draw curved arcs connecting compared/swapping bars."""
        w, h, n, gap, bar_w, top_pad, bot_pad, draw_h, baseline_y, max_val = layout

        # Draw comparison arcs
        if len(self.comparing_indices) >= 2:
            self._draw_connection_arc(
                p, self.comparing_indices, layout,
                QColor(245, 158, 11, 120),  # amber, semi-transparent
                QColor(245, 158, 11, 40))

        # Draw swap arcs (red, on top)
        if len(self.swapping_indices) >= 2:
            self._draw_connection_arc(
                p, self.swapping_indices, layout,
                QColor(239, 68, 68, 160),
                QColor(239, 68, 68, 50),
                line_width=2.5)

    def _draw_connection_arc(self, p: QPainter, indices: list[int],
                              layout, color: QColor,
                              fill_color: QColor, line_width: float = 2.0):
        """Draw a curved arc between two bar tops."""
        w, h, n, gap, bar_w, top_pad, bot_pad, draw_h, baseline_y, max_val = layout
        if len(indices) < 2:
            return
        # Use first and last index in the list
        i1, i2 = indices[0], indices[-1]
        if i1 == i2 or i1 >= n or i2 >= n:
            return

        x1, y1, bw1, bh1 = self._bar_rect(i1, layout)
        x2, y2, bw2, bh2 = self._bar_rect(i2, layout)

        cx1 = x1 + bw1 / 2
        cx2 = x2 + bw2 / 2

        # Arc control point: above the bars
        mid_x = (cx1 + cx2) / 2
        arc_height = min(draw_h * 0.3, 80)
        ctrl_y = min(y1, y2) - arc_height

        # Draw filled area under arc (very subtle)
        fill_path = QPainterPath()
        fill_path.moveTo(cx1, y1)
        fill_path.quadTo(mid_x, ctrl_y, cx2, y2)
        fill_path.lineTo(cx2, baseline_y)
        fill_path.lineTo(cx1, baseline_y)
        fill_path.closeSubpath()
        p.setBrush(QBrush(fill_color))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPath(fill_path)

        # Draw the arc line
        arc_path = QPainterPath()
        arc_path.moveTo(cx1, y1)
        arc_path.quadTo(mid_x, ctrl_y, cx2, y2)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(color, line_width))
        p.drawPath(arc_path)

        # Draw small circles at endpoints
        for cx, cy, bw in [(cx1, y1, bw1), (cx2, y2, bw2)]:
            dot_r = max(3, bw * 0.25)
            p.setBrush(QBrush(color))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(cx, cy), dot_r, dot_r)

    def _draw_reflections(self, p: QPainter, layout):
        """Draw a subtle mirrored reflection below the baseline."""
        w, h, n, gap, bar_w, top_pad, bot_pad, draw_h, baseline_y, max_val = layout
        reflection_h = 28

        # Clip to reflection area
        p.setClipRect(QRectF(0, baseline_y, w, reflection_h + 5))

        for i in range(n):
            x, y, bw, bh = self._bar_rect(i, layout)
            if bh < 1:
                continue
            color = self._bar_color(i, layout)

            # Reflected bar (below baseline, faded)
            ref_h = min(bh * 0.35, reflection_h)
            ref_y = baseline_y

            grad = QLinearGradient(x, ref_y, x, ref_y + ref_h)
            rc = QColor(color); rc.setAlpha(50)
            grad.setColorAt(0.0, rc)
            rc2 = QColor(color); rc2.setAlpha(0)
            grad.setColorAt(1.0, rc2)
            p.setBrush(QBrush(grad))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRect(QRectF(x, ref_y, bw, ref_h))

        p.setClipping(False)

# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class SortingVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Sorting Algorithm Visualizer  -  15 Algorithms + Sound")
        self.resize(1280, 780)
        self.setMinimumSize(800, 550)

        self.array: list[int] = []
        self.bar_states: list[BarState] = []
        self.generator: Optional[Generator] = None
        self.comparisons = 0
        self.swaps = 0
        self.is_sorting = False
        self.is_paused = False
        self.is_complete = False

        self._build_ui()
        self._apply_style()
        self._populate_algo_combo()
        self._generate_array()

        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._tick)

    # ---------------------------------------------------------------- UI

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # -- Header --
        hdr = QHBoxLayout()
        title = QLabel("Sorting Algorithm Visualizer")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet(f"color:{FG_PRIMARY.name()};border:none;")
        hdr.addWidget(title)
        hdr.addStretch()

        # Sound controls in header
        self.sound_cb = QCheckBox(" Sound")
        self.sound_cb.setChecked(True)
        self.sound_cb.setFont(QFont("Segoe UI", 10))
        self.sound_cb.setStyleSheet(
            f"QCheckBox{{color:{SOUND_COLOR.name()};border:none;"
            f"spacing:6px;font-weight:bold;}}"
            f"QCheckBox::indicator{{width:18px;height:18px;border-radius:4px;"
            f"border:2px solid {SOUND_COLOR.name()};"
            f"background:{BG_WIDGET.name()};}}"
            f"QCheckBox::indicator:checked{{background:{SOUND_COLOR.name()};"
            f"border-color:{SOUND_COLOR.name()};}}")
        self.sound_cb.toggled.connect(self._on_sound_toggle)
        hdr.addWidget(self.sound_cb)

        vol_lbl = QLabel("Vol:")
        vol_lbl.setFont(QFont("Segoe UI", 9))
        vol_lbl.setStyleSheet(f"color:{FG_MUTED.name()};border:none;")
        hdr.addWidget(vol_lbl)

        self.vol_sl = QSlider(Qt.Orientation.Horizontal)
        self.vol_sl.setRange(0, 100); self.vol_sl.setValue(35)
        self.vol_sl.setFixedWidth(100)
        self.vol_sl.valueChanged.connect(self._on_volume)
        hdr.addWidget(self.vol_sl)

        self.vol_val = QLabel("35%")
        self.vol_val.setFont(QFont("Segoe UI", 9))
        self.vol_val.setFixedWidth(35)
        self.vol_val.setStyleSheet(f"color:{FG_MUTED.name()};border:none;")
        hdr.addWidget(self.vol_val)

        hdr.addStretch()
        self.badge = QLabel("Ready")
        self.badge.setFont(QFont("Segoe UI", 10))
        self._style_badge(FG_SECONDARY)
        hdr.addWidget(self.badge)
        root.addLayout(hdr)

        # -- Controls card --
        ctrl = QGroupBox("Controls")
        ctrl.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        ctrl.setStyleSheet(
            f"QGroupBox{{color:{FG_PRIMARY.name()};"
            f"border:1px solid {BORDER.name()};"
            f"border-radius:10px;margin-top:12px;padding-top:20px;"
            f"background:{BG_CARD.name()};}}"
            f"QGroupBox::title{{subcontrol-origin:margin;"
            f"left:14px;padding:0 6px;}}")
        gl = QGridLayout(); gl.setSpacing(12)

        gl.addWidget(self._lbl("Algorithm:"), 0, 0)
        self.algo_box = QComboBox()
        self.algo_box.setFixedWidth(260)
        self.algo_box.currentIndexChanged.connect(self._on_algo)
        gl.addWidget(self.algo_box, 0, 1)

        gl.addWidget(self._lbl("Array Size:"), 0, 2)
        self.size_sl = QSlider(Qt.Orientation.Horizontal)
        self.size_sl.setRange(5, 150); self.size_sl.setValue(50)
        self.size_sl.setFixedWidth(180)
        self.size_sl.valueChanged.connect(self._on_size)
        gl.addWidget(self.size_sl, 0, 3)
        self.size_val = QLabel("50"); self.size_val.setFixedWidth(30)
        gl.addWidget(self.size_val, 0, 4)

        gl.addWidget(self._lbl("Speed:"), 1, 0)
        self.speed_sl = QSlider(Qt.Orientation.Horizontal)
        self.speed_sl.setRange(1, 100); self.speed_sl.setValue(50)
        self.speed_sl.setFixedWidth(180)
        self.speed_sl.valueChanged.connect(self._on_speed)
        gl.addWidget(self.speed_sl, 1, 3)
        self.speed_val = QLabel("50%"); self.speed_val.setFixedWidth(40)
        gl.addWidget(self.speed_val, 1, 4)

        brow = QHBoxLayout(); brow.setSpacing(8)
        self.b_sort  = self._btn("  Sort",   GREEN_BTN,  self._on_sort)
        self.b_pause = self._btn("  Pause",  YELLOW_BTN, self._on_pause)
        self.b_pause.setEnabled(False)
        self.b_step  = self._btn("  Step",   ACCENT,     self._on_step)
        self.b_stop  = self._btn("  Stop",   RED_BTN,    self._on_stop)
        self.b_stop.setEnabled(False)
        self.b_new   = self._btn("  New",    BG_WIDGET,  self._generate_array)
        for b in (self.b_sort, self.b_pause, self.b_step,
                  self.b_stop, self.b_new):
            brow.addWidget(b)
        gl.addLayout(brow, 2, 0, 1, 5)
        ctrl.setLayout(gl)
        root.addWidget(ctrl)

        # -- Splitter: chart | info --
        sp = QSplitter(Qt.Orientation.Horizontal)

        chart_grp = self._card_group("Visualization")
        cl = QVBoxLayout(chart_grp)
        cl.setContentsMargins(12, 24, 12, 12)
        self.chart = BarChartWidget()
        cl.addWidget(self.chart)

        # Legend
        leg = QHBoxLayout(); leg.setSpacing(14)
        for text, st in [("Value", BarState.DEFAULT),
                         ("Comparing", BarState.COMPARING),
                         ("Swapping", BarState.SWAPPING),
                         ("Pivot", BarState.PIVOT),
                         ("Active", BarState.ACTIVE),
                         ("Sorted", BarState.SORTED)]:
            sw = QFrame(); sw.setFixedSize(14, 14)
            if st == BarState.DEFAULT:
                # Show a mini rainbow gradient for value-based color
                sw.setStyleSheet(
                    "background:qlineargradient("
                    "x1:0,y1:0,x2:1,y2:0,"
                    "stop:0 #5555ff,stop:0.25 #55cccc,"
                    "stop:0.5 #44cc44,stop:0.75 #cccc22,"
                    "stop:1 #ff4444);"
                    "border-radius:3px;border:none;")
            else:
                sw.setStyleSheet(
                    f"background:{COLORS[st].name()};"
                    f"border-radius:3px;border:none;")
            lb = QLabel(text)
            lb.setFont(QFont("Segoe UI", 9))
            lb.setStyleSheet(
                f"color:{FG_MUTED.name()};border:none;")
            leg.addWidget(sw); leg.addWidget(lb)
        leg.addStretch(); cl.addLayout(leg)
        sp.addWidget(chart_grp)

        info_grp = self._card_group("Algorithm Info")
        info_grp.setMinimumWidth(270); info_grp.setMaximumWidth(360)
        il = QVBoxLayout(info_grp)
        il.setContentsMargins(16, 24, 16, 16); il.setSpacing(10)

        self.i_name = QLabel()
        self.i_name.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.i_name.setStyleSheet(
            f"color:{FG_PRIMARY.name()};border:none;")
        il.addWidget(self.i_name)

        self.i_cat = QLabel()
        self.i_cat.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        self.i_cat.setStyleSheet(
            f"color:{ACCENT.name()};border:none;"
            f"border:1px solid {ACCENT.name()};"
            f"border-radius:4px;padding:2px 8px;")
        il.addWidget(self.i_cat, alignment=Qt.AlignmentFlag.AlignLeft)

        self.i_desc = QLabel(); self.i_desc.setWordWrap(True)
        self.i_desc.setFont(QFont("Segoe UI", 9))
        self.i_desc.setStyleSheet(
            f"color:{FG_SECONDARY.name()};border:none;")
        il.addWidget(self.i_desc)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color:{BORDER.name()};")
        il.addWidget(sep)

        self.info_rows: list[tuple[QLabel, QLabel]] = []
        for _ in range(5):
            lk = QLabel(); lk.setFont(QFont("Segoe UI", 9))
            lk.setStyleSheet(
                f"color:{FG_MUTED.name()};border:none;")
            lv = QLabel()
            lv.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
            lv.setStyleSheet(
                f"color:{FG_PRIMARY.name()};border:none;")
            r = QHBoxLayout(); r.addWidget(lk); r.addStretch()
            r.addWidget(lv)
            il.addLayout(r); self.info_rows.append((lk, lv))

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet(f"color:{BORDER.name()};")
        il.addWidget(sep2)

        lt = QLabel("Live Stats")
        lt.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        lt.setStyleSheet(
            f"color:{FG_PRIMARY.name()};border:none;")
        il.addWidget(lt)
        self.s_comp = QLabel("Comparisons:  0")
        self.s_swap = QLabel("Swaps / Writes:  0")
        self.s_size = QLabel("Array Size:  0")
        self.s_algo_count = QLabel("Algorithms:  15")
        for s in (self.s_comp, self.s_swap, self.s_size,
                  self.s_algo_count):
            s.setFont(QFont("Consolas", 9))
            s.setStyleSheet(
                f"color:{FG_SECONDARY.name()};border:none;")
            il.addWidget(s)
        il.addStretch()
        sp.addWidget(info_grp)
        sp.setStretchFactor(0, 3); sp.setStretchFactor(1, 1)
        sp.setHandleWidth(2)
        root.addWidget(sp, stretch=1)

        self.statusBar().showMessage(
            "Ready  |  15 algorithms  |  Toggle sound in the top bar  |  "
            "Try Bogo Sort for fun!")
        self.statusBar().setStyleSheet(
            f"color:{FG_MUTED.name()};background:{BG_CARD.name()};"
            f"border-top:1px solid {BORDER.name()};font-size:11px;")

        self._refresh_info()

    # helpers
    @staticmethod
    def _lbl(t) -> QLabel:
        l = QLabel(t); l.setFont(QFont("Segoe UI", 9))
        l.setStyleSheet(
            f"color:{FG_SECONDARY.name()};border:none;")
        return l

    @staticmethod
    def _card_group(title) -> QGroupBox:
        g = QGroupBox(title)
        g.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        g.setStyleSheet(
            f"QGroupBox{{color:{FG_PRIMARY.name()};"
            f"border:1px solid {BORDER.name()};"
            f"border-radius:10px;margin-top:12px;padding-top:20px;"
            f"background:{BG_CARD.name()};}}"
            f"QGroupBox::title{{subcontrol-origin:margin;"
            f"left:14px;padding:0 6px;}}")
        return g

    @staticmethod
    def _btn(text, color: QColor, slot) -> QPushButton:
        b = QPushButton(text)
        b.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setFixedHeight(36)
        r, g, bl = color.red(), color.green(), color.blue()
        fg = "#fff" if color.lightness() < 160 else "#000"
        b.setStyleSheet(
            f"QPushButton{{background:rgb({r},{g},{bl});color:{fg};"
            f"border:none;border-radius:8px;padding:0 18px;}}"
            f"QPushButton:hover{{"
            f"background:{color.lighter(115).name()};}}"
            f"QPushButton:pressed{{"
            f"background:{color.darker(110).name()};}}"
            f"QPushButton:disabled{{"
            f"background:{BG_WIDGET.name()};"
            f"color:{FG_MUTED.name()};}}")
        b.clicked.connect(slot)
        return b

    def _style_badge(self, color: QColor, text: str = ""):
        if text:
            self.badge.setText(text)
        r, g, b = color.red(), color.green(), color.blue()
        fg = "#fff" if color.lightness() < 160 else "#000"
        self.badge.setStyleSheet(
            f"color:{fg};background:rgb({r},{g},{b});"
            f"border:none;border-radius:6px;padding:4px 14px;")

    # -- Algorithm combo with categories --
    def _populate_algo_combo(self):
        self.algo_box.blockSignals(True)
        self.algo_box.clear()
        prev_cat = None
        for key in ALGORITHMS:
            cat = _algo_category(key)
            if cat != prev_cat:
                self.algo_box.addItem(f"--- {cat} Sorts ---")
                idx = self.algo_box.count() - 1
                model = self.algo_box.model()
                item = model.item(idx)
                item.setFlags(
                    item.flags()
                    & ~Qt.ItemFlag.ItemIsSelectable
                    & ~Qt.ItemFlag.ItemIsEnabled)
                item.setForeground(FG_MUTED)
                prev_cat = cat
            self.algo_box.addItem(
                ALGORITHMS[key].name, userData=key)
        self.algo_box.blockSignals(False)
        self.algo_box.setCurrentIndex(1)  # first real algorithm

    def _algo_key(self) -> str:
        data = self.algo_box.currentData()
        if data:
            return data
        name = self.algo_box.currentText()
        for k, v in ALGORITHMS.items():
            if v.name == name:
                return k
        return "bubble"

    # -------------------------------------------------------------- style

    def _apply_style(self):
        self.setStyleSheet(f"""
            QMainWindow,QWidget{{
                background:{BG_DARK.name()};color:{FG_PRIMARY.name()};
                font-family:"Segoe UI","Helvetica Neue",sans-serif;}}
            QComboBox{{
                background:{BG_WIDGET.name()};color:{FG_PRIMARY.name()};
                border:1px solid {BORDER.name()};border-radius:8px;
                padding:6px 12px;min-height:28px;}}
            QComboBox::drop-down{{border:none;width:28px;}}
            QComboBox QAbstractItemView{{
                background:{BG_WIDGET.name()};color:{FG_PRIMARY.name()};
                border:1px solid {BORDER.name()};
                selection-background-color:{ACCENT.name()};
                selection-color:white;}}
            QSlider::groove:horizontal{{
                height:6px;background:{BG_WIDGET.name()};
                border-radius:3px;}}
            QSlider::handle:horizontal{{
                background:{ACCENT.name()};width:18px;height:18px;
                margin:-6px 0;border-radius:9px;}}
            QSlider::handle:horizontal:hover{{
                background:{ACCENT_HOVER.name()};}}
            QSlider::sub-page:horizontal{{
                background:{ACCENT.name()};border-radius:3px;}}
            QSplitter::handle{{background:{BORDER.name()};}}
        """)

    # ----------------------------------------------------------- info panel

    def _refresh_info(self):
        a = ALGORITHMS[self._algo_key()]
        self.i_name.setText(a.name)
        self.i_cat.setText(f"  {a.category} Sort  ")
        self.i_desc.setText(a.description)
        for (lk, lv), lt, vt in zip(
            self.info_rows,
            ["Best Case:", "Average:", "Worst Case:",
             "Space:", "Stable:"],
            [a.time_best, a.time_avg, a.time_worst, a.space,
             "Yes" if a.stable else "No"]):
            lk.setText(lt); lv.setText(vt)

    def _refresh_stats(self):
        self.s_comp.setText(f"Comparisons:  {self.comparisons}")
        self.s_swap.setText(f"Swaps/Writes:  {self.swaps}")
        self.s_size.setText(f"Array Size:  {len(self.array)}")

    # ------------------------------------------------------- sound controls

    def _on_sound_toggle(self, checked):
        _audio.enabled = checked

    def _on_volume(self, v):
        _audio.volume = v / 100.0
        self.vol_val.setText(f"{v}%")

    # --------------------------------------------------------- array / sort

    def _generate_array(self):
        if self.is_sorting:
            return
        sz = self.size_sl.value()
        self.array = [random.randint(5, 99) for _ in range(sz)]
        self.bar_states = [BarState.DEFAULT] * sz
        self.comparisons = self.swaps = 0
        self.is_complete = False
        self.generator = None
        self.chart.set_data(self.array, self.bar_states,
                            comparing=[], swapping=[])
        self._refresh_stats()
        self._style_badge(FG_SECONDARY, "Ready")
        self.statusBar().showMessage(
            f"New array generated  |  {sz} elements")
        self._controls(True)

    def _on_sort(self):
        if self.is_sorting and not self.is_paused:
            return
        if self.is_paused:
            self.is_paused = False
            self.timer.start()
            self._controls(False, pause=True)
            self._style_badge(ACCENT, "Sorting...")
            self.statusBar().showMessage("Sorting resumed")
            _audio.start()
            return
        self._generate_array()
        key = self._algo_key()
        # Enforce max array size for bogo sort
        if key == "bogo" and len(self.array) > 10:
            self.array = self.array[:10]
            self.bar_states = [BarState.DEFAULT] * 10
            self.chart.set_data(self.array, self.bar_states,
                                comparing=[], swapping=[])
            self.statusBar().showMessage(
                "Bogo Sort limited to 10 elements (otherwise it "
                "would never finish!)")
        self.generator = GENERATORS[key](self.array)
        self.comparisons = self.swaps = 0
        self.is_sorting = True; self.is_complete = False
        self._controls(False, pause=True, stop=True)
        self._style_badge(ACCENT, "Sorting...")
        self.statusBar().showMessage(
            f"Running {ALGORITHMS[key].name}...")
        _audio.start()
        self.timer.start()

    def _on_pause(self):
        if not self.is_sorting or self.is_paused:
            return
        self.is_paused = True; self.timer.stop()
        self._controls(True, pause=False, stop=True)
        self._style_badge(YELLOW_BTN, "Paused")
        self.statusBar().showMessage(
            "Paused  |  click Sort to resume or Step to advance")

    def _on_step(self):
        if self.is_sorting and not self.is_paused:
            return
        if self.generator is None:
            key = self._algo_key()
            self.generator = GENERATORS[key](self.array)
            self.comparisons = self.swaps = 0
            self.is_sorting = True; self.is_complete = False
            self._controls(False, pause=True, stop=True)
            self._style_badge(ACCENT, "Stepping...")
            _audio.start()
        self._do_step()

    def _on_stop(self):
        self.timer.stop()
        self.is_sorting = self.is_paused = self.is_complete = False
        self.generator = None
        _audio.stop()
        self._generate_array()

    def _tick(self):
        self._do_step()

    def _do_step(self):
        if self.generator is None:
            return
        try:
            step = next(self.generator)
        except StopIteration:
            self._finish()
            return
        self.array = step.array
        self.bar_states = [self._state(i, step)
                           for i in range(len(step.array))]
        max_val = max(self.array) if self.array else 100
        if step.comparing:
            self.comparisons += 1
            vals = [self.array[i] for i in step.comparing
                    if i < len(self.array)]
            if vals:
                _audio.play_compare(
                    vals[0],
                    vals[-1] if len(vals) > 1 else vals[0],
                    max_val)
        if step.swapping:
            self.swaps += 1
            vals = [self.array[i] for i in step.swapping
                    if i < len(self.array)]
            if vals:
                _audio.play_swap(
                    vals[0],
                    vals[-1] if len(vals) > 1 else vals[0],
                    max_val)
        self.chart.set_data(self.array, self.bar_states,
                            comparing=step.comparing,
                            swapping=step.swapping)
        self._refresh_stats()

    @staticmethod
    def _state(i, s: SortStep) -> BarState:
        if i in s.swapping:  return BarState.SWAPPING
        if i in s.comparing: return BarState.COMPARING
        if i in s.pivot:     return BarState.PIVOT
        if i in s.active:    return BarState.ACTIVE
        if i in s.sorted:    return BarState.SORTED
        return BarState.DEFAULT

    def _finish(self):
        self.timer.stop()
        self.is_sorting = False
        self.is_complete = True
        self.generator = None
        self.bar_states = [BarState.SORTED] * len(self.array)
        # Clear comparing/swapping arcs
        self.chart.set_data(self.array, self.bar_states,
                            comparing=[], swapping=[])
        # Trigger the rainbow completion wave
        self.chart.start_completion_wave()
        self._refresh_stats(); self._controls(True)
        # Play a celebratory ascending arpeggio
        max_val = max(self.array) if self.array else 100
        n = min(len(self.array), 20)
        step = max(1, len(self.array) // n)
        for idx in range(0, len(self.array), step):
            _audio.play_tone(self.array[idx], max_val,
                             duration_ms=40)
        self._style_badge(GREEN_BTN, "Complete!")
        self.statusBar().showMessage(
            f"Done  |  {self.comparisons} comparisons, "
            f"{self.swaps} swaps/writes")
        _audio.stop()

    def _controls(self, on: bool, pause: bool = False,
                  stop: bool = False):
        self.algo_box.setEnabled(on)
        self.size_sl.setEnabled(on)
        self.b_sort.setEnabled(on)
        self.b_pause.setEnabled(pause)
        self.b_step.setEnabled(on or self.is_paused)
        self.b_stop.setEnabled(stop)
        self.b_new.setEnabled(on)

    # slider callbacks
    def _on_algo(self):
        if not self.is_sorting:
            self._refresh_info()
            self._generate_array()

    def _on_size(self, v):
        self.size_val.setText(str(v))
        if self._algo_key() == "bogo" and v > 10:
            self.size_sl.blockSignals(True)
            self.size_sl.setValue(10)
            self.size_val.setText("10")
            self.size_sl.blockSignals(False)
            v = 10
        if not self.is_sorting:
            self._generate_array()

    def _on_speed(self, v):
        self.speed_val.setText(f"{v}%")
        self.timer.setInterval(max(1, int(501 - v * 5)))

    def closeEvent(self, event):                         # noqa: N802
        _audio.stop()
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window,          BG_DARK)
    pal.setColor(QPalette.ColorRole.WindowText,      FG_PRIMARY)
    pal.setColor(QPalette.ColorRole.Base,            BG_WIDGET)
    pal.setColor(QPalette.ColorRole.AlternateBase,   BG_CARD)
    pal.setColor(QPalette.ColorRole.Text,            FG_PRIMARY)
    pal.setColor(QPalette.ColorRole.Button,          BG_WIDGET)
    pal.setColor(QPalette.ColorRole.ButtonText,      FG_PRIMARY)
    pal.setColor(QPalette.ColorRole.Highlight,       ACCENT)
    pal.setColor(QPalette.ColorRole.HighlightedText,
                 QColor(255, 255, 255))
    pal.setColor(QPalette.ColorRole.PlaceholderText, FG_MUTED)
    app.setPalette(pal)

    win = SortingVisualizer()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()