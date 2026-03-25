"""rubin Plot-Theme – Farbpalette und matplotlib-Konfiguration.

Wird von allen Plotting-Modulen importiert, um ein konsistentes
rubinrotes Erscheinungsbild zu gewährleisten.

Verwendung::

    from rubin.utils.plot_theme import apply_rubin_theme, RUBIN_COLORS
    apply_rubin_theme()  # einmal am Anfang aufrufen
"""

from __future__ import annotations

from typing import Dict

# ── Farbpalette ──
RUBIN_COLORS: Dict[str, str] = {
    "ruby":       "#9B111E",   # Hauptfarbe
    "ruby_dark":  "#6B0D15",   # Dunkel – Titel, Akzente
    "ruby_light": "#C4343F",   # Hell – sekundäre Linien
    "ruby_pale":  "#FDF2F3",   # Sehr hell – Hintergründe, Fills
    "gold":       "#D4A853",   # Kontrastfarbe – Vergleich, Warnung
    "gold_light": "#E8D49C",   # Gold hell – Fills
    "slate":      "#57606A",   # Grau – Referenzlinien, Text
    "slate_light":"#8B949E",   # Grau hell – Achsen, Gitter
    "bg":         "#FEFAFA",   # Hintergrund
    "text":       "#24292F",   # Text
    "grid":       "#EDE6E7",   # Gitterlinien
    "white":      "#FFFFFF",
}

# Sequentielle Palette für mehrere Modelle / Kategorien
# Rubin → Gold → gedämpfte Komplementärfarben
RUBIN_PALETTE = [
    "#9B111E",  # Ruby
    "#D4A853",  # Gold
    "#6B0D15",  # Ruby Dark
    "#C4343F",  # Ruby Light
    "#8B6914",  # Gold Dark
    "#E07A5F",  # Terracotta
    "#57606A",  # Slate
    "#A45A52",  # Muted Rose
]

# Für Binary-Vergleiche (Modell vs. Referenz)
COLOR_MODEL = RUBIN_COLORS["ruby"]
COLOR_MODEL_FILL = RUBIN_COLORS["ruby_pale"]
COLOR_REFERENCE = RUBIN_COLORS["gold"]
COLOR_REFERENCE_FILL = RUBIN_COLORS["gold_light"]
COLOR_BASELINE = RUBIN_COLORS["slate"]
COLOR_DIFFERENCE = RUBIN_COLORS["ruby_light"]
COLOR_HIGHLIGHT_BOX = RUBIN_COLORS["ruby_pale"]


def apply_rubin_theme() -> None:
    """Setzt matplotlib rcParams auf das rubin-Theme.

    Aufruf einmal am Anfang eines Skripts oder Pipeline-Laufs.
    Überschreibt nur visuelle Parameter, keine Backend-Einstellungen.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        # Farben
        "axes.prop_cycle": plt.cycler(color=RUBIN_PALETTE),
        "axes.facecolor": RUBIN_COLORS["white"],
        "axes.edgecolor": RUBIN_COLORS["slate_light"],
        "axes.labelcolor": RUBIN_COLORS["text"],
        "axes.titlecolor": RUBIN_COLORS["ruby_dark"],

        # Grid
        "axes.grid": True,
        "grid.color": RUBIN_COLORS["grid"],
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,

        # Figur
        "figure.facecolor": RUBIN_COLORS["bg"],
        "figure.edgecolor": "none",
        "figure.dpi": 120,

        # Text
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,

        # Ticks
        "xtick.color": RUBIN_COLORS["slate"],
        "ytick.color": RUBIN_COLORS["slate"],
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        # Legend
        "legend.framealpha": 0.9,
        "legend.edgecolor": RUBIN_COLORS["grid"],
        "legend.fontsize": 10,

        # Lines
        "lines.linewidth": 2.0,
        "lines.markersize": 6,

        # Savefig
        "savefig.facecolor": RUBIN_COLORS["white"],
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
    })


# ---------------------------------------------------------------------------
# Post-Processing: Figure-Level Farbmapping für externe Bibliotheken
# ---------------------------------------------------------------------------

# Standard-Farbpaletten externer Bibliotheken → rubin-Palette.
# Mapping-Regeln: (Quellfarbe, Zielfarbe). Die Quellfarben werden
# tolerant gematcht (Ähnlichkeit), damit leichte Farbvariationen
# (z.B. alpha-Blending) ebenfalls erkannt werden.
_DEFAULT_COLOR_MAP = {
    # matplotlib tab10 cycle
    "#1f77b4": RUBIN_COLORS["ruby"],        # tab:blue → Ruby
    "#ff7f0e": RUBIN_COLORS["gold"],        # tab:orange → Gold
    "#2ca02c": RUBIN_COLORS["ruby_dark"],   # tab:green → Ruby Dark
    "#d62728": RUBIN_COLORS["ruby_light"],  # tab:red → Ruby Light
    "#9467bd": RUBIN_COLORS["ruby_light"],  # tab:purple → Ruby Light
    "#8c564b": RUBIN_COLORS["gold"],        # tab:brown → Gold
    "#e377c2": RUBIN_COLORS["ruby_light"],  # tab:pink → Ruby Light
    "#7f7f7f": RUBIN_COLORS["slate"],       # tab:gray → Slate
    "#bcbd22": RUBIN_COLORS["gold"],        # tab:olive → Gold
    "#17becf": RUBIN_COLORS["slate_light"], # tab:cyan → Slate Light
    # CSS named colors (sklift, seaborn)
    "blue":    RUBIN_COLORS["ruby"],
    "red":     RUBIN_COLORS["ruby_light"],
    "green":   RUBIN_COLORS["ruby_dark"],
    "orange":  RUBIN_COLORS["gold"],
    "purple":  RUBIN_COLORS["ruby_light"],
    # Fill-Versionen (hellere Varianten)
    "#aec7e8": RUBIN_COLORS["ruby_pale"],   # light blue fill → Ruby Pale
    "#ffbb78": RUBIN_COLORS["gold_light"],  # light orange fill → Gold Light
    "#98df8a": RUBIN_COLORS["ruby_pale"],   # light green fill → Ruby Pale
    "#ff9896": RUBIN_COLORS["ruby_pale"],   # light red fill → Ruby Pale
    "#c5b0d5": RUBIN_COLORS["ruby_pale"],   # light purple fill → Ruby Pale
}


def _hex_to_rgb(h: str) -> tuple:
    """Konvertiert Hex-String zu (R, G, B) float-Tuple [0..1]."""
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def _color_to_rgb(c) -> tuple | None:
    """Extrahiert (R, G, B) [0..1] aus beliebigem matplotlib-Farbformat."""
    if c is None or c == "none":
        return None
    import matplotlib.colors as mcolors
    try:
        rgba = mcolors.to_rgba(c)
        return rgba[:3]
    except (ValueError, TypeError):
        return None


def _color_distance(c1: tuple, c2: tuple) -> float:
    """Euklidische Distanz im RGB-Raum [0..1]."""
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5


def _find_mapping(rgb: tuple, color_map: dict, tolerance: float = 0.15) -> str | None:
    """Findet die nächste Quellfarbe im Mapping und gibt die Zielfarbe zurück."""
    best_dist = tolerance
    best_target = None
    for src_hex, target_hex in color_map.items():
        src_rgb = _color_to_rgb(src_hex)
        if src_rgb is None:
            continue
        d = _color_distance(rgb, src_rgb)
        if d < best_dist:
            best_dist = d
            best_target = target_hex
    return best_target


def recolor_figure(
    fig,
    color_map: dict | None = None,
    tolerance: float = 0.15,
    recolor_fills: bool = True,
) -> None:
    """Mappt Farben einer matplotlib-Figure von externen Paletten auf rubin.

    Arbeitet auf Artist-Ebene (Lines, Patches, Collections, ErrorBars) –
    kein Pixel-Processing nötig. Kann direkt nach dem Erzeugen eines Plots
    aus externen Bibliotheken (EconML, scikit-uplift) aufgerufen werden.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Die umzufärbende Figure.
    color_map : dict | None
        Optionales Mapping {Quellfarbe: Zielfarbe} als Hex-Strings.
        Default: Mapping von matplotlib tab10 + CSS-Farben → rubin-Palette.
    tolerance : float
        Maximale RGB-Distanz [0..1] für Farb-Matching. Default 0.15.
        Höhere Werte matchen mehr Farben, aber können zu Fehltreffern führen.
    recolor_fills : bool
        Ob auch Füllflächen (PolyCollection, Patches) umgefärbt werden.

    Usage
    -----
    ::

        from econml.validate import DRTester
        # ... tester aufsetzen ...
        res = tester.evaluate_all(X, X_train)
        cal_fig = res.plot_cal(1).get_figure()

        from rubin.utils.plot_theme import recolor_figure
        recolor_figure(cal_fig)  # ← einzeilig, fertig
    """
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    import matplotlib.collections as mcoll
    import matplotlib.text as mtext

    cmap = color_map if color_map is not None else _DEFAULT_COLOR_MAP

    def _remap(color_val):
        """Gibt remappte Farbe oder None zurück."""
        rgb = _color_to_rgb(color_val)
        if rgb is None:
            return None
        target = _find_mapping(rgb, cmap, tolerance)
        return target

    for ax in fig.get_axes():
        # ── Lines (Line2D) ──
        for line in ax.get_lines():
            c = line.get_color()
            new = _remap(c)
            if new:
                line.set_color(new)
            # Marker edge/face
            mec = line.get_markeredgecolor()
            new_mec = _remap(mec)
            if new_mec:
                line.set_markeredgecolor(new_mec)
            mfc = line.get_markerfacecolor()
            new_mfc = _remap(mfc)
            if new_mfc:
                line.set_markerfacecolor(new_mfc)

        # ── Patches (Bars, Rectangles, etc.) ──
        for patch in ax.patches:
            fc = patch.get_facecolor()
            new_fc = _remap(fc)
            if new_fc and recolor_fills:
                patch.set_facecolor(new_fc)
            ec = patch.get_edgecolor()
            new_ec = _remap(ec)
            if new_ec:
                patch.set_edgecolor(new_ec)

        # ── Collections (fill_between, scatter, etc.) ──
        for coll in ax.collections:
            if isinstance(coll, mcoll.PolyCollection) and recolor_fills:
                fcs = coll.get_facecolor()
                new_fcs = []
                for fc in fcs:
                    rgb = tuple(fc[:3])
                    target = _find_mapping(rgb, cmap, tolerance)
                    if target:
                        target_rgb = _hex_to_rgb(target)
                        new_fcs.append((*target_rgb, fc[3]))  # Alpha beibehalten
                    else:
                        new_fcs.append(fc)
                coll.set_facecolor(new_fcs)

                ecs = coll.get_edgecolor()
                new_ecs = []
                for ec in ecs:
                    rgb = tuple(ec[:3])
                    target = _find_mapping(rgb, cmap, tolerance)
                    if target:
                        target_rgb = _hex_to_rgb(target)
                        new_ecs.append((*target_rgb, ec[3]))
                    else:
                        new_ecs.append(ec)
                coll.set_edgecolor(new_ecs)

            elif isinstance(coll, mcoll.PathCollection):
                # Scatter plots
                fcs = coll.get_facecolor()
                new_fcs = []
                for fc in fcs:
                    rgb = tuple(fc[:3])
                    target = _find_mapping(rgb, cmap, tolerance)
                    if target:
                        target_rgb = _hex_to_rgb(target)
                        new_fcs.append((*target_rgb, fc[3]))
                    else:
                        new_fcs.append(fc)
                coll.set_facecolor(new_fcs)

            elif isinstance(coll, mcoll.LineCollection):
                # ErrorBars
                colors = coll.get_colors()
                new_colors = []
                for c in colors:
                    rgb = tuple(c[:3])
                    target = _find_mapping(rgb, cmap, tolerance)
                    if target:
                        target_rgb = _hex_to_rgb(target)
                        new_colors.append((*target_rgb, c[3]))
                    else:
                        new_colors.append(c)
                coll.set_colors(new_colors)

        # ── ErrorBar containers ──
        for container in getattr(ax, "containers", []):
            if hasattr(container, "lines"):
                # ErrorbarContainer: (data_line, caplines, barlinecols)
                for part in container:
                    if isinstance(part, mlines.Line2D):
                        c = part.get_color()
                        new = _remap(c)
                        if new:
                            part.set_color(new)
                    elif isinstance(part, (list, tuple)):
                        for sub in part:
                            if isinstance(sub, mlines.Line2D):
                                c = sub.get_color()
                                new = _remap(c)
                                if new:
                                    sub.set_color(new)

        # ── Legend ──
        legend = ax.get_legend()
        if legend:
            for handle in legend.legend_handles if hasattr(legend, "legend_handles") else legend.legendHandles:
                if isinstance(handle, mlines.Line2D):
                    c = handle.get_color()
                    new = _remap(c)
                    if new:
                        handle.set_color(new)
                    mfc = handle.get_markerfacecolor()
                    new_mfc = _remap(mfc)
                    if new_mfc:
                        handle.set_markerfacecolor(new_mfc)
                elif isinstance(handle, mpatches.Patch):
                    fc = handle.get_facecolor()
                    new_fc = _remap(fc)
                    if new_fc and recolor_fills:
                        handle.set_facecolor(new_fc)
