import tkinter as tk  # Import Tkinter GUI toolkit

# Shared dark-slate background theme for all console widgets
BG = "#1e242b"
# Muted colors for a dimmed (unavailable) action button
DIM_BG, DIM_FG = "#3a414b", "#8b95a3"


def make_label(parent, text: str, fg: str = "#e6edf3", size: int = 10, bold: bool = True) -> tk.Label:
    """Create a styled Tkinter label matching the parent container background."""
    font = ("Sans", size, "bold") if bold else ("Sans", size)
    return tk.Label(parent, text=text, bg=parent["bg"], fg=fg, font=font)


def make_button(parent, text: str, bg: str, cmd=None, width: int = 30, pady: int = 5, size: int = 9) -> tk.Button:
    """Create a flat, high-contrast action button.

    takefocus=0 keeps buttons out of Tab focus so Space (STOP key) never 'clicks' a focused button.
    """
    return tk.Button(
        parent, text=text, command=cmd, bg=bg, fg="white", activebackground=bg, activeforeground="white",
        font=("Sans", size, "bold"), width=width, pady=pady, bd=0, takefocus=0
    )


def set_button_dimmed(btn: tk.Button, dimmed: bool, bg: str):
    """Grey a button out (still clickable, so the node can explain why) or restore its color."""
    if dimmed:
        btn.config(bg=DIM_BG, fg=DIM_FG, activebackground=DIM_BG, activeforeground=DIM_FG)
    else:
        btn.config(bg=bg, fg="white", activebackground=bg, activeforeground="white")


def make_slider_row(parent, name: str, lo: float, hi: float, res: float, init_val: float, on_change,
                    length: int = 185) -> tk.Scale:
    """Create a labeled horizontal slider row and return the Scale."""
    # Container row holding the joint name on the left and slider on the right
    row = tk.Frame(parent, bg=BG)
    row.pack(fill="x", padx=10)
    tk.Label(row, text=name, bg=BG, fg="#cbd5e1", width=9, anchor="w", font=("Sans", 8)).pack(side="left")

    # Configure horizontal scale and bind float callback
    scale = tk.Scale(
        row, from_=lo, to=hi, resolution=res, orient="horizontal", length=length, width=10,
        bg=BG, fg="white", highlightthickness=0, takefocus=0, font=("Sans", 7),
        command=lambda v: on_change(float(v))
    )
    scale.set(init_val)
    scale.pack(side="right")
    return scale


def make_scrollable(root, width: int) -> tk.Frame:
    """Fill ``root`` with a vertically scrollable area and return the inner frame to pack into.

    Keeps the console usable on a 1600x900 desktop next to the MuJoCo viewer when the window is
    shorter than its content. Mouse wheel works on X11 (Button-4/5) and Windows/macOS (MouseWheel).
    """
    canvas = tk.Canvas(root, bg=BG, highlightthickness=0, width=width)
    bar = tk.Scrollbar(root, orient="vertical", command=canvas.yview, takefocus=0)
    canvas.configure(yscrollcommand=bar.set)
    bar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    inner = tk.Frame(canvas, bg=BG)
    window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
    # Keep the scroll region matched to the content and the content as wide as the canvas
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window_id, width=e.width))

    def _scroll(units: int):
        # Only scroll when the content is actually taller than the visible area
        if inner.winfo_reqheight() > canvas.winfo_height():
            canvas.yview_scroll(units, "units")

    root.bind_all("<Button-4>", lambda e: _scroll(-2))
    root.bind_all("<Button-5>", lambda e: _scroll(2))
    root.bind_all("<MouseWheel>", lambda e: _scroll(-1 if e.delta > 0 else 1))
    return inner
