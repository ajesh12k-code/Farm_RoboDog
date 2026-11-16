import tkinter as tk  # Import Tkinter GUI toolkit

# Shared dark-slate background theme for all console widgets
BG = "#1e242b"


def make_label(parent, text: str, fg: str = "#e6edf3", size: int = 10, bold: bool = True) -> tk.Label:
    """Create a styled Tkinter label matching the parent container background."""
    font = ("Sans", size, "bold") if bold else ("Sans", size)
    return tk.Label(parent, text=text, bg=parent["bg"], fg=fg, font=font)


def make_button(parent, text: str, bg: str, cmd=None, width: int = 30, pady: int = 5, size: int = 9) -> tk.Button:
    """Create a flat, high-contrast action button."""
    return tk.Button(
        parent, text=text, command=cmd, bg=bg, fg="white",
        font=("Sans", size, "bold"), width=width, pady=pady, bd=0
    )


def make_slider_row(parent, name: str, lo: float, hi: float, res: float, init_val: float, on_change):
    """Create a labeled horizontal joint-override slider row."""
    # Container row holding the joint name on the left and slider on the right
    row = tk.Frame(parent, bg=BG)
    row.pack(fill="x", padx=10)
    tk.Label(row, text=name, bg=BG, fg="#cbd5e1", width=9, anchor="w", font=("Sans", 8)).pack(side="left")

    # Configure horizontal scale and bind float callback
    scale = tk.Scale(
        row, from_=lo, to=hi, resolution=res, orient="horizontal", length=185,
        bg=BG, fg="white", highlightthickness=0, command=lambda v: on_change(float(v))
    )
    scale.set(init_val)
    scale.pack(side="right")
