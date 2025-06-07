import tkinter as tk
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from ttkbootstrap.style import Style
from tkinter import messagebox
from threading import Thread

from backend import (
    load_settings, save_settings, run_clustering,
    default_settings
)

def launch_ui():
    settings = load_settings()

    app = tb.Window(themename="cosmo")
    app.title("Lapplace's Custom Hairstyles Merging Tool")
    app.geometry("640x740")

    # Set window icon
    try:
        icon_img = tk.PhotoImage(file="icon.png")
        app.iconphoto(False, icon_img)
    except Exception as e:
        print(f"Failed to load icon.png: {e}")
        icon_img = None

    # Header with centered icon + title
    header_container = tb.Frame(app)
    header_container.pack(pady=10, fill="x")

    header = tb.Frame(header_container)
    header.pack(anchor="center")

    if icon_img:
        icon_label = tb.Label(header, image=icon_img)
        icon_label.image = icon_img  # keep a reference
        icon_label.pack(side="left", padx=(0, 8))

    tb.Label(header, text="Lapplace's Custom Hairstyles Merging Tool", font=("Helvetica", 16, "bold")).pack(side="left")

    def make_card(parent, **kwargs):
        card = tb.Frame(parent, bootstyle="light", padding=15, **kwargs)
        card.pack(fill="x", pady=10, padx=20)
        card.configure(borderwidth=1, relief="solid")
        return card

    def add_hover(widget, base_style="Card.TFrame", hover_style="Card.Hover.TFrame"):
        style = Style()
        if not style.lookup(base_style, "background"):
            style.configure(base_style, background="white")
        if not style.lookup(hover_style, "background"):
            style.configure(hover_style, background="#e9ecef")
        def on_enter(e): widget.configure(style=hover_style)
        def on_leave(e): widget.configure(style=base_style)
        widget.configure(style=base_style)
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    main_frame = tb.Frame(app, padding=10)
    main_frame.pack(fill="both", expand=True)

    # === UI Cards ===
    card1 = make_card(main_frame)
    tb.Label(card1, text="Display Name", font=("Helvetica", 10, "bold")).pack(anchor="w")
    name_var = tk.StringVar(value="Lapplace's Custom Hairs")
    tb.Entry(card1, textvariable=name_var, width=40).pack(anchor="w", pady=(5, 0))
    add_hover(card1)

    card2 = make_card(main_frame)
    tb.Label(card2, text="Agglomerative Threshold", font=("Helvetica", 10, "bold")).pack(anchor="w")
    tb.Label(card2, text="This makes the program more likely to group hairstyles. Lower = fewer groups. Max = 0.5",
             font=("-size", 8), foreground="gray").pack(anchor="w", padx=5)
    agg_frame = tb.Frame(card2)
    agg_frame.pack(anchor="w", pady=5)
    agg_slider = tb.Scale(agg_frame, from_=0.01, to=0.5, orient="horizontal", length=200)
    agg_slider.set(settings["agg_threshold"])
    agg_slider.pack(side="left")
    agg_val = tk.DoubleVar(value=settings["agg_threshold"])
    tb.Entry(agg_frame, textvariable=agg_val, width=6).pack(side="left", padx=5)
    agg_slider.configure(command=lambda val: agg_val.set(round(float(val), 3)))
    agg_val.trace_add("write", lambda *_: agg_slider.set(agg_val.get()))
    add_hover(card2)

    card3 = make_card(main_frame)
    tb.Label(card3, text="Merge Threshold", font=("Helvetica", 10, "bold")).pack(anchor="w")
    tb.Label(card3, text="Merges small groups (<3) into larger ones. Higher = more merging. Max = 1.0",
             font=("-size", 8), foreground="gray").pack(anchor="w", padx=5)
    merge_frame = tb.Frame(card3)
    merge_frame.pack(anchor="w", pady=5)
    merge_slider = tb.Scale(merge_frame, from_=0.01, to=1.0, orient="horizontal", length=200)
    merge_slider.set(settings["merge_threshold"])
    merge_slider.pack(side="left")
    merge_val = tk.DoubleVar(value=settings["merge_threshold"])
    tb.Entry(merge_frame, textvariable=merge_val, width=6).pack(side="left", padx=5)
    merge_slider.configure(command=lambda val: merge_val.set(round(float(val), 3)))
    merge_val.trace_add("write", lambda *_: merge_slider.set(merge_val.get()))
    add_hover(card3)

    card4 = make_card(main_frame)
    group_var = tk.BooleanVar(value=settings["grouping"])
    tb.Checkbutton(card4, text="Enable Grouping", variable=group_var, bootstyle="success").pack(anchor="w", pady=(0, 5))
    tb.Label(card4, text="Groups hairstyles by similarity with version numbers.",
             font=("-size", 8), foreground="gray").pack(anchor="w", padx=5)
    debug_var = tk.BooleanVar(value=False)
    tb.Checkbutton(card4, text="Enable Debug Mode", variable=debug_var, bootstyle="info").pack(anchor="w", pady=(10, 0))
    tb.Label(card4, text="This mode puts grouped hairstyles into 'grouped_hairstyles' for review.",
             font=("-size", 8), foreground="gray").pack(anchor="w", padx=5)
    add_hover(card4)

    card5 = make_card(main_frame)
    progress = tb.Progressbar(card5, maximum=100, mode='determinate')
    progress.pack(fill="x", pady=(0, 5))
    status_label = tb.Label(card5, text="Idle.")
    status_label.pack()
    add_hover(card5)

    def update_progress(current, total):
        percent = int((current / total) * 100)
        progress["value"] = percent
        status_label.config(text=f"Computing distances: {percent}%")

    def animate_running():
        dots = ["", ".", "..", "..."]
        idx = 0
        def loop():
            nonlocal idx
            if running[0]:
                status_label.config(text=f"Running{dots[idx]}")
                idx = (idx + 1) % 4
                status_label.after(500, loop)
        loop()

    def on_done(success, msg):
        running[0] = False
        run_btn.config(state="normal")
        reset_btn.config(state="normal")
        progress["value"] = 100 if success else 0
        status_label.config(text=msg)
        if success:
            messagebox.showinfo("Done", msg)
        else:
            messagebox.showerror("Error", msg)

    def on_run():
        progress["value"] = 0
        running[0] = True
        run_btn.config(state="disabled")
        reset_btn.config(state="disabled")
        animate_running()
        save_settings(agg_val.get(), merge_val.get(), group_var.get())
        Thread(target=run_clustering, args=(
            agg_val.get(), merge_val.get(), group_var.get(),
            update_progress, on_done, name_var.get(), debug_var.get()
        ), daemon=True).start()

    def on_reset():
        agg_slider.set(default_settings["agg_threshold"])
        merge_slider.set(default_settings["merge_threshold"])
        group_var.set(default_settings["grouping"])

    running = [False]
    btn_frame = tb.Frame(app)
    btn_frame.pack(pady=10)
    run_btn = tb.Button(btn_frame, text="Start Merging", command=on_run, bootstyle="primary")
    reset_btn = tb.Button(btn_frame, text="Reset Settings", command=on_reset, bootstyle="secondary")
    run_btn.pack(side="left", padx=10)
    reset_btn.pack(side="left", padx=10)

    app.mainloop()

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    launch_ui()
