from __future__ import annotations

import os
from pathlib import Path

import gradio as gr  # type: ignore
import rerun as rr
import rerun.blueprint as rrb
from gradio_rerun import Rerun  # type: ignore
from ocr import detect_and_log_layout  # type: ignore


@rr.thread_local_stream("OCR")
def log_to_rr(img_path: Path):
    print(img_path)
    stream = rr.binary_stream()

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial2DView(name="Input", contents=["Image/**"]),
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)

    detect_and_log_layout(img_path)

    yield stream.read()

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                input_image = gr.Image(label="Input Image", image_mode="RGBA", sources="upload", type="filepath")
            with gr.Row():
                button = gr.Button()
            with gr.Row():
                gr.Examples(
                    examples=[os.path.join("examples", img_name) for img_name in sorted(os.listdir("examples"))],
                    inputs=[input_image],
                    label="Examples",
                    cache_examples=False,
                    examples_per_page=12,
                )
        with gr.Column(scale=4):
            viewer = Rerun(streaming=True, height=900)
    button.click(log_to_rr, inputs=[input_image], outputs=[viewer])

demo.launch()
