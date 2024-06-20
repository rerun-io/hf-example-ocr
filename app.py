from __future__ import annotations

import os
from pathlib import Path
from queue import SimpleQueue
from threading import Thread
from typing import Any

import gradio as gr  # type: ignore
import rerun as rr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from gradio_rerun import Rerun  # type: ignore

from ocr import detect_and_log_layouts

CUSTOM_PATH = "/"

app = FastAPI()

origins = [
    "https://app.rerun.io",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
)

def file_ocr(log_queue: SimpleQueue[Any], file_path: str):
    detect_and_log_layouts(log_queue, file_path)
    log_queue.put("done")

@rr.thread_local_stream("PaddleOCR")
def log_to_rr(file_path: Path):
    stream = rr.binary_stream()

    log_queue: SimpleQueue[Any] = SimpleQueue()
    handle = Thread(target=file_ocr, args=[log_queue, str(file_path)])
    handle.start()

    while True:
        msg = log_queue.get()
        if msg == "done":
            break

        msg_type = msg[0]

        if msg_type == "blueprint":
            blueprint = msg[1]
            rr.send_blueprint(blueprint)
        elif msg_type == "log":
            entity_path = msg[1]
            args = msg[2]
            kwargs = msg[3] if len(msg) >= 4 else {}
            # print(entity_path)
            # print(args)
            # print(kwargs)
            rr.log(entity_path, *args, **kwargs)

        yield stream.read()

    handle.join()
    print("done")

DESCRIPTION = """
## PaddleOCR with [Rerun](https://rerun.io/) for visualization
This space demonstrates the ability to visualize and verify the document layout analysis and text detection using [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).
The [PP-Structure](https://github.com/PaddlePaddle/PaddleOCR/tree/main/ppstructure) used for this task, is an intelligent document analysis system developed by the PaddleOCR team, aims to help developers better complete tasks related to document understanding such as layout analysis and table recognition.
"""

with gr.Blocks() as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                #input_image = gr.Image(label="Input Image", image_mode="RGBA", sources="upload", type="filepath")
                input_file = gr.File(label="Input file (image/pdf)")
            with gr.Row():
                button = gr.Button()
            with gr.Row():
                gr.Examples(
                    examples=[os.path.join("examples", img_name) for img_name in sorted(os.listdir("examples"))],
                    inputs=[input_file],
                    label="Examples",
                    cache_examples=False,
                    examples_per_page=12,
                )
        with gr.Column(scale=4):
            viewer = Rerun(streaming=True, height=900)
    button.click(log_to_rr, inputs=[input_file], outputs=[viewer])

app = gr.mount_gradio_app(app, demo, path=CUSTOM_PATH)
