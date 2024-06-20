from __future__ import annotations

import os
from pathlib import Path
from queue import SimpleQueue
from threading import Thread
from time import sleep
from typing import Any

import gradio as gr  # type: ignore
import rerun as rr
import rerun.blueprint as rrb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from gradio_rerun import Rerun  # type: ignore

from ocr import detect_and_log_layouts, PAGE_LIMIT

CUSTOM_PATH = "/"

app = FastAPI()

origins = [
    "https://app.rerun.io",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
)


def progress_log(log_queue: SimpleQueue[Any], done: SimpleQueue[Any]):
    dots = 0
    while True:
        if not done.empty():
            break
        sleep(0.7)
        log_queue.put([
            "log",
            "progress",
            [rr.TextDocument(f"working{'.'*(dots+1)}")]
        ])
        dots = (dots + 1) % 5


def file_ocr(log_queue: SimpleQueue[Any], file_path: str, start_page: int, end_page: int):
    detect_and_log_layouts(log_queue, file_path, start_page, end_page)
    log_queue.put("done")


@rr.thread_local_stream("PaddleOCR")
def log_to_rr(file_path: Path, start_page: int = 1, end_page: int = -1):
    stream = rr.binary_stream()

    log_queue: SimpleQueue[Any] = SimpleQueue()
    done: SimpleQueue[Any] = SimpleQueue()
    Thread(target=progress_log, args=[log_queue, done]).start()
    handle = Thread(target=file_ocr, args=[log_queue, str(file_path), start_page, end_page])
    handle.start()

    rr.send_blueprint(rrb.Blueprint(
        rrb.TextDocumentView(contents=["progress/**"]),
        collapse_panels=True,
    ))
    yield stream.read()

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
            rr.log(entity_path, *args, **kwargs)

        yield stream.read()

    rr.log("progress",rr.TextDocument("Done!"))
    yield stream.read()
    done.put(())
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
            with gr.Tab(label="Upload Image"):
                with gr.Row():
                    input_image_file = gr.Image(label="Input Image", image_mode="RGBA", sources="upload", type="filepath")
                    # input_image_file = gr.Image(label="Input image")
                with gr.Row():
                    image_button = gr.Button()
                with gr.Row():
                    gr.Examples(
                        examples=[
                            os.path.join("image_examples", img_name)
                            for img_name in sorted(os.listdir("image_examples"))
                        ],
                        inputs=[input_image_file],
                        label="Examples",
                        cache_examples=False,
                        examples_per_page=12,
                    )
            with gr.Tab(label="Upload pdf"):
                with gr.Row():
                    input_pdf_file = gr.File(label="Input pdf")
                gr.Markdown(f"Max {PAGE_LIMIT} pages, -1 on end page means max number of pages")
                with gr.Row():
                    start_page_number = gr.Number(1, label="Start page", minimum=1)
                with gr.Row():
                    end_page_number = gr.Number(-1, label="End page")
                with gr.Row():
                    pdf_button = gr.Button()
                with gr.Row():
                    gr.Examples(
                        examples=[
                            os.path.join("pdf_examples", img_name) for img_name in sorted(os.listdir("pdf_examples"))
                        ],
                        inputs=[input_pdf_file],
                        label="Examples",
                        cache_examples=False,
                        examples_per_page=12,
                    )
        with gr.Column(scale=4):
            viewer = Rerun(streaming=True, height=900)

        image_button.click(log_to_rr, inputs=[input_image_file], outputs=[viewer])
        pdf_button.click(log_to_rr, inputs=[input_pdf_file, start_page_number, end_page_number], outputs=[viewer])


app = gr.mount_gradio_app(app, demo, path=CUSTOM_PATH)
