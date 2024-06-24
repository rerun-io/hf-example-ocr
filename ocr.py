#!/usr/bin/env python3
"""OCR template."""

from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path
from queue import SimpleQueue
from typing import Any, Final, Iterable, Optional, TypeAlias

import cv2 as cv2
import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore
import pdf2image  # type: ignore
import rerun as rr  # pip install rerun-sdk
import rerun.blueprint as rrb
from paddleocr import PPStructure  # type: ignore
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes  # type: ignore

EXAMPLE_DIR: Final = Path(os.path.dirname(__file__))
DATASET_DIR: Final = EXAMPLE_DIR / "dataset"

SAMPLE_IMAGE_URLs = ["https://storage.googleapis.com/rerun-example-datasets/ocr/paper.png"]

PAGE_LIMIT = 10

LayoutStructure: TypeAlias = tuple[
    list[str], list[str], list[rrb.Spatial2DView], list[rrb.Spatial2DView], list[rrb.Spatial2DView]
]

# Supportive Classes


class Color:
    Red = (255, 0, 0)
    Green = (0, 255, 0)
    Blue = (0, 0, 255)
    Yellow = (255, 255, 0)
    Cyan = (0, 255, 255)
    Magenta = (255, 0, 255)
    Purple = (128, 0, 128)
    Orange = (255, 165, 0)


"""
LayoutType:
    Defines an enumeration for different types of document layout elements, each associated with a unique number, name,
    and color. Types:
    - UNKNOWN: Default type for undefined or unrecognized elements, represented by purple.
    - TITLE: Represents the title of a document, represented by red.
    - TEXT: Represents plain text content within the document, represented by green.
    - FIGURE: Represents graphical or image content, represented by blue.
    - FIGURE_CAPTION: Represents captions for figures, represented by yellow.
    - TABLE: Represents tabular data, represented by cyan.
    - TABLE_CAPTION: Represents captions for tables, represented by magenta.
    - REFERENCE: Represents citation references within the document, also represented by purple.
    - Footer: Represents footer of the document, represented as orange.
"""


class LayoutType(Enum):
    UNKNOWN = (0, "unknown", Color.Purple)
    TITLE = (1, "title", Color.Red)
    TEXT = (2, "text", Color.Green)
    FIGURE = (3, "figure", Color.Blue)
    FIGURE_CAPTION = (4, "figure_caption", Color.Yellow)
    TABLE = (5, "table", Color.Cyan)
    TABLE_CAPTION = (6, "table_caption", Color.Magenta)
    REFERENCE = (7, "reference", Color.Purple)
    FOOTER = (8, "footer", Color.Orange)

    def __str__(self) -> str:
        return str(self.value[1])  # Returns the string part (type)

    @property
    def number(self) -> int:
        return self.value[0]  # Returns the numerical identifier

    @property
    def type(self) -> str:
        return self.value[1]  # Returns the type

    @property
    def color(self) -> tuple[int, int, int]:
        return self.value[2]  # Returns the color

    @staticmethod
    def get_class_id(text: str) -> int:
        try:
            return LayoutType[text.upper()].number
        except KeyError:
            logging.warning(f"Invalid layout type {text}")
            return 0

    @staticmethod
    def get_type(text: str) -> LayoutType:
        try:
            return LayoutType[text.upper()]
        except KeyError:
            logging.warning(f"Invalid layout type {text}")
            return LayoutType.UNKNOWN

    @classmethod
    def get_annotation(cls) -> list[tuple[int, str, tuple[int, int, int]]]:
        return [(layout.number, layout.type, layout.color) for layout in cls]


"""
Layout Class:
    The main purpose of this class is to:
    1. Keep track of the layout types (including type, numbering)
    2. Save the detections for each layout (text, img or table)
    3. Save the bounding box of each detected layout
    4. Generate the recovery text document
"""


class Layout:
    def __init__(self, page_number: int, show_unknown: bool = False):
        self.counts = {layout_type: 0 for layout_type in LayoutType}
        self.records: dict[LayoutType, Any] = {layout_type: [] for layout_type in LayoutType}
        self.recovery = """"""
        self.page_number = page_number
        self.show_unknown = show_unknown

    def add(
        self,
        layout_type: LayoutType,
        bounding_box: list[int],
        detections: Optional[Iterable[dict[str, Any]]] = None,
        table: Optional[str] = None,
        figure: Optional[dict[str, Any]] = None,
    ) -> None:
        if layout_type in LayoutType:
            self.counts[layout_type] += 1
            name = f"{layout_type}{self.counts[layout_type]}"
            logging.info(f"Saved layout type {layout_type} with name: {name}")
            self.records[layout_type].append({
                "type": layout_type,
                "name": name,
                "bounding_box": bounding_box,
                "detections": detections,
                "table": table,
            })
            if layout_type != LayoutType.UNKNOWN or self.show_unknown:  # Discards the unknown layout types detections
                path = f"recording://page_{self.page_number}/Image/{layout_type.type.title()}/{name.title()}"
                self.recovery += f"\n\n## [{name.title()}]({path})\n\n"  # Log Type as Heading
                # Enhancement - Logged image for Figure type TODO(#6517)
                if layout_type == LayoutType.TABLE:
                    if table:
                        self.recovery += table  # Log details (table)
                elif detections:
                    for index, detection in enumerate(detections):
                        path_text = f"recording://page_{self.page_number}/Image/{layout_type.type.title()}/{name.title()}/Detections/{index}"
                        self.recovery += f' [{detection["text"]}]({path_text})'  # Log details (text)
        else:
            logging.warning(f"Invalid layout type detected: {layout_type}")

    def get_count(self, layout_type: LayoutType) -> int:
        if layout_type in LayoutType:
            return self.counts[layout_type]
        else:
            raise ValueError("Invalid layout type")

    def get_records(self) -> dict[LayoutType, list[dict[str, Any]]]:
        return self.records

    def save_all_layouts(self, results: list[dict[str, Any]]) -> None:
        for line in results:
            self.save_layout_data(line)
        for layout_type in LayoutType:
            logging.info(f"Number of detections for type {layout_type}: {self.counts[layout_type]}")

    def save_layout_data(self, line: dict[str, Any]) -> None:
        type = line.get("type", "empty")
        box = line.get("bbox", [0, 0, 0, 0])
        layout_type = LayoutType.get_type(type)
        detections, table, img = [], None, None
        if layout_type == LayoutType.TABLE:
            table = self.get_table_markdown(line)
        elif layout_type == LayoutType.FIGURE:
            detections = self.get_detections(line)
            img = line.get("img")  # Currently not in use
        else:
            detections = self.get_detections(line)
        self.add(layout_type, box, detections=detections, table=table, figure=img)

    @staticmethod
    def get_detections(line: dict[str, Any]) -> list[dict[str, Any]]:
        detections = []
        results = line.get("res")
        if results is not None:
            for i, result in enumerate(results):
                text = result.get("text")
                confidence = result.get("confidence")
                box = result.get("text_region")
                x_min, y_min = box[0]
                x_max, y_max = box[2]
                new_box = [x_min, y_min, x_max, y_max]
                detections.append({"id": i, "text": text, "confidence": confidence, "box": new_box})
        return detections

    # Safely attempt to extract the HTML table from the results
    @staticmethod
    def get_table_markdown(line: dict[str, Any]) -> str:
        try:
            html_table = line.get("res", {}).get("html")
            if not html_table:
                return "No table found."

            dataframes = pd.read_html(html_table)
            if not dataframes:
                return "No data extracted from the table."

            markdown_table = dataframes[0].to_markdown()
            return markdown_table  # type: ignore[no-any-return]

        except Exception as e:
            return f"Error processing the table: {str(e)}"


def process_layout_records(log_queue: SimpleQueue[Any], layout: Layout) -> LayoutStructure:
    paths, detections_paths = [], []
    zoom_paths: list[rrb.Spatial2DView] = []
    zoom_paths_figures: list[rrb.Spatial2DView] = []
    zoom_paths_tables: list[rrb.Spatial2DView] = []
    zoom_paths_texts: list[rrb.Spatial2DView] = []

    page_path = f'page_{layout.page_number}'
    for layout_type in LayoutType:
        for record in layout.records[layout_type]:
            record_name = record["name"].title()
            record_base_path = f"{page_path}/Image/{layout_type.type.title()}/{record_name}"
            paths.append(f"-{record_base_path}/**")
            detections_paths.append(f"-{record_base_path}/Detections/**")

            # Log bounding box
            log_queue.put([
                "log",
                record_base_path,
                [
                    rr.Boxes2D(
                        array=record["bounding_box"],
                        array_format=rr.Box2DFormat.XYXY,
                        labels=[str(layout_type.type)],
                        class_ids=[str(layout_type.number)],
                    ),
                    rr.AnyValues(name=record_name),
                ],
            ])

            log_detections(log_queue, layout_type, record, record_base_path)

            # Prepare zoom path views
            update_zoom_paths(
                layout,
                layout_type,
                record,
                paths,
                page_path,
                zoom_paths,
                zoom_paths_figures,
                zoom_paths_tables,
                zoom_paths_texts,
            )

    return paths, detections_paths, zoom_paths_figures, zoom_paths_tables, zoom_paths_texts


def log_detections(log_queue: SimpleQueue, layout_type: LayoutType, record: dict[str, Any], page_path: str) -> None:
    if layout_type == LayoutType.TABLE:
        log_queue.put([
            "log",
            f"Extracted{record['name']}",
            [rr.TextDocument(record["table"], media_type=rr.MediaType.MARKDOWN)],
        ])
    else:
        for detection in record.get("detections", []):
            log_queue.put([
                "log",
                f"{page_path}/Detections/{detection['id']}",
                [
                    rr.Boxes2D(
                        array=detection["box"], array_format=rr.Box2DFormat.XYXY, class_ids=[str(layout_type.number)]
                    ),
                    rr.AnyValues(
                        DetectionID=detection["id"], Text=detection["text"], Confidence=detection["confidence"]
                    ),
                ],
            ])


def update_zoom_paths(
    layout: Layout,
    layout_type: LayoutType,
    record: dict[str, Any],
    paths: list[str],
    page_path: str,
    zoom_paths: list[rrb.Spatial2DView],
    zoom_paths_figures: list[rrb.Spatial2DView],
    zoom_paths_tables: list[rrb.Spatial2DView],
    zoom_paths_texts: list[rrb.Spatial2DView],
) -> None:
    if layout_type in [LayoutType.FIGURE, LayoutType.TABLE, LayoutType.TEXT]:
        current_paths = paths.copy()
        current_paths.remove(f"-{page_path}/Image/{layout_type.type.title()}/{record['name'].title()}/**")
        bounds = rrb.VisualBounds2D(
            x_range=[record["bounding_box"][0] - 10, record["bounding_box"][2] + 10],
            y_range=[record["bounding_box"][1] - 10, record["bounding_box"][3] + 10],
        )

        # Add to zoom paths
        view = rrb.Spatial2DView(
            name=record["name"].title(), contents=[f"{page_path}/Image/**"] + current_paths, visual_bounds=bounds
        )
        zoom_paths.append(view)

        # Add to type-specific zoom paths
        if layout_type == LayoutType.FIGURE:
            zoom_paths_figures.append(view)
        elif layout_type == LayoutType.TABLE:
            zoom_paths_tables.append(view)
        elif layout_type != LayoutType.UNKNOWN or layout.show_unknown:
            zoom_paths_texts.append(view)


def generate_blueprint(
    layouts: list[Layout],
    processed_layouts: list[LayoutStructure],
) -> rrb.Blueprint:
    page_tabs = []
    for layout, processed_layout in zip(layouts, processed_layouts):
        page_path = f'page_{layout.page_number}'
        paths, detections_paths, zoom_paths_figures, zoom_paths_tables, zoom_paths_texts = processed_layout

        section_tabs = []
        content_data: dict[str, Any] = {
            "Figures": zoom_paths_figures,
            "Tables": zoom_paths_tables,
            "Texts": zoom_paths_texts,
        }

        for name, paths in content_data.items():
            if paths:
                section_tabs.append(rrb.Tabs(*paths, name=name))  # type: ignore[arg-type]

        page_tabs.append(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial2DView(
                        name="Layout",
                        origin=f"{page_path}/Image/",
                        contents=[f"{page_path}/Image/**"] + detections_paths,
                    ),
                    rrb.Spatial2DView(name="Detections", contents=[f"{page_path}/Image/**"]),
                    rrb.Vertical(
                        rrb.TextDocumentView(name="Progress", contents=["progress/**"]),
                        rrb.TextDocumentView(name="Recovery", contents=f"{page_path}/Recovery"),
                        row_shares=[1, 4]
                    )
                ),
                rrb.Horizontal(*section_tabs),
                name=page_path,
                row_shares=[4, 3],
            )
        )

    return rrb.Blueprint(
        rrb.Tabs(*page_tabs),
        collapse_panels=True,
    )


def detect_and_log_layouts(log_queue: SimpleQueue[Any], file_path: str, start_page: int = 1, end_page: int | None = -1) -> None:
    if end_page == -1:
        end_page = start_page + PAGE_LIMIT-1
    if end_page < start_page:
        end_page = start_page
    print(start_page, end_page)

    images: list[npt.NDArray[np.uint8]] = []
    if file_path.endswith(".pdf"):
        # convert pdf to images
        images.extend(np.array(img, dtype=np.uint8) for img in pdf2image.convert_from_path(file_path, first_page=start_page, last_page=end_page))
        print(len(images))
        if len(images) > PAGE_LIMIT:
            log_queue.put([
                "log",
                "progress",
                [rr.TextDocument(f"Too many pages requsted: {len(images)} requested but the limit is {PAGE_LIMIT}")],
            ])
            return
    else:
        # read image
        img = cv2.imread(file_path)
        coloured_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(coloured_image.astype(np.uint8))

    # Extracte the layout from each image
    layouts: list[Layout] = []
    page_numbers = [i + start_page for i in range(len(images))]
    processed_layouts: list[LayoutStructure] = []
    for i, (image, page_number) in enumerate(zip(images, page_numbers)):
        layouts.append(detect_and_log_layout(log_queue, image, page_number))

        # Generate and send a blueprint based on the detected layouts
        processed_layouts.append(
            process_layout_records(
                log_queue,
                layouts[-1],
            )
        )
        logging.info("Sending blueprint...")
        blueprint = generate_blueprint(layouts, processed_layouts)
        log_queue.put(["blueprint", blueprint])
        logging.info("Blueprint sent...")


def detect_and_log_layout(log_queue: SimpleQueue, coloured_image: npt.NDArray[np.uint8], page_number: int) -> Layout:
    # Layout Object - This will contain the detected layouts and their detections
    layout = Layout(page_number)
    page_path = f'page_{page_number}'

    # Log Image and add Annotation Context
    log_queue.put([
        "log",
        f"{page_path}/Image",
        [rr.Image(coloured_image)],
    ])
    log_queue.put([
        "log",
        f"{page_path}/Image",
        # The annotation is defined in the Layout class based on its properties
        [rr.AnnotationContext(LayoutType.get_annotation())],
        {
            "static": True,
        },
    ])

    # Paddle Model - Getting Predictions
    logging.info("Start detection... (It usually takes more than 10-20 seconds per page)")
    ocr_model_pp = PPStructure(show_log=False, recovery=True)
    logging.info("model loaded")
    result_pp = ocr_model_pp(coloured_image)
    _, w, _ = coloured_image.shape
    result_pp = sorted_layout_boxes(result_pp, w)
    logging.info("Detection finished...")

    # Add results to the layout
    layout.save_all_layouts(result_pp)
    logging.info("All results are saved...")

    # Recovery Text Document for the detected text
    log_queue.put([
        "log",
        f"{page_path}/Recovery",
        [rr.TextDocument(layout.recovery, media_type=rr.MediaType.MARKDOWN)],
    ])

    return layout
