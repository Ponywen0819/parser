import logging
import os
from pathlib import Path
from typing import Optional
import requests
from docling_core.types.doc.page import SegmentedPage
from dotenv import load_dotenv
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
)
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
import time

def lms_vlm_options(model: str, prompt: str, format: ResponseFormat):
    options = ApiVlmOptions(
        url="http://localhost:8080/v1/chat/completions",  # the default LM Studio
        params=dict(
            model=model,
        ),
        prompt=prompt,
        timeout=90,
        scale=1.0,
        response_format=format,
    )
    return options
def lms_olmocr_vlm_options(model: str):
    def _dynamic_olmocr_prompt(page: Optional[SegmentedPage]) -> str:
        if page is None:
            return (
                "Below is the image of one page of a document. Just return the plain text"
                " representation of this document as if you were reading it naturally.\n"
                "Do not hallucinate.\n"
            )

        anchor = [
            f"Page dimensions: {int(page.dimension.width)}x{int(page.dimension.height)}"
        ]

        for text_cell in page.textline_cells:
            if not text_cell.text.strip():
                continue
            bbox = text_cell.rect.to_bounding_box().to_bottom_left_origin(
                page.dimension.height
            )
            anchor.append(f"[{int(bbox.l)}x{int(bbox.b)}] {text_cell.text}")

        for image_cell in page.bitmap_resources:
            bbox = image_cell.rect.to_bounding_box().to_bottom_left_origin(
                page.dimension.height
            )
            anchor.append(
                f"[Image {int(bbox.l)}x{int(bbox.b)} to {int(bbox.r)}x{int(bbox.t)}]"
            )

        if len(anchor) == 1:
            anchor.append(
                f"[Image 0x0 to {int(page.dimension.width)}x{int(page.dimension.height)}]"
            )

        # Original prompt uses cells sorting. We are skipping it in this demo.

        base_text = "\n".join(anchor)

        return (
            f"Below is the image of one page of a document, as well as some raw textual"
            f" content that was previously extracted for it. Just return the plain text"
            f" representation of this document as if you were reading it naturally.\n"
            f"Do not hallucinate.\n"
            f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"
        )

    options = ApiVlmOptions(
        url="http://localhost:1234/v1/chat/completions",
        params=dict(
            model=model,
        ),
        prompt=_dynamic_olmocr_prompt,
        timeout=90,
        scale=1.0,
        max_size=1024,  # from OlmOcr pipeline
        response_format=ResponseFormat.MARKDOWN,
    )
    return options


def main():
    logging.basicConfig(level=logging.INFO)

    # model = "gemma-3-4b-it-q4_0"
    # model="olmocr-7b-0225-preview"
    model 
    source = "https://arxiv.org/pdf/2501.17887" # 8 page
    # source = "https://arxiv.org/pdf/2507.06230" # 20 pages
    # source = "https://arxiv.org/pdf/2507.06211" # 66 pages
    
    
    # data_folder = Path(__file__).parent / "../../tests/data"
    # input_doc_path = data_folder / "pdf/2507.06211.pdf"
    # input_doc_path = source

    pipeline_options = VlmPipelineOptions(
        enable_remote_services=True  # <-- this is required!
    )

    # The ApiVlmOptions() allows to interface with APIs supporting
    # the multi-modal chat interface. Here follow a few example on how to configure those.

    # One possibility is self-hosting model, e.g. via LM Studio, Ollama or others.

    # Example using the SmolDocling model with LM Studio:
    # (uncomment the following lines)
    # pipeline_options.vlm_options = lms_vlm_options(
    #     model=model,
    #     prompt="OCR the full page to markdown.",
    #     format=ResponseFormat.MARKDOWN,
    # )

    # Example using the Granite Vision model with LM Studio:
    # (uncomment the following lines)
    # pipeline_options.vlm_options = lms_vlm_options(
    #     model="granite-vision-3.2-2b",
    #     prompt="OCR the full page to markdown.",
    #     format=ResponseFormat.MARKDOWN,
    # )

    # Example using the OlmOcr (dynamic prompt) model with LM Studio:
    # (uncomment the following lines)
    pipeline_options.vlm_options = lms_olmocr_vlm_options(
        model=model,
    )

    # Example using the Granite Vision model with Ollama:
    # (uncomment the following lines)
    # pipeline_options.vlm_options = ollama_vlm_options(
    #     model="granite3.2-vision:2b",
    #     prompt="OCR the full page to markdown.",
    # )

    # Another possibility is using online services, e.g. watsonx.ai.
    # Using requires setting the env variables WX_API_KEY and WX_PROJECT_ID.
    # (uncomment the following lines)
    # pipeline_options.vlm_options = watsonx_vlm_options(
    #     model="ibm/granite-vision-3-2-2b", prompt="OCR the full page to markdown."
    # )

    # Create the DocumentConverter and launch the conversion.

    start_time = time.time()
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                pipeline_cls=VlmPipeline,
            )
        }
    )
    result = doc_converter.convert(source)
    with open("result.md", "w") as f:
        f.write(result.document.export_to_text())

    end_time = time.time() - start_time

    print(f"Total time: {end_time:.2f} seconds")

if __name__ == "__main__":
    main()