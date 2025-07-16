from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
)
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
import time
import pandas as pd
import logging
from pathlib import Path
_log = logging.getLogger(__name__)

output_dir = Path("./result")
# source = "../insurance/1.pdf"  # document per local path or URL
# source = "https://arxiv.org/pdf/2501.17887"
# source = "https://arxiv.org/pdf/2507.06230" # 20 pages
source = "https://arxiv.org/pdf/2507.06211" # 66 page
start_time = time.time()

model = "gemma-3-4b-it-q4_0"
# model = "gemma-3-12b-it-q4_0"

options = ApiVlmOptions(
    url="http://127.0.0.1:8080/v1/chat/completions",  # the default Ollama endpoint
    params=dict(
        model=model,
    ),
    prompt="OCR the full page to markdown.",
    timeout=60,
    scale=1.0,
    response_format=ResponseFormat.MARKDOWN,
)
pipeline_options = VlmPipelineOptions(
    enable_remote_services=True,
    vlm_options=options,
    # images_scale=1.0,
    # generate_page_images=True,
)

pipeline_options.vlm_options  = options

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        ),
    }
)

result = converter.convert(source)
print(result.document.export_to_markdown())
# print(result.document.export_to_html())

end_time = time.time() - start_time

print(f"Total time: {end_time:.2f} seconds")

# print(res.document.export_to_markdown())