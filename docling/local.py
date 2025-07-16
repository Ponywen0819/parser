from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

source = "https://arxiv.org/pdf/2501.17887"
# source = "https://arxiv.org/pdf/2507.06230" # 20 pages
# source = "https://arxiv.org/pdf/2507.06211" # 66 page

pipeline_options = VlmPipelineOptions(
    # vlm_options=vlm_model_specs.PHI4_TRANSFORMERS,
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        ),
    }
)

doc = converter.convert(source=source).document

print(doc.export_to_markdown())