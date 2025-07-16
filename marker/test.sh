marker_single insurance/1.pdf \
 --output_dir insurance/1_marker_result \
 --output_format markdown \
 --use_llm \
 --llm_service=marker.services.ollama.OllamaService \
 --ollama_base_url http://127.0.0.1:8080 \
 --ollama_model gemma-3-12b-it-q4_0
 