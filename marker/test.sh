SECONDS=0


marker_single ../pdfs/c.pdf \
 --output_dir basic_result \
 --output_format markdown \
 --debug \
#  --use_llm \
#  --llm_service=marker.services.openai.OpenAIService \
#  --openai_base_url http://127.0.0.1:1234/v1\
#  --openai_model qwen2-vl-2b-instruct \
#  --openai_api_key test\
#  --gemini_api_key AIzaSyDIoM9_h6ft4WSt18xXp2tFCzB691mx4H8


echo "命令執行時間：$SECONDS 秒"