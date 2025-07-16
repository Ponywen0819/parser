# mineru -p ../insurance/1.pdf -o ./result/1_mineru_result -b vlm-sglang-engine --source local -m ocr -l ch
SECONDS=0

mineru -p ../pdfs/20.pdf -o ./result/pipeline -b pipeline --source local -m ocr -l ch

echo "命令執行時間：$SECONDS 秒"
