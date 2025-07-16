from markitdown import MarkItDown
import openai


llm_url = "http://127.0.0.1:8080/v1"

model_name = "gemma-3-12b-it-q4_0.gguf"

llm = openai.OpenAI(base_url=llm_url, api_key="")


md = MarkItDown(llm_client=llm, llm_model=model_name)
# md = MarkItDown(enable_plugins=False)
import time

start_time = time.time()
result = md.convert("insurance/1_page_001.png")
end_time = time.time()
print(result.markdown)
print(f"处理耗时: {end_time - start_time:.2f} 秒")


# import os
# import time
# import psutil

# def get_vram_usage():
#     try:
#         import pynvml
#         pynvml.nvmlInit()
#         handle = pynvml.nvmlDeviceGetHandleByIndex(0)
#         meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
#         vram_used_mb = meminfo.used / 1024 / 1024
#         pynvml.nvmlShutdown()
#         return vram_used_mb
#     except Exception as e:
#         return None

# pdf_folder = "insurance/insurance"
# results = []

# for filename in os.listdir(pdf_folder):
#     if filename.lower().endswith(".pdf"):
#         pdf_path = os.path.join(pdf_folder, filename)
#         vram_before = get_vram_usage()
#         start_time = time.time()
#         try:
#             result = md.convert(pdf_path)
#             text_len = len(result.text_content) if hasattr(result, "text_content") else 0
#         except Exception as e:
#             text_len = 0
#             print(f"处理文件 {filename} 时出错: {e}")
#         end_time = time.time()
#         vram_after = get_vram_usage()
#         elapsed = end_time - start_time
#         results.append({
#             "file": filename,
#             "vram_before_mb": vram_before,
#             "vram_after_mb": vram_after,
#             "time_sec": elapsed,
#             "text_length": text_len
#         })
#         print(f"文件: {filename}, VRAM前: {vram_before}MB, VRAM后: {vram_after}MB, 用时: {elapsed:.2f}s, 文本长度: {text_len}")

# # 结果保存到CSV
# import csv
# with open("pdf_test_results-no-llm.csv", "w", newline='', encoding="utf-8") as f:
#     writer = csv.DictWriter(f, fieldnames=["file", "vram_before_mb", "vram_after_mb", "time_sec", "text_length"])
#     writer.writeheader()
#     for row in results:
#         writer.writerow(row)

# print("所有PDF处理完毕，结果已保存到 pdf_test_results.csv")
