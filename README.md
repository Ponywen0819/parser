# PDF提取器比对系统

一个用于比较不同PDF文本提取工具性能和结果的完整框架。

## 功能特性

🔍 **多引擎支持**: 支持三种主流PDF提取工具
- **Docling**: 高质量文档转换
- **Marker**: Markdown格式输出
- **MinerU**: 多模态文档理解

📊 **性能监控**: 实时监控各项指标
- 处理时间
- 内存使用
- CPU占用率
- 文本输出质量

🔄 **结果比对**: 全面的结果分析
- 文本相似度计算
- 单词重叠率分析
- 字符级比较
- 统计指标对比

📈 **可视化报告**: 专业的报告生成
- HTML格式详细报告
- 图表可视化分析
- CSV数据导出
- JSON格式原始数据

## 系统要求

- Python 3.8+
- 足够的内存空间（建议8GB+）
- 硬盘空间（用于存储结果和日志）

## 安装步骤

### 1. 克隆或下载项目
```bash
# 如果有git仓库
git clone <repository_url>
cd parser

# 或者直接下载并解压到parser目录
```

### 2. 安装Python依赖
```bash
pip install -r requirements.txt
```

### 3. 安装PDF提取工具

#### Docling
```bash
pip install docling
```

#### Marker
```bash
pip install marker-pdf
# 或者从源码安装
git clone https://github.com/datalab-to/marker.git
cd marker
pip install -e .
```

#### MinerU
```bash
pip install magic-pdf
# 或者从源码安装
git clone https://github.com/opendatalab/MinerU.git
cd MinerU
pip install -e .
```

### 4. 准备测试数据
将PDF文件放置在 `insurance/insurance/` 目录中。项目已包含示例数据集。

## 使用方法

### 检查系统状态
```bash
python main.py check
```
此命令会检查：
- 所有提取器的可用性
- 数据目录中的PDF文件数量
- 系统依赖状态

### 运行基准测试

#### 样本测试 (推荐首次使用)
```bash
python main.py benchmark --sample-size 5 --generate-report
```

#### 完整测试
```bash
python main.py benchmark --full --generate-report
```

#### 仅运行基准测试（不生成报告）
```bash
python main.py benchmark --sample-size 10
```

### 测试单个文件
```bash
# 使用默认提取器 (docling)
python main.py single path/to/test.pdf --show-text

# 指定提取器
python main.py single path/to/test.pdf --extractor marker --show-text

# 不显示文本内容
python main.py single path/to/test.pdf --extractor mineru
```

## 配置选项

编辑 `config.py` 文件来调整系统设置：

```python
# 测试配置
TEST_SAMPLE_SIZE = 10  # 样本测试的文件数量
FULL_TEST = False      # 是否默认运行完整测试

# 性能监控配置
MEMORY_INTERVAL = 0.1  # 内存监控间隔（秒）
ENABLE_PROFILING = True

# 提取器配置
EXTRACTORS_CONFIG = {
    'docling': {
        'enabled': True,
        'timeout': 300,
        'options': {
            'do_ocr': True,
            'do_table_structure': True
        }
    },
    'marker': {
        'enabled': True,
        'timeout': 300,
        'options': {
            'output_format': 'markdown',
            'clean_markdown': False
        }
    },
    'mineru': {
        'enabled': True,
        'timeout': 300,
        'options': {
            'parse_mode': 'auto',
            'enable_ocr': True
        }
    }
}
```

## 输出文件

测试完成后，结果将保存在以下位置：

### `results/` 目录
- `detailed_results_YYYYMMDD_HHMMSS.json`: 详细测试结果
- `comparisons_YYYYMMDD_HHMMSS.json`: 提取器比较结果
- `summary_YYYYMMDD_HHMMSS.json`: 汇总统计报告
- `results_YYYYMMDD_HHMMSS.csv`: CSV格式结果
- `report_YYYYMMDD_HHMMSS.html`: HTML报告
- `*_chart_YYYYMMDD_HHMMSS.png`: 可视化图表

### `logs/` 目录
- `pdf_extraction_benchmark_YYYYMMDD_HHMMSS.log`: 执行日志

### `output/` 目录
- 临时文件和中间结果

## 常见问题

### Q: 某个提取器显示"未安装"或"不可用"
A: 请检查：
1. 是否正确安装了对应的Python包
2. 是否有必要的系统依赖
3. 查看错误日志了解具体问题

### Q: 测试运行很慢
A: 建议：
1. 先使用 `--sample-size 5` 进行小规模测试
2. 确保有足够的内存和CPU资源
3. 某些提取器首次运行时需要下载模型

### Q: 内存不足
A: 可以：
1. 减少样本大小
2. 在配置中禁用某些重量级提取器
3. 增加系统虚拟内存

### Q: 生成报告失败
A: 检查：
1. 是否安装了matplotlib和seaborn
2. 是否有写入权限
3. 磁盘空间是否充足

## 扩展开发

### 添加新的提取器

1. 在 `extractors/` 目录创建新的提取器文件
2. 继承 `BasePDFExtractor` 类
3. 实现 `extract()` 和 `is_available()` 方法
4. 在 `extractors/__init__.py` 中导入
5. 在 `config.py` 中添加配置

示例：
```python
from base_extractor import BasePDFExtractor, ExtractionResult

class NewExtractor(BasePDFExtractor):
    def __init__(self, config=None):
        super().__init__("new_extractor", config)
    
    def extract(self, pdf_path):
        # 实现提取逻辑
        pass
    
    def is_available(self):
        # 检查可用性
        pass
```

### 自定义比较指标

在 `benchmark.py` 的 `TextSimilarityCalculator` 类中添加新的相似度计算方法。

### 修改报告格式

编辑 `report_generator.py` 来自定义HTML报告模板或添加新的图表类型。

## 许可证

[根据实际情况填写许可证信息]

## 贡献指南

欢迎提交Issues和Pull Requests来改进这个项目！

## 联系方式

[根据实际情况填写联系方式] 