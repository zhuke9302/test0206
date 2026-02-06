import os
from tempfile import NamedTemporaryFile
import time
from pathlib import Path
import uuid
import re
import base64
import zipfile
import json

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

from mineru.cli.common import read_fn, prepare_env
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.engine_utils import get_vlm_engine
from mineru.utils.enum_class import MakeMode
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.hybrid.hybrid_analyze import doc_analyze as hybrid_doc_analyze
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox

# 创建FastAPI应用
app = FastAPI(title="PDF转Markdown服务", description="提供本地文件上传并转换为Markdown的API")

# 使用环境变量获取日志文件路径，默认为项目根目录下的 app.log
log_file_path = os.getenv('LOG_FILE_PATH', os.path.join(os.getcwd(), 'app.log'))

# 确保目录存在
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# 添加日志处理器
logger.add(log_file_path, rotation="500 MB")


def compress_directory_to_zip(directory_path, output_zip_path):
    """压缩指定目录到一个 ZIP 文件"""
    try:
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 遍历目录中的所有文件和子目录
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    # 构建完整的文件路径
                    file_path = os.path.join(root, file)
                    # 计算相对路径
                    arcname = os.path.relpath(file_path, directory_path)
                    # 添加文件到 ZIP 文件
                    zipf.write(file_path, arcname)
        return 0
    except Exception as e:
        logger.exception(e)
        return -1


def parse_pdf_with_new_method(
    pdf_bytes,
    output_dir,
    file_name,
    backend="hybrid-auto-engine",
    parse_method="auto",
    language="ch",
    formula_enable=True,
    table_enable=True,
    start_page_id=0,
    end_page_id=None
):
    """使用新方法解析PDF文件"""
    try:
        # 准备环境
        local_image_dir, local_md_dir = prepare_env(output_dir, file_name, parse_method)
        image_writer = FileBasedDataWriter(local_image_dir)
        md_writer = FileBasedDataWriter(local_md_dir)
        
        middle_json = None
        infer_result = None
        
        # 根据backend选择不同的解析方法
        if backend == "pipeline":
            # Pipeline模式
            infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
                [pdf_bytes], 
                [language], 
                parse_method=parse_method,
                formula_enable=formula_enable,
                table_enable=table_enable
            )
            
            model_list = infer_results[0]
            images_list = all_image_lists[0]
            pdf_doc = all_pdf_docs[0]
            _lang = lang_list[0]
            _ocr_enable = ocr_enabled_list[0]
            
            middle_json = pipeline_result_to_middle_json(
                model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable, formula_enable
            )
            infer_result = model_list
            
        elif backend.startswith("vlm-"):
            # VLM模式
            backend_name = backend[4:]
            if backend_name == "auto-engine":
                backend_name = get_vlm_engine(inference_engine='auto', is_async=False)
            
            middle_json, infer_result = vlm_doc_analyze(
                pdf_bytes, 
                image_writer=image_writer, 
                backend=backend_name
            )
            
        elif backend.startswith("hybrid-"):
            # Hybrid模式（默认）
            backend_name = backend[7:]
            if backend_name == "auto-engine":
                backend_name = get_vlm_engine(inference_engine='auto', is_async=False)
            
            middle_json, infer_result, _vlm_ocr_enable = hybrid_doc_analyze(
                pdf_bytes,
                image_writer=image_writer,
                backend=backend_name,
                parse_method=f"hybrid_{parse_method}",
                language=language,
                inline_formula_enable=formula_enable
            )
        else:
            raise ValueError(f"不支持的backend类型: {backend}")
        
        # 获取pdf_info
        pdf_info = middle_json["pdf_info"]
        
        # 绘制布局结果
        try:
            draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{file_name}_layout.pdf")
        except Exception as e:
            logger.warning(f"绘制布局PDF失败: {e}")
        
        # 生成markdown内容
        image_dir = str(os.path.basename(local_image_dir))
        if backend == "pipeline":
            md_content_str = pipeline_union_make(pdf_info, MakeMode.MM_MD, image_dir)
        else:
            md_content_str = vlm_union_make(pdf_info, MakeMode.MM_MD, image_dir)
        
        # 写入markdown文件
        md_writer.write_string(f"{file_name}.md", md_content_str)
        
        # 写入原始PDF
        md_writer.write(f"{file_name}_origin.pdf", pdf_bytes)
        
        # 写入middle_json
        md_writer.write_string(
            f"{file_name}_middle.json",
            json.dumps(middle_json, ensure_ascii=False, indent=4)
        )
        
        # 写入model输出
        if infer_result:
            md_writer.write_string(
                f"{file_name}_model.json",
                json.dumps(infer_result, ensure_ascii=False, indent=4)
            )
        
        logger.info(f"解析完成，输出目录: {local_md_dir}")
        
        return local_md_dir, file_name, md_content_str
        
    except Exception as e:
        logger.exception(e)
        raise


def to_markdown(
    file_path,
    backend="hybrid-auto-engine",
    parse_method="auto",
    language="ch",
    formula_enable=True,
    table_enable=True,
    start_page_id=0,
    end_page_id=None
):
    """将文件转换为Markdown格式（使用新方法）"""
    try:
        # 读取PDF文件
        pdf_bytes = read_fn(file_path)
        
        # 生成唯一的文件名
        base_name = str(Path(file_path).stem)
        file_name = f'{base_name}_{int(time.time())}'
        
        # 使用新方法解析PDF
        local_md_dir, file_name, md_content = parse_pdf_with_new_method(
            pdf_bytes,
            './output',
            file_name,
            backend=backend,
            parse_method=parse_method,
            language=language,
            formula_enable=formula_enable,
            table_enable=table_enable,
            start_page_id=start_page_id,
            end_page_id=end_page_id
        )
        
        # 压缩输出目录
        archive_zip_path = os.path.join('./output', f'{file_name}.zip')
        zip_archive_success = compress_directory_to_zip(local_md_dir, archive_zip_path)
        if zip_archive_success == 0:
            logger.info('压缩成功')
        else:
            logger.error('压缩失败')
        
        # 读取生成的markdown内容（已经在解析过程中生成）
        txt_content = md_content
        
        # 返回转换后的PDF路径
        new_pdf_path = os.path.join(local_md_dir, file_name + '_layout.pdf')
        
        return md_content, txt_content, archive_zip_path, new_pdf_path
        
    except Exception as e:
        logger.exception(e)
        raise


@app.post('/upload_to_md', tags=['parse_interface'], summary='转换上传文件为Markdown')
async def upload_parse(
        file: UploadFile = File(...),
        max_pages: int = Form(10),  # 默认最大页数为10
        is_ocr: bool = Form(False),
        layout_mode: str = Form('layoutlmv3'),
        formula_enable: bool = Form(True),
        table_enable: bool = Form(False),
        language: str = Form('ch')
):
    """
    上传文件并转换为Markdown格式
    
    参数:
    - file: 上传的文件
    - max_pages: 最大处理页数（默认10页）
    - is_ocr: 是否使用OCR模式（默认False，新方法会自动判断）
    - layout_mode: 布局模式（默认layoutlmv3，新方法中此参数影响较小）
    - formula_enable: 是否启用公式识别（默认True）
    - table_enable: 是否启用表格识别（默认False）
    - language: 语言设置（默认ch）
    """
    try:
        # 创建临时文件保存上传的文件数据
        temp_file_suffix = file.filename.split('.')[-1]
        temp_dir = os.getenv('TEMP') if os.name == 'nt' else '/tmp'  # Windows使用TEMP环境变量，Linux使用/tmp
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.{temp_file_suffix}")
        
        with open(temp_file_path, 'wb') as temp_file:
            content = await file.read()
            temp_file.write(content)

        logger.info(f"接收到文件 {file.filename}，临时保存在 {temp_file_path}")

        # 确定解析方法
        if is_ocr:
            parse_method = "ocr"
        else:
            parse_method = "auto"  # 自动判断
        
        # 使用hybrid模式作为默认backend（新方法推荐）
        backend = "hybrid-auto-engine"
        
        # 调用转换函数
        md_content, txt_content, archive_zip_path, new_pdf_path = to_markdown(
            temp_file_path,
            backend=backend,
            parse_method=parse_method,
            language=language if language else 'ch',
            formula_enable=formula_enable,
            table_enable=table_enable,
            start_page_id=0,
            end_page_id=max_pages if max_pages > 0 else None
        )

        # 清理临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        return JSONResponse(content={
            "md_content": md_content,
            "txt_content": txt_content,
            "archive_zip_path": archive_zip_path,
            "new_pdf_path": new_pdf_path,
            "message": "文件转换完成"
        })

    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/', summary='服务状态检查')
async def root():
    """检查服务是否正常运行"""
    return {"status": "运行中", "service": "PDF转Markdown服务"}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8888)