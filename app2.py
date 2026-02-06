import os
from tempfile import NamedTemporaryFile
import time
from pathlib import Path
import uuid
import re
import base64
import zipfile

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
import pymupdf

import magic_pdf.model as model_config
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.tools.common import do_parse, prepare_env
from magic_pdf.libs.hash_utils import compute_sha256

# 配置模型
model_config.__use_inside_model__ = True

# 创建FastAPI应用
app = FastAPI(title="PDF转Markdown服务", description="提供本地文件上传并转换为Markdown的API")

# 使用环境变量获取日志文件路径，默认为项目根目录下的 app.log
log_file_path = os.getenv('LOG_FILE_PATH', os.path.join(os.getcwd(), 'app.log'))

# 确保目录存在
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# 添加日志处理器
logger.add(log_file_path, rotation="500 MB")


def to_pdf(file_path):
    """将其他格式文件转换为PDF"""
    with pymupdf.open(file_path) as f:
        if f.is_pdf:
            return file_path
        else:
            pdf_bytes = f.convert_to_pdf()
            # 生成唯一的文件名
            unique_filename = f'{uuid.uuid4()}.pdf'
            # 构建完整的文件路径
            tmp_file_path = os.path.join(os.path.dirname(file_path), unique_filename)
            # 将字节数据写入文件
            with open(tmp_file_path, 'wb') as tmp_pdf_file:
                tmp_pdf_file.write(pdf_bytes)
            return tmp_file_path


def read_fn(path):
    """读取文件内容"""
    disk_rw = FileBasedDataReader(os.path.dirname(path))
    return disk_rw.read(os.path.basename(path))


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


def parse_pdf(doc_path, output_dir, end_page_id, is_ocr, layout_mode, formula_enable, table_enable, language):
    """解析PDF文件"""
    os.makedirs(output_dir, exist_ok=True)

    try:
        file_name = f'{str(Path(doc_path).stem)}_{time.time()}'
        pdf_data = read_fn(doc_path)
        if is_ocr:
            parse_method = 'ocr'
        else:
            parse_method = 'auto'
        local_image_dir, local_md_dir = prepare_env(output_dir, file_name, parse_method)
        do_parse(
            output_dir,
            file_name,
            pdf_data,
            [],
            parse_method,
            False,
            end_page_id=end_page_id,
            layout_model=layout_mode,
            formula_enable=formula_enable,
            table_enable=table_enable,
            lang=language,
        )
        return local_md_dir, file_name
    except Exception as e:
        logger.exception(e)


def to_markdown(file_path, end_pages, is_ocr, layout_mode, formula_enable, table_enable, language):
    """将文件转换为Markdown格式"""
    # 获取识别的md文件以及压缩包文件路径
    local_md_dir, file_name = parse_pdf(file_path, './output', end_pages - 1, is_ocr,
                                        layout_mode, formula_enable, table_enable, language)
    archive_zip_path = os.path.join('./output', compute_sha256(local_md_dir) + '.zip')
    zip_archive_success = compress_directory_to_zip(local_md_dir, archive_zip_path)
    if zip_archive_success == 0:
        logger.info('压缩成功')
    else:
        logger.error('压缩失败')
    md_path = os.path.join(local_md_dir, file_name + '.md')
    with open(md_path, 'r', encoding='utf-8') as f:
        txt_content = f.read()
    md_content = txt_content
    # 返回转换后的PDF路径
    new_pdf_path = os.path.join(local_md_dir, file_name + '_layout.pdf')
    return md_content, txt_content, archive_zip_path, new_pdf_path


@app.post('/upload_to_md', tags=['parse_interface'], summary='转换上传文件为Markdown')
async def upload_parse(
        file: UploadFile = File(...),
        max_pages: int = Form(10),  # 默认最大页数为10
        is_ocr: bool = Form(False),
        layout_mode: str = Form('layoutlmv3'),
        formula_enable: bool = Form(True),
        table_enable: bool = Form(False),
        language: str = Form('')
):
    """
    上传文件并转换为Markdown格式
    
    参数:
    - file: 上传的文件
    - max_pages: 最大处理页数（默认10页）
    - is_ocr: 是否使用OCR模式（默认False）
    - layout_mode: 布局模式（默认layoutlmv3）
    - formula_enable: 是否启用公式识别（默认True）
    - table_enable: 是否启用表格识别（默认False）
    - language: 语言设置（默认为空）
    """
    try:
        # 创建临时文件保存上传的文件数据
        temp_file_suffix = file.filename.split('.')[-1]
        temp_file_path = f"/tmp/{uuid.uuid4()}.{temp_file_suffix}"
        with open(temp_file_path, 'wb') as temp_file:
            content = await file.read()
            temp_file.write(content)

        logger.info(f"接收到文件 {file.filename}，临时保存在 {temp_file_path}")

        # 如果不是PDF，则先转换为PDF
        if not temp_file_path.lower().endswith('.pdf'):
            temp_file_path = to_pdf(temp_file_path)
            logger.info(f"非PDF文件已转换为 {temp_file_path}")

        # 调用转换函数
        md_content, txt_content, archive_zip_path, new_pdf_path = to_markdown(
            temp_file_path,
            end_pages=max_pages,
            is_ocr=is_ocr,
            layout_mode=layout_mode,
            formula_enable=formula_enable,
            table_enable=table_enable,
            language=language
        )

        # 清理临时文件
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
