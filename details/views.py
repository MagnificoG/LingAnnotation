from django.shortcuts import render

# Create your views here.
from . import utils, config
from listing.models import TaskRecord
from django.http import HttpResponse, JsonResponse, FileResponse
from django.views.decorators.csrf import ensure_csrf_cookie
from pathlib import Path
from django.shortcuts import redirect
from django.views.generic import DetailView
from .models import DynamicTable
from .forms import UploadFileForm
from .models import DynamicTable
from .utils import parse_tabular_file

import json
import zipfile
import os
import logging

class TableDetailView(DetailView):
    model = DynamicTable
    template_name = 'details/table_detail.html'
    context_object_name = 'table'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        table = self.get_object()
        columns, data = table.get_table_data()
        context['columns'] = columns
        context['data'] = data
        return context
    
def task_detail(request, task_id):
    # Fetch the task record from the database using the task_id
    task_record = TaskRecord.objects.get(task_id=task_id)
    task_dirpath = task_record.task_dirpath
    data_filepath = Path(task_dirpath) / 'data.json'
    with data_filepath.open("r", encoding='utf-8') as f:
        data = json.load(f)
    
    # Render the template with the task record
    return render(request, 'details/index.html', {'task': task_record, "corpus_items": data})

@ensure_csrf_cookie
def upload_data(request, task_id):
    if request.method == 'POST':
        try:
            logging.info(f"Starting file upload for task_id: {task_id}")
            # Handle the file upload
            task_record = TaskRecord.objects.get(task_id=task_id)
            task_dir = task_record.task_dirpath
            logging.info(f"Task directory: {task_dir}")

            # Assuming the uploaded file is in request.FILES['file']
            uploaded_file = request.FILES.get('file')
            if not uploaded_file:
                logging.error("No file found in request.FILES")
                return JsonResponse({'status': 'error', 'message': 'No file uploaded.'})
            
            logging.info(f"Uploaded file: {uploaded_file.name}")

            # parse the file based on its extension
            parsed_data = utils.parse_file(uploaded_file)
            logging.info(f"Parsed {len(parsed_data)} items from the uploaded file.")

            # Save the parsed data to a file in the task directory
            file_path = Path(task_dir) / 'data.json'
            logging.info(f"Data file path: {file_path}")
            # Handling existing data. Commented for now.
            # with file_path.open("r", encoding='utf-8') as f:
            #     existing_data: list[dict] = json.load(f)
            # max_id = 0
            # for item in existing_data:
            #     if item[config.ID] > max_id:
            #         max_id = item[config.ID]
            # for i, item in enumerate(parsed_data, start=max_id):
            #     item[config.ID] = i + 1
            # existing_data.extend(parsed_data)
            with file_path.open("w", encoding='utf-8') as f:
                json.dump(parsed_data, f, ensure_ascii=False, indent=4)

            logging.info("Successfully saved data to data.json")
            return JsonResponse({'status': 'success'})  # Return a success response after processing the upload
        except Exception as e:
            logging.error(f"An error occurred during file upload: {e}", exc_info=True)
            return JsonResponse({'status': 'error', 'message': f"文件上传出现错误: {str(e)}"})
    return JsonResponse({'status': 'error', 'message': '只接受POST请求'})

@ensure_csrf_cookie
def download_data(request, task_id):
    try:
        # 获取任务记录
        task_record = TaskRecord.objects.get(task_id=task_id)
        task_dir = task_record.task_dirpath
        task_name = task_record.task_name.replace(" ", "_")  # 替换空格，用于文件名
    
        # 创建zip文件路径
        zip_file_path = Path(task_dir) / 'data.zip'
        
        # 创建zip文件
        with zipfile.ZipFile(zip_file_path, 'w') as zipf:
            for root, dirs, files in os.walk(task_dir):
                for file in files:
                    # 跳过zip文件本身
                    if file == 'data.zip':
                        continue
                    file_path = Path(root) / file
                    # 将文件添加到zip中，保留相对路径
                    zipf.write(file_path, file_path.relative_to(task_dir))
        
        # 确认文件已生成
        if not os.path.exists(zip_file_path):
            return JsonResponse({'status': 'error', 'message': '压缩文件生成失败'})
        
        # 设置文件名
        filename = f"task_{task_id}_{task_name}.zip"
        
        # 打开文件并返回
        with open(zip_file_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='application/zip')
            response['Content-Disposition'] = f'attachment; filename="{filename}"'
            return response
            
    except TaskRecord.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': f'找不到任务ID: {task_id}'})
    except Exception as e:
        # 详细记录异常信息
        import traceback
        error_details = traceback.format_exc()
        return JsonResponse({'status': 'error', 'message': f"文件下载失败: {str(e)}", 'details': error_details})

@ensure_csrf_cookie
def update_task_info(request, task_id):
    # Fetch the task record from the database using the task_id
    task_record = TaskRecord.objects.get(task_id=task_id)
    task_name = task_record.task_name
    task_description = task_record.task_description
    return render(request, 'details/update_info.html', {
        'task_id': task_id,
        'task_name': task_name,
        'task_description': task_description,
    })

@ensure_csrf_cookie
def delete_task_item(request):
    if request.method == 'POST':
        try:
            # 从请求体中获取JSON数据
            data = json.loads(request.body)
            task_id = data.get('task_id')
            item_id = data.get('item_id')
            print(task_id, item_id)
            
            # 检查task_id和item_id是否为整数
            if not isinstance(task_id, int) or not isinstance(item_id, int):
                return JsonResponse({'status': 'error', 'message': '无效的任务ID或条目ID'})
            
            # 查询任务记录
            task_record = TaskRecord.objects.get(task_id=task_id)
            task_dirpath = task_record.task_dirpath
            
            # 删除条目
            file_path = Path(task_dirpath) / 'data.json'
            with file_path.open("r", encoding='utf-8') as f:
                data = json.load(f)
            
            # 删除指定ID的条目
            data = [item for item in data if item[config.ID] != item_id]
            
            # 保存更新后的数据
            with file_path.open("w", encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            return JsonResponse({'status': 'success'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f"删除条目失败: {str(e)}"})
    return JsonResponse({'status': 'error', 'message': '只接受POST请求'})

def upload_table(request, task_id):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            try:
                json_data = parse_tabular_file(file)
                table_name = Path(file.name).stem
                table = DynamicTable.objects.create(name=table_name, json_data=json_data)
                return redirect('details:table-detail', pk=table.pk)
            except ValueError as e:
                form.add_error('file', str(e))
    else:
        form = UploadFileForm()
    return render(request, 'details/upload_table.html', {'form': form})