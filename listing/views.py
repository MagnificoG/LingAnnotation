from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import ensure_csrf_cookie
from .models import TaskRecord
from pathlib import Path
import json

def index(request):
    try:
        # 尝试获取所有任务记录
        records = TaskRecord.objects.all().order_by('-task_id')
        # 即使没有记录，也会返回一个空的 QuerySet，不会抛出异常
    except Exception as e:
        # 如果发生异常，记录错误并返回空列表
        print(f"获取任务记录时出错: {str(e)}")
        records = []
    
    # 无论是否有数据，都正常渲染模板
    return render(request, 'listing/index.html', {'records': records})

def task_create(request):
    return render(request, 'listing/task_create.html')

@ensure_csrf_cookie
def create_complete(request):
    if request.method == 'POST':
        # 从请求体中获取JSON数据
        try:
            data = json.loads(request.body)
            task_name = data.get('task_name')
            task_description = data.get('task_description')
            
            # 现在您可以使用task_name和task_description进行后续操作
            # 例如存入数据库等
            task_dir = Path(__file__).parent / '../media/task_dir' / str(task_name)
            if task_dir.exists():
                return JsonResponse({'status': 'error', 'message': f'任务名称{task_name}已存在'})
            task_dir.mkdir(parents=True, exist_ok=True)
            # 在task_dir目录下创建data.json tag.json label.json relation.json等文件
            (task_dir / 'data.json').touch(exist_ok=True)
            (task_dir / 'data.json').write_text(json.dumps([]), encoding='utf-8')
            (task_dir / 'tag.json').touch(exist_ok=True)
            (task_dir / 'tag.json').write_text(json.dumps([]), encoding='utf-8')
            (task_dir / 'label.json').touch(exist_ok=True)
            (task_dir / 'label.json').write_text(json.dumps([]), encoding='utf-8')
            (task_dir / 'relation.json').touch(exist_ok=True)
            (task_dir / 'relation.json').write_text(json.dumps([]), encoding='utf-8')
            # 创建任务记录
            task_record = TaskRecord(
                task_name=task_name,
                task_description=task_description,
                task_dirpath=str(task_dir)
            )
            task_record.save()
            
            # 处理成功后返回成功响应
            return JsonResponse({'status': 'success'})
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': '无效的JSON数据'})
    
    return JsonResponse({'status': 'error', 'message': '只接受POST请求'})

@ensure_csrf_cookie
def task_delete(request):
    if request.method == 'POST':
        # 从请求体中获取JSON数据
        try:
            data = json.loads(request.body)
            task_id = data.get('task_id')
            # 检查task_id是否为整数
            if not isinstance(task_id, int):
                try:
                    task_id = int(task_id)
                except ValueError:
                    return JsonResponse({'status': 'error', 'message': f'无效的任务ID: {task_id}'})
            
            # 查询任务记录
            try:
                task_record = TaskRecord.objects.get(task_id=task_id)
                # 获得task_dirpath字段的文本信息
                task_dirpath: str = task_record.task_dirpath
                # 删除task_dirpath目录
                task_dir = Path(task_dirpath)
                if task_dir.exists() and task_dir.is_dir():
                    # 遍历目录和文件并删除
                    for item in task_dir.iterdir():
                        if item.is_file():
                            item.unlink()
                        else:
                            for sub_item in item.iterdir():
                                sub_item.unlink()
                            item.rmdir()
                    task_dir.rmdir()
                task_record.delete()
                return JsonResponse({'status': 'success'})
            except TaskRecord.DoesNotExist:
                return JsonResponse({'status': 'error', 'message': '任务不存在'})
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': '无效的JSON数据'})
        
    return JsonResponse({'status': 'error', 'message': '只接受POST请求'})