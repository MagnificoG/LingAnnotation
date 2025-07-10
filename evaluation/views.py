# Django 核心导入
from django.shortcuts import render
from django.conf import settings
from django.utils import timezone
from django.views.generic import TemplateView
from django.contrib import messages

# Django REST framework 导入
from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response

# 本地应用导入
from .models import EvaluationTask, ModelSelection, EvaluationResult, ModelConfiguration
from .serializers import (
    EvaluationTaskSerializer, 
    TaskCreateSerializer,
    ModelConfigurationSerializer
)
from listing.models import TaskRecord

# 系统库导入
import os
import uuid
import threading
import json

from .services.evaluation_service import (
    DataLoader,
    DataTransformer,
    LLMProvider,
    Evaluator,
    EvaluationRunner
)

# 用于渲染模板的视图
def index(request):
    # Fetch all available annotation tasks from the listing module
    annotation_tasks = TaskRecord.objects.all().order_by('-created_at')
    evaluation_tasks = EvaluationTask.objects.all().order_by('-created_at')
    
    return render(request, 'evaluation/index.html', {
        'annotation_tasks': annotation_tasks,
        'tasks': evaluation_tasks
    })

def model_config_list(request):
    """View for listing and managing model configurations"""
    try:
        # Try to get configurations
        model_configs = ModelConfiguration.objects.all().order_by('-created_at')
    except Exception as e:
        # Handle any database errors (including missing tables)
        model_configs = []
        if not request.session.get('db_error_shown'):
            messages.warning(request, 'Database setup required. Please run migrations first: python manage.py migrate')
            request.session['db_error_shown'] = True
    
    return render(request, 'evaluation/model_config_list.html', {
        'model_configs': model_configs,
        'providers': [
            {'id': 'openai', 'name': 'OpenAI'},
            {'id': 'anthropic', 'name': 'Anthropic'},
            {'id': 'google', 'name': 'Google'},
            {'id': 'other', 'name': 'Other'}
        ]
    })

class ModelConfigurationViewSet(viewsets.ModelViewSet):
    """API endpoint for managing model configurations"""
    queryset = ModelConfiguration.objects.all().order_by('-created_at')
    serializer_class = ModelConfigurationSerializer

class EvaluationTaskViewSet(viewsets.ModelViewSet):
    queryset = EvaluationTask.objects.all().order_by('-created_at')
    parser_classes = (MultiPartParser, FormParser)
    
    def get_serializer_class(self):
        if self.action == 'create':
            return TaskCreateSerializer
        return EvaluationTaskSerializer
    
    def create(self, request, *args, **kwargs):
        # Parse models data from JSON string
        models_data = json.loads(request.data.get('models', '[]'))
        
        # Create mutable copy of request data
        data = request.data.copy()
        data['models'] = models_data
        
        serializer = self.get_serializer(data=data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)
    
    @action(detail=True, methods=['post'])
    def start_evaluation(self, request, pk=None):
        task = self.get_object()
        
        if task.status != 'pending':
            return Response({"error": "只有等待中的任务可以启动"}, status=status.HTTP_400_BAD_REQUEST)
        
        if not task.models.exists():
            return Response({"error": "请至少选择一个模型"}, status=status.HTTP_400_BAD_REQUEST)
        
        # 更新任务状态
        task.status = 'running'
        task.save()
        
        # 在后台线程中运行评测
        threading.Thread(target=self._run_evaluation, args=(task,)).start()
        
        return Response({"message": "评测任务已启动"})
    
    def _run_evaluation(self, task):
        try:
            # 准备评测所需的组件
            data_loader = DataLoader(task)
            data_transformer = DataTransformer()
            evaluator = Evaluator()
            
            # 准备LLM提供商
            providers = []
            for model in task.models.all():
                provider = LLMProvider(
                    provider_name=model.provider_name,
                    model_name=model.model_name,
                    api_key=model.api_key,
                    base_url=model.base_url if model.base_url else None
                )
                providers.append(provider)
            
            # 设置结果文件路径
            timestamp = timezone.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"evaluation_results_{task.id}_{timestamp}.json"
            result_path = os.path.join(settings.MEDIA_ROOT, 'evaluation_results', result_filename)
            
            # 确保目录存在
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            
            # 创建并运行评测
            runner = EvaluationRunner(
                data_loader=data_loader,
                data_transformer=data_transformer,
                providers=providers,
                evaluator=evaluator,
                output_json_path=result_path,
                checkpoint_interval=10,
                concurrency_limit=5
            )
            
            # 运行评测并获取结果
            provider_accuracies, detailed_results = runner.run_evaluation()
            
            # 更新任务状态和结果
            task.status = 'completed'
            task.completed_at = timezone.now()
            task.result_file = f"evaluation_results/{result_filename}"
            task.save()
            
            # 保存评测结果
            if provider_accuracies:
                for provider_id, accuracy in provider_accuracies.items():
                    # 计算正确任务数和总任务数
                    correct_count = 0
                    total_count = 0
                    
                    for result in detailed_results:
                        if result.get('provider_identifier') == provider_id and result.get('is_correct') is not None:
                            total_count += 1
                            if result.get('is_correct'):
                                correct_count += 1
                    
                    # 创建结果记录
                    result_obj = EvaluationResult.objects.create(
                        task=task,
                        provider_identifier=provider_id,
                        accuracy=accuracy,
                        total_tasks=total_count,
                        correct_tasks=correct_count
                    )
                    
                    # 保存详细结果数据
                    provider_results = [r for r in detailed_results if r.get('provider_identifier') == provider_id]
                    result_obj.set_result_data(provider_results)
                    result_obj.save()
            
        except Exception as e:
            # 如果发生错误，更新任务状态
            task.status = 'failed'
            task.save()
            print(f"评测任务 {task.id} 失败: {str(e)}")

# 修改文件上传视图以使用Django风格的响应
@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def upload_file(request):
    if 'file' not in request.FILES:
        return Response({"error": "未提供文件"}, status=400)
    
    file = request.FILES['file']
    
    # 验证文件类型
    if not file.name.endswith(('.xlsx', '.xls')):
        return Response({"error": "仅支持Excel文件"}, status=400)
    
    filename = f"{uuid.uuid4()}_{file.name}"
    file_path = os.path.join(settings.MEDIA_ROOT, 'evaluation_files', filename)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 保存文件
    with open(file_path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    
    # 返回相对路径
    relative_path = os.path.join('evaluation_files', filename)
    
    return Response({
        "message": "文件上传成功",
        "file_path": relative_path
    })

# 修改API视图以使用Django原生的JsonResponse
@api_view(['GET', 'POST'])
def task_list(request):
    if request.method == 'GET':
        tasks = EvaluationTask.objects.all().values(
            'task_id', 'name', 'status', 'created_at'
        ).order_by('-created_at')
        return Response(list(tasks))
    elif request.method == 'POST':
        serializer = TaskCreateSerializer(data=request.data)
        if serializer.is_valid():
            task = serializer.save()
            return Response(
                TaskCreateSerializer(task).data, 
                status=status.HTTP_201_CREATED
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET', 'DELETE'])
def task_detail(request, task_id):
    try:
        task = EvaluationTask.objects.get(pk=task_id)
    except EvaluationTask.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method == 'GET':
        serializer = EvaluationTaskSerializer(task)
        return Response(serializer.data)
    elif request.method == 'DELETE':
        task.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

@api_view(['POST'])
def stop_task(request, task_id):
    try:
        task = EvaluationTask.objects.get(pk=task_id)
        task.status = 'failed'
        task.save()
        return Response({'status': 'stopped'})
    except EvaluationTask.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

# 其他视图保持不变
@action(detail=True, methods=['get'])
def task_results(request, task_id):
    """获取任务结果"""
    try:
        task = EvaluationTask.objects.get(pk=task_id)
        results = task.results.all()
        return Response([{
            'id': result.id,
            'provider_identifier': result.provider_identifier,
            'accuracy': result.accuracy,
            'correct_tasks': result.correct_tasks,
            'total_tasks': result.total_tasks
        } for result in results])
    except EvaluationTask.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

@action(detail=True, methods=['get'])
def result_details(request, task_id, result_id):
    """获取结果详情"""
    try:
        result = EvaluationResult.objects.get(task_id=task_id, id=result_id)
        return Response(result.get_result_data())
    except EvaluationResult.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

# 数据集管理相关视图
@api_view(['GET', 'POST'])
def dataset_list(request):
    """获取数据集列表或创建新数据集"""
    if request.method == 'GET':
        # 实现获取数据集列表的逻辑
        datasets = [
            {
                'id': 1,
                'name': 'Zeugma',
                'type': '评测集-文本生成',
                'version': 'V1',
                'count': 300,
                'import_status': 'success',
                'publish_status': 'published',
                'updated_at': '2025-02-09 22:07:41'
            }
        ]
        return Response(datasets)
    elif request.method == 'POST':
        # 实现创建数据集的逻辑
        return Response({'id': 1, 'name': request.data.get('name')}, status=status.HTTP_201_CREATED)

@api_view(['GET', 'DELETE'])
def dataset_detail(request, dataset_id):
    """获取、删除数据集详情"""
    if request.method == 'GET':
        # 实现获取数据集详情的逻辑
        dataset = {
            'id': dataset_id,
            'name': 'Zeugma',
            'type': '评测集-文本生成',
            'version': 'V1',
            'count': 300,
            'import_status': 'success',
            'publish_status': 'published',
            'created_at': '2025-02-09 22:00:00',
            'updated_at': '2025-02-09 22:07:41',
            'description': '这是一个用于评测大语言模型文本生成能力的数据集。'
        }
        return Response(dataset)
    elif request.method == 'DELETE':
        # 实现删除数据集的逻辑
        return Response(status=status.HTTP_204_NO_CONTENT)

@api_view(['GET'])
def dataset_items(request, dataset_id):
    """获取数据集内容"""
    # 实现获取数据集内容的逻辑
    items = [
        {'id': i, 'question': f'示例问题 {i}', 'answer': f'示例答案 {i}'}
        for i in range(1, 11)
    ]
    return Response({
        'count': 300,
        'results': items
    })

@api_view(['POST'])
def publish_dataset(request, dataset_id):
    """发布数据集"""
    # 实现发布数据集的逻辑
    return Response({'status': 'published'})

@api_view(['POST'])
def update_dataset(request, dataset_id):
    """更新数据集"""
    # 实现更新数据集的逻辑
    return Response({'status': 'updated', 'version': 'V2'})

@api_view(['POST'])
def upload_dataset_file(request):
    """上传数据集文件"""
    # 实现上传数据集文件的逻辑
    return Response({'file_path': 'datasets/example.xlsx'})
