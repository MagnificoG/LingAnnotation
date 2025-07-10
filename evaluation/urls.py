from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

app_name = 'evaluation'

# Create a router and register our viewsets
router = DefaultRouter()
router.register(r'api/tasks', views.EvaluationTaskViewSet, basename='task')
router.register(r'api/model-configs', views.ModelConfigurationViewSet, basename='model-config')

urlpatterns = [
    # path('', views.task_list, name='task_list'),
    path('', views.index, name='index'),  # 添加这行作为主页
    # path('api/tasks/', views.api_task_list, name='api_task_list'),  # 修改API路径
    path('task/<int:task_id>/', views.task_detail, name='task_detail'),
    path('tasks/<int:task_id>/stop/', views.stop_task, name='stop_task'),
    path('tasks/<int:task_id>/results/', views.task_results, name='task_results'),
    path('tasks/<int:task_id>/results/<int:result_id>/details/', views.result_details, name='result_details'),
    path('api/upload/', views.upload_file, name='upload_file'),
    path('model-configs/', views.model_config_list, name='model_config_list'),
    path('', include(router.urls)),
]