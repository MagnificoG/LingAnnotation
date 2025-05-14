from django.urls import path
from . import views

app_name = 'evaluation'

urlpatterns = [
    # path('', views.task_list, name='task_list'),
    path('', views.index, name='index'),  # 添加这行作为主页
    # path('api/tasks/', views.api_task_list, name='api_task_list'),  # 修改API路径
    path('tasks/<int:task_id>/', views.task_detail, name='task_detail'),
    path('tasks/<int:task_id>/stop/', views.stop_task, name='stop_task'),
    path('tasks/<int:task_id>/results/', views.task_results, name='task_results'),
    path('tasks/<int:task_id>/results/<int:result_id>/details/', views.result_details, name='result_details'),
    path('upload/', views.upload_file, name='upload_file'),
]