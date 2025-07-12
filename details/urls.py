from django.urls import path
from . import views
from .views import TableDetailView

app_name = 'details'

urlpatterns = [
    path('<int:task_id>/', views.task_detail, name='task_detail'),
    path('upload/<int:task_id>/', views.upload_data, name='upload_data'),
    path('download/<int:task_id>/', views.download_data, name='download_data'),
    path('info_edit/<int:task_id>/', views.update_task_info, name='update_task_info'),
    path('item_delete/', views.delete_task_item, name='delete_task_item'),
    path('tables/<int:pk>/', TableDetailView.as_view(), name='table-detail'),
    path('upload_table/', views.upload_table, name='upload_table'),
]