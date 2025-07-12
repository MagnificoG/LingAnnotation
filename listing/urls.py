from django.urls import path
from . import views
from details import views as details_views

app_name = 'listing'

urlpatterns = [
    # List and create views
    path('', views.index, name='index'),
    path('task_create/', views.task_create, name='task_create'),
    path('create_complete/', views.create_complete, name='create_complete'),
    path('task_delete/', views.task_delete, name='task_delete'),
    
    # Detail views (moved from details app)
    path('<int:task_id>/', details_views.task_detail, name='task_detail'),
    path('<int:task_id>/upload/', details_views.upload_data, name='upload_data'),
    path('<int:task_id>/download/', details_views.download_data, name='download_data'),
    path('<int:task_id>/edit/', details_views.update_task_info, name='update_task_info'),
    path('<int:task_id>/items/delete/', details_views.delete_task_item, name='delete_task_item'),
]