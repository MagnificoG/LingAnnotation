from django.db import models

# Create your models here.
"""
class DataRecord(models.Model):
    text = models.TextField(verbose_name="文本内容")
    label1 = models.CharField(max_length=255, blank=True, verbose_name="标注1")
    label2 = models.CharField(max_length=255, blank=True, verbose_name="标注2")
    label3 = models.CharField(max_length=255, blank=True, verbose_name="标注3")
    label4 = models.CharField(max_length=255, blank=True, verbose_name="标注4")
    llm_answer = models.TextField(blank=True, verbose_name="大模型答案")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "数据记录"
        verbose_name_plural = "数据记录"
"""
class TaskRecord(models.Model):
    task_id = models.AutoField(primary_key=True, verbose_name="任务ID")
    task_name = models.CharField(max_length=255, verbose_name="任务名称")
    task_description = models.TextField(verbose_name="任务描述", default="")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建日期")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新日期")
    task_dirpath = models.CharField(max_length=255, verbose_name="任务文件路径", default="")