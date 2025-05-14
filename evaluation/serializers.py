from rest_framework import serializers
from .models import EvaluationTask, ModelSelection, EvaluationResult

class ModelSelectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelSelection
        fields = ['id', 'provider_name', 'model_name', 'api_key', 'base_url']
        extra_kwargs = {
            'api_key': {'write_only': True}
        }

class EvaluationResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = EvaluationResult
        fields = ['id', 'provider_identifier', 'accuracy', 'total_tasks', 'correct_tasks']

class EvaluationTaskSerializer(serializers.ModelSerializer):
    models = ModelSelectionSerializer(many=True, read_only=True)
    results = EvaluationResultSerializer(many=True, read_only=True)
    
    class Meta:
        model = EvaluationTask
        fields = ['task_id', 'name', 'status', 'created_at', 'completed_at', 'source_task', 'result_file', 'models', 'results']
        read_only_fields = ['task_id', 'created_at', 'completed_at', 'status']

class TaskCreateSerializer(serializers.ModelSerializer):
    models = ModelSelectionSerializer(many=True, required=False)
    
    class Meta:
        model = EvaluationTask
        fields = ['name', 'source_task', 'models']
    
    def create(self, validated_data):
        models_data = validated_data.pop('models', [])
        task = EvaluationTask.objects.create(**validated_data)
        
        for model_data in models_data:
            ModelSelection.objects.create(task=task, **model_data)
        
        return task