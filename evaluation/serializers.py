from rest_framework import serializers
from .models import EvaluationTask, ModelSelection, EvaluationResult, ModelConfiguration

class ModelConfigurationSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelConfiguration
        fields = ['id', 'name', 'provider_name', 'model_name', 'api_key', 'base_url', 'is_active']
        extra_kwargs = {
            'api_key': {'write_only': True}
        }

class ModelSelectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelSelection
        fields = ['id', 'configuration']

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
    model_configurations = serializers.PrimaryKeyRelatedField(
        many=True,
        queryset=ModelConfiguration.objects.filter(is_active=True),
        source='models'
    )
    
    class Meta:
        model = EvaluationTask
        fields = ['name', 'source_task', 'model_configurations']
    
    def create(self, validated_data):
        model_configs = validated_data.pop('models', [])
        
        # Create the task
        task = EvaluationTask.objects.create(**validated_data)
        
        # Create associated model selections
        for config in model_configs:
            ModelSelection.objects.create(task=task, configuration=config)
        
        return task