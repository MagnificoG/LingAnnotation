from django.db import models
import json

class JSONField(models.TextField):
    """Custom JSON field for Django 3.0.6 because we cannot import JSONField here"""
    
    def from_db_value(self, value, expression, connection):
        if value is None:
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def to_python(self, value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    def get_prep_value(self, value):
        if value is None:
            return value
        return json.dumps(value)

class DynamicTable(models.Model):
    name = models.CharField(max_length=255)
    json_data = JSONField()  # Use our custom field
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = 'details'

    def __str__(self):
        return self.name
    
    def get_table_data(self):
        """Return normalized table data"""
        data = self.json_data
        
        # Handle different JSON formats
        if isinstance(data, dict):
            # Matrix format: {"columns": [...], "data": [...]}
            if 'columns' in data and 'data' in data:
                return data['columns'], data['data']
            # Object format: {"key": [...]}
            elif len(data) > 0:
                columns = list(data.keys())
                rows = list(zip(*data.values()))
                return columns, rows
        
        # Array of objects format: [{"col1": val1, ...}, ...]
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            columns = list(data[0].keys())
            rows = [list(item.values()) for item in data]
            return columns, rows
        
        # Fallback: treat as single column
        return ['Value'], [[item] for item in data]