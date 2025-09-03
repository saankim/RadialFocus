
from src.utils import register_model

@register_model('node_regression')
class NodeRegression(...):
    # ...existing code...

@register_model('node_classification')
class NodeClassification(...):
    # ...existing code...

@register_model('graph_classification')
class GraphClassification(...):
    # ...existing code...

@register_model('graph_regression')
class GraphRegression(...):
    # ...existing code...
