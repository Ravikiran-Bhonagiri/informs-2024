from Models import (
    NumericalOnlyModel,
    CategoricalOnlyModel,
    NLPOnlyModel,
    NumericalCategoricalModel,
    NumericalNLPModel,
    CategoricalNLPModel,
    MultiInputModel,
    WideAndDeepModel
)
experiment_configs = [
    {
        'name': 'Numerical Only 1',
        'model_class': NumericalOnlyModel,
        'numerical_cols': ['author_follower_count', 'author_following_count', 'author_total_video_count', 'author_total_video_count'],
        'categorical_cols': [],
        'nlp_cols': [],
        'model_params': {
            'hidden_sizes': [128, 64],
            'dropout': 0.1
        }
    }
]