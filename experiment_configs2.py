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
    },
    {
        'name': 'Numerical Only 2',
        'model_class': NumericalOnlyModel,
        'numerical_cols': ['author_follower_count', 'author_following_count', 'author_total_video_count',
                           'author_total_video_count', 'follower_following_ratio', 'avg_hearts_per_video', 
                           'Engagement_Rate', 'videos_per_follower', 'Video_Length'],
        'categorical_cols': [],
        'nlp_cols': [],
        'model_params': {
            'hidden_sizes': [128, 64],
            'dropout': 0.1
        }
    },
    {
        'name': 'Numerical Only 3',
        'model_class': NumericalOnlyModel,
        'numerical_cols': ['zero_crossing_rate_mean', 'mfcc_1', 'mfcc_2',
                           'mfcc_3', 'mfcc_4', 'mfcc_5', 
                           'mfcc_6', 'mfcc_7', 'mfcc_8',
                           'mfcc_9', 'mfcc_10', 'mfcc_11',
                           'mfcc_12', 'mfcc_13', 'spectral_contrast_1',
                           'spectral_contrast_1', 'spectral_contrast_2', 'spectral_contrast_3',
                           'spectral_contrast_4', 'spectral_contrast_5',
                           'spectral_contrast_6', 'spectral_contrast_7', 'chroma_1',
                           'chroma_2', 'chroma_3','chroma_4',
                           'chroma_5', 'chroma_6', 'chroma_7',
                           'chroma_8', 'chroma_9', 'chroma_10', 'chroma_11', 'chroma_12'],
        'categorical_cols': [],
        'nlp_cols': [],
        'model_params': {
            'hidden_sizes': [128, 64],
            'dropout': 0.1
        }
    },
    {
        'name': 'Categorical Only',
        'model_class': CategoricalOnlyModel,
        'numerical_cols': [],
        'categorical_cols': ['Collaborations', 'Series', 'Post_Day', 'Post_Month', 'Post_Season', 'Post_Quarter', 'Post_Part_of_Day', 'Is_Weekend', 'Next_Day_Holiday', 'Is_Long_Weekend',
                             'industry', 'Sentiment Score', 'Speech vs. Action Focus', 'Setting', 'Interaction Type', 'Tone', 'Dialogue/Monologue', 'Audience Engagement'],
        'nlp_cols': [],
        'model_params': {
            'embed_dim': 16,
            'hidden_sizes': [128, 64],
            'dropout': 0.1
        }
    },
    {
        'name': 'NLP Only',
        'model_class': NLPOnlyModel,
        'numerical_cols': [],
        'categorical_cols': [],
        'nlp_cols': ['video_description', 'transcribe_text' ,'generated_vlm_text','llm_response','processed_response'],
        'model_params': {
            'embed_size': 256,
            'nlp_encoder_type': 'lstm',  # Options: 'lstm', 'transformer'
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.1
        }
    },
    {
        'name': 'Numerical and Categorical',
        'model_class': NumericalCategoricalModel,
        'numerical_cols': ['author_follower_count', 'author_following_count', 'author_total_video_count',
                           'author_total_video_count', 'follower_following_ratio', 'avg_hearts_per_video', 
                           'Engagement_Rate', 'videos_per_follower', 'Video_Length', 'zero_crossing_rate_mean', 'mfcc_1', 'mfcc_2',
                           'mfcc_3', 'mfcc_4', 'mfcc_5', 
                           'mfcc_6', 'mfcc_7', 'mfcc_8',
                           'mfcc_9', 'mfcc_10', 'mfcc_11',
                           'mfcc_12', 'mfcc_13', 'spectral_contrast_1',
                           'spectral_contrast_1', 'spectral_contrast_2', 'spectral_contrast_3',
                           'spectral_contrast_4', 'spectral_contrast_5',
                           'spectral_contrast_6', 'spectral_contrast_7', 'chroma_1',
                           'chroma_2', 'chroma_3','chroma_4',
                           'chroma_5', 'chroma_6', 'chroma_7',
                           'chroma_8', 'chroma_9', 'chroma_10', 'chroma_11', 'chroma_12'
                           ],
        'categorical_cols': ['Collaborations', 'Series', 'Post_Day', 'Post_Month', 'Post_Season', 'Post_Quarter', 'Post_Part_of_Day', 'Is_Weekend', 'Next_Day_Holiday', 'Is_Long_Weekend',
                           'industry', 'Sentiment Score', 'Speech vs. Action Focus', 'Setting', 'Interaction Type', 'Tone', 'Dialogue/Monologue', 'Audience Engagement'],
        'nlp_cols': [],
        'model_params': {
            'hidden_sizes': [128, 64],
            'dropout': 0.1
        }
    },
    {
        'name': 'Numerical and NLP',
        'model_class': NumericalNLPModel,
        'numerical_cols': ['author_follower_count', 'author_following_count', 'author_total_video_count',
                           'author_total_video_count', 'follower_following_ratio', 'avg_hearts_per_video', 
                           'Engagement_Rate', 'videos_per_follower', 'Video_Length', 'zero_crossing_rate_mean', 'mfcc_1', 'mfcc_2',
                           'mfcc_3', 'mfcc_4', 'mfcc_5', 
                           'mfcc_6', 'mfcc_7', 'mfcc_8',
                           'mfcc_9', 'mfcc_10', 'mfcc_11',
                           'mfcc_12', 'mfcc_13', 'spectral_contrast_1',
                           'spectral_contrast_1', 'spectral_contrast_2', 'spectral_contrast_3',
                           'spectral_contrast_4', 'spectral_contrast_5',
                           'spectral_contrast_6', 'spectral_contrast_7', 'chroma_1',
                           'chroma_2', 'chroma_3','chroma_4',
                           'chroma_5', 'chroma_6', 'chroma_7',
                           'chroma_8', 'chroma_9', 'chroma_10', 'chroma_11', 'chroma_12'],
        'categorical_cols': [],
        'nlp_cols': ['video_description', 'transcribe_text' ,'generated_vlm_text','llm_response','processed_response'],
        'model_params': {
            'embed_size': 256,
            'nlp_encoder_type': 'transformer',  # Options: 'lstm', 'transformer'
            'hidden_size': 128,
            'num_layers': 2,
            'hidden_sizes': [128, 64],
            'dropout': 0.1,
        }
    },
    {
        'name': 'Categorical and NLP',
        'model_class': CategoricalNLPModel,
        'numerical_cols': [],
        'categorical_cols': ['Collaborations', 'Series', 'Post_Day', 'Post_Month', 'Post_Season', 'Post_Quarter', 'Post_Part_of_Day', 'Is_Weekend', 'Next_Day_Holiday', 'Is_Long_Weekend',
                             'industry', 'Sentiment Score', 'Speech vs. Action Focus', 'Setting', 'Interaction Type', 'Tone', 'Dialogue/Monologue', 'Audience Engagement'],
        'nlp_cols': ['video_description', 'transcribe_text' ,'generated_vlm_text','llm_response','processed_response'],
        'model_params': {
            'embed_size': 256,
            'nlp_encoder_type': 'transformer',  # Options: 'lstm', 'transformer'
            'hidden_size': 128,
            'num_layers': 2,
            'hidden_sizes': [128, 64],
            'dropout': 0.1,
        }
    },
    {
        'name': 'All Features (Multi-Input)',
        'model_class': MultiInputModel,
        'numerical_cols': ['author_follower_count', 'author_following_count', 'author_total_video_count',
                           'author_total_video_count', 'follower_following_ratio', 'avg_hearts_per_video', 
                           'Engagement_Rate', 'videos_per_follower', 'Video_Length', 'zero_crossing_rate_mean', 'mfcc_1', 'mfcc_2',
                           'mfcc_3', 'mfcc_4', 'mfcc_5', 
                           'mfcc_6', 'mfcc_7', 'mfcc_8',
                           'mfcc_9', 'mfcc_10', 'mfcc_11',
                           'mfcc_12', 'mfcc_13', 'spectral_contrast_1',
                           'spectral_contrast_1', 'spectral_contrast_2', 'spectral_contrast_3',
                           'spectral_contrast_4', 'spectral_contrast_5',
                           'spectral_contrast_6', 'spectral_contrast_7', 'chroma_1',
                           'chroma_2', 'chroma_3','chroma_4',
                           'chroma_5', 'chroma_6', 'chroma_7',
                           'chroma_8', 'chroma_9', 'chroma_10', 'chroma_11', 'chroma_12'],
        'categorical_cols': ['Collaborations', 'Series', 'Post_Day', 'Post_Month', 'Post_Season', 'Post_Quarter', 'Post_Part_of_Day', 'Is_Weekend', 'Next_Day_Holiday', 'Is_Long_Weekend',
                             'industry', 'Sentiment Score', 'Speech vs. Action Focus', 'Setting', 'Interaction Type', 'Tone', 'Dialogue/Monologue', 'Audience Engagement'],
        'nlp_cols': ['video_description', 'transcribe_text' ,'generated_vlm_text','llm_response','processed_response'],
        'model_params': {
            'embed_size': 256,
            'nlp_feature_configs': [
                {'type': 'transformer', 'hidden_size': 128, 'num_layers': 2},
                {'type': 'transformer', 'num_layers': 2, 'heads': 8, 'forward_expansion': 4}
            ],
            'shared_layer_sizes': [256, 128],
            'dropout': 0.1
        }
    },
    {
        'name': 'Wide and Deep Model',
        'model_class': WideAndDeepModel,
        'numerical_cols': ['author_follower_count', 'author_following_count', 'author_total_video_count',
                           'author_total_video_count', 'follower_following_ratio', 'avg_hearts_per_video', 
                           'Engagement_Rate', 'videos_per_follower', 'Video_Length', 'zero_crossing_rate_mean', 'mfcc_1', 'mfcc_2',
                           'mfcc_3', 'mfcc_4', 'mfcc_5', 
                           'mfcc_6', 'mfcc_7', 'mfcc_8',
                           'mfcc_9', 'mfcc_10', 'mfcc_11',
                           'mfcc_12', 'mfcc_13', 'spectral_contrast_1',
                           'spectral_contrast_1', 'spectral_contrast_2', 'spectral_contrast_3',
                           'spectral_contrast_4', 'spectral_contrast_5',
                           'spectral_contrast_6', 'spectral_contrast_7', 'chroma_1',
                           'chroma_2', 'chroma_3','chroma_4',
                           'chroma_5', 'chroma_6', 'chroma_7',
                           'chroma_8', 'chroma_9', 'chroma_10', 'chroma_11', 'chroma_12'],
        'categorical_cols': ['Collaborations', 'Series', 'Post_Day', 'Post_Month', 'Post_Season', 'Post_Quarter', 'Post_Part_of_Day', 'Is_Weekend', 'Next_Day_Holiday', 'Is_Long_Weekend',
                             'industry', 'Sentiment Score', 'Speech vs. Action Focus', 'Setting', 'Interaction Type', 'Tone', 'Dialogue/Monologue', 'Audience Engagement'],
        'nlp_cols': ['video_description', 'transcribe_text' ,'generated_vlm_text','llm_response','processed_response'],
        'model_params': {
            'embed_size': 256,
            'nlp_feature_config': [
                {'type': 'transformer', 'num_layers': 2, 'heads': 8, 'forward_expansion': 4},
                {'type': 'transformer', 'num_layers': 2, 'heads': 8, 'forward_expansion': 4},
                {'type': 'transformer', 'num_layers': 2, 'heads': 8, 'forward_expansion': 4},
                {'type': 'transformer', 'num_layers': 2, 'heads': 8, 'forward_expansion': 4},
                {'type': 'transformer', 'num_layers': 2, 'heads': 8, 'forward_expansion': 4}
            ],
            'deep_layer_sizes': [256, 128],
            'dropout': 0.1,
            'max_length': 256
        }
    },
]