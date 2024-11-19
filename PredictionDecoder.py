from typing import Dict, Any, List, Union
import tensorflow as tf
import numpy as np

class PredictionDecoder:
    """Handles prediction decoding for various model types"""
    
    KERAS_APPLICATIONS = {
        'resnet': 'tensorflow.keras.applications.resnet50',
        'vgg16': 'tensorflow.keras.applications.vgg16',
        'vgg19': 'tensorflow.keras.applications.vgg19',
        'inception_v3': 'tensorflow.keras.applications.inception_v3',
        'inception_resnet_v2': 'tensorflow.keras.applications.inception_resnet_v2',
        'mobilenet': 'tensorflow.keras.applications.mobilenet',
        'densenet': 'tensorflow.keras.applications.densenet',
        'nasnet': 'tensorflow.keras.applications.nasnet',
        'efficientnet': 'tensorflow.keras.applications.efficientnet',
        'xception': 'tensorflow.keras.applications.xception',
        'convnext': 'tensorflow.keras.applications.convnext',
    }

    def __init__(self):
        self._keras_decode_fn = None
        self._keras_module = None

    def _get_keras_decoder(self, model_name: str):
        """Dynamically imports and caches the appropriate keras decode_predictions function"""
        if self._keras_decode_fn is None:
            # Default to ResNet50 if model_name not specified or not found
            module_path = self.KERAS_APPLICATIONS.get(
                model_name.lower() if model_name else 'resnet',
                'tensorflow.keras.applications.resnet50'
            )
            
            # Import the appropriate module
            module_parts = module_path.split('.')
            self._keras_module = __import__(module_path, fromlist=['decode_predictions'])
            self._keras_decode_fn = self._keras_module.decode_predictions
            
        return self._keras_decode_fn

    @staticmethod
    def decode(
        outputs: Any,
        model_info: Dict[str, Any],
        top_k: int = 1
    ) -> Dict[str, Any]:
        """Create a decoder instance and process the outputs"""
        decoder = PredictionDecoder()
        if model_info['is_huggingface']:
            return decoder._decode_huggingface(outputs, model_info, top_k)
        else:
            return decoder._decode_keras(outputs, model_info, top_k)

    def _decode_keras(
        self,
        outputs: Any,
        model_info: Dict[str, Any],
        top_k: int
    ) -> Dict[str, Any]:
        """Decode Keras model outputs"""
        if model_info['model_type'] == 'image':
            # For native Keras models, outputs are already probabilities
            predictions = outputs
            
            # Get the appropriate decode_predictions function
            keras_decode_predictions = self._get_keras_decoder(model_info.get('model_name'))
            
            # Use keras_decode_predictions for ImageNet models
            decoded_predictions = keras_decode_predictions(predictions.numpy(), top=top_k)
            
            # Format the results
            batch_labels = []
            batch_probs = []
            for batch_preds in decoded_predictions:
                batch_labels.append([pred[1] for pred in batch_preds])  # Class names
                batch_probs.append([pred[2] for pred in batch_preds])   # Probabilities
            
            return {
                'probabilities': predictions.numpy(),
                'predicted_labels': [[pred[0][1]] for pred in decoded_predictions],  # Just the top prediction for compatibility
                'top_k_probabilities': np.array(batch_probs),
                'top_k_labels': batch_labels
            }
        else:
            raise ValueError(f"Unsupported Keras model type: {model_info['model_type']}")

    @staticmethod
    def _decode_huggingface(
        outputs: Any,
        model_info: Dict[str, Any],
        top_k: int
    ) -> Dict[str, Any]:
        """Decode HuggingFace model outputs"""
        if model_info['model_type'] == 'image':
            logits = outputs.logits
            probabilities = tf.nn.softmax(logits, axis=-1)
            # Get top k predictions
            top_k_values, top_k_indices = tf.math.top_k(probabilities, k=top_k)
            
            # Convert indices to labels
            label_names = [
                [model_info['id2label'][idx] for idx in batch_indices]
                for batch_indices in top_k_indices.numpy()
            ]
            
            return {
                'probabilities': probabilities.numpy(),
                'predicted_labels': label_names,
                'top_k_probabilities': top_k_values.numpy(),
                'top_k_labels': label_names
            }
        else:
            # Add handling for other HuggingFace model types here
            raise ValueError(f"Unsupported HuggingFace model type: {model_info['model_type']}")