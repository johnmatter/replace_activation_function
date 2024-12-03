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
            if model_name:
                # Find the matching module path
                base_name = next(
                    (
                        key for key in self.KERAS_APPLICATIONS.keys()
                        if model_name.lower().startswith(key)
                    ),
                    None
                )
                if base_name is None:
                    raise ValueError(f"Unsupported Keras model: {model_name}")
                    
                # Get the full module path from KERAS_APPLICATIONS
                module_path = self.KERAS_APPLICATIONS[base_name]
                
                # Import the appropriate module
                module_parts = module_path.split('.')
                base_module = __import__(module_parts[0])
                for part in module_parts[1:]:
                    base_module = getattr(base_module, part)
                
                self._keras_decode_fn = base_module.decode_predictions
                
            else:
                raise ValueError("model_name must be specified")
                
        return self._keras_decode_fn

    @staticmethod
    def decode(
        outputs: Any,
        model_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a decoder instance and process the outputs"""
        decoder = PredictionDecoder()
        top_k = model_info.get('top_k', None)

        if model_info['is_huggingface']:
            return decoder._decode_huggingface(outputs, model_info, top_k)
        else:
            return decoder._decode_keras(outputs, model_info, top_k)

    def _decode_keras(
        self,
        outputs: Any,
        model_info: Dict[str, Any],
        top_k: int = None
    ) -> Dict[str, Any]:
        """Decode Keras model outputs"""
        if model_info['model_type'] == 'image':
            # For native Keras models, outputs are already probabilities
            predictions = outputs
            
            # Get the appropriate decode_predictions function
            keras_decode_predictions = self._get_keras_decoder(model_info.get('model_name'))
            
            # Decode predictions
            if top_k:
                decoded_predictions = keras_decode_predictions(predictions.numpy(), top=top_k)
            else:
                decoded_predictions = keras_decode_predictions(predictions.numpy())
            
            # Format the results
            batch_labels = []
            batch_probs = []
            for batch_preds in decoded_predictions:
                labels = [pred[1] for pred in batch_preds]
                probs = [pred[2] for pred in batch_preds]
                batch_labels.append(labels)
                batch_probs.append(probs)
            
            return {
                'probabilities': predictions.numpy(),
                'predicted_labels': batch_labels,
                'top_k_probabilities': np.array(batch_probs),
                'top_k_labels': batch_labels
            }
        else:
            raise ValueError(f"Unsupported Keras model type: {model_info['model_type']}")

    @staticmethod
    def _decode_huggingface(
        outputs: Any,
        model_info: Dict[str, Any],
        top_k: int = None
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