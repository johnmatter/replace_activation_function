# Iterate over models in tensorflow.kers.applications and pretty-print the model summary

import tensorflow as tf
from figlet_color import JojjjajjrPrettyPrint

# https://www.tensorflow.org/api_docs/python/tf/keras/applications

model_modules = ['convnext', 'densenet', 'efficientnet', 'efficientnet_v2', 'imagenet_utils', 'inception_resnet_v2', 'inception_v3', 'mobilenet', 'mobilenet_v2', 'mobilenet_v3', 'nasnet', 'regnet', 'resnet', 'resnet50', 'resnet_rs', 'resnet_v2', 'vgg16', 'vgg19', 'xception']

model_subtypes = ['ConvNeXtBase', 'ConvNeXtLarge', 'ConvNeXtSmall', 'ConvNeXtTiny', 'ConvNeXtXLarge', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7', 'EfficientNetV2B0', 'EfficientNetV2B1', 'EfficientNetV2B2', 'EfficientNetV2B3', 'EfficientNetV2L', 'EfficientNetV2M', 'EfficientNetV2S', 'InceptionResNetV2', 'InceptionV3', 'MobileNet', 'MobileNetV2', 'MobileNetV3Large', 'MobileNetV3Small', 'NASNetLarge', 'NASNetMobile', 'RegNetX002', 'RegNetX004', 'RegNetX006', 'RegNetX008', 'RegNetX016', 'RegNetX032', 'RegNetX040', 'RegNetX064', 'RegNetX080', 'RegNetX120', 'RegNetX160', 'RegNetX320', 'RegNetY002', 'RegNetY004', 'RegNetY006', 'RegNetY008', 'RegNetY016', 'RegNetY032', 'RegNetY040', 'RegNetY064', 'RegNetY080', 'RegNetY120', 'RegNetY160', 'RegNetY320', 'ResNet101', 'ResNet101V2', 'ResNet152', 'ResNet152V2', 'ResNet50', 'ResNet50V2', 'ResNetRS101', 'ResNetRS152', 'ResNetRS200', 'ResNetRS270', 'ResNetRS350', 'ResNetRS420', 'ResNetRS50', 'VGG16', 'VGG19', 'Xception']

fixed_gradient_config = {
    'style': 'multi_gradient',
    'colors': [(1,0,1), (1,1,1), (0,1,0), (1,1,1), (1,0,1)],
}
printer = JojjjajjrPrettyPrint(fixed_gradient_config)

for model_name in model_modules:

    # Find matching model subtypes
    model_module_subtypes = [model for model in model_subtypes if model.lower().startswith(model_name.lower())]
    if len(model_module_subtypes) == 0:
        print(f"No model subtypes found for {model_name}")
        continue

    # Get the first model subtype for now
    model_subtype = model_module_subtypes[0]

    # Get the model class directly from tf.keras.applications
    model_class = getattr(tf.keras.applications, model_subtype)
    
    # Create model instance without weights
    model = model_class(weights=None)

    # Print model summary to individual files per model
    with open(f"model_summaries/{model_name}_summary.txt", "w") as f:
        printer.print(model_name)
        f.write(printer.get_colored_text(model_name))
        model.summary(print_fn=lambda x: f.write(x + "\n"))

        f.write(printer.get_colored_text("Activation functions:"))
        for layer in model.layers:
            if hasattr(layer, 'activation'):
                f.write(f"Layer {layer.name}: {layer.activation.__name__}\n")
    
    # Clean up to free memory
    del model
    import gc
    gc.collect()
