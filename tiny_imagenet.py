import os
from typing import Dict, Tuple
import pandas as pd
from PIL import Image
import numpy as np

class TinyImagenetLoader:
    @staticmethod
    def load_tiny_imagenet(root_dir: str) -> Tuple[Dict, Dict, Dict]:
        """
        Load TinyImageNet dataset from disk.
        
        Args:
            root_dir: Path to TinyImageNet root directory
            
        Returns:
            Tuple containing:
            - train_data: Dict with keys 'images' (np.array), 'labels' (np.array), 'label_map' (Dict)
            - val_data: Dict with keys 'images' (np.array), 'labels' (np.array)
            - test_data: Dict with keys 'images' (np.array)
        """
        # Load class ID mappings
        with open(os.path.join(root_dir, 'wnids.txt'), 'r') as f:
            class_ids = [line.strip() for line in f.readlines()]
        
        # Create label mapping
        label_map = {class_id: idx for idx, class_id in enumerate(class_ids)}
        
        # Load human readable labels
        word_map = {}
        with open(os.path.join(root_dir, 'words.txt'), 'r') as f:
            for line in f.readlines():
                class_id, label = line.strip().split('\t')
                word_map[class_id] = label

        # Load training data
        train_images = []
        train_labels = []
        for class_id in class_ids:
            class_dir = os.path.join(root_dir, 'train', class_id, 'images')
            for img_file in os.listdir(class_dir):
                if img_file.endswith('.JPEG'):
                    img_path = os.path.join(class_dir, img_file)
                    img = Image.open(img_path)
                    img = img.convert('RGB')
                    img = np.array(img)
                    train_images.append(img)
                    train_labels.append(label_map[class_id])
        
        train_data = {
            'images': np.array(train_images),
            'labels': np.array(train_labels),
            'label_map': {v: word_map[k] for k,v in label_map.items()}
        }

        # Load validation data
        val_images = []
        val_labels = []
        val_annotations = pd.read_csv(os.path.join(root_dir, 'val', 'val_annotations.txt'), 
                                    sep='\t', header=None,
                                    names=['filename', 'class_id', 'x', 'y', 'w', 'h'])
        
        for _, row in val_annotations.iterrows():
            img_path = os.path.join(root_dir, 'val', 'images', row['filename'])
            img = Image.open(img_path)
            img = img.convert('RGB')
            img = np.array(img)
            val_images.append(img)
            val_labels.append(label_map[row['class_id']])

        val_data = {
            'images': np.array(val_images),
            'labels': np.array(val_labels)
        }

        # Load test data
        test_images = []
        test_dir = os.path.join(root_dir, 'test', 'images')
        for img_file in sorted(os.listdir(test_dir)):
            if img_file.endswith('.JPEG'):
                img_path = os.path.join(test_dir, img_file)
                img = Image.open(img_path)
                img = img.convert('RGB')
                img = np.array(img)
                test_images.append(img)

        test_data = {
            'images': np.array(test_images)
        }

        return train_data, val_data, test_data
