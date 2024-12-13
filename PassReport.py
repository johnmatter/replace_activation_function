from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from matplotlib.backends.backend_pdf import PdfPages

@dataclass
class ReportPage:
    title: str
    content: Any
    type: str  # 'plot', 'text', 'table', etc.
    metadata: Dict[str, Any] = None

class PassReport(ABC):
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.pages: List[ReportPage] = []
    
    @abstractmethod
    def generate_pages(self) -> List[ReportPage]:
        """Generate all pages for the report"""
        pass
    
    @abstractmethod
    def render_page(self, page: ReportPage) -> plt.Figure:
        """
        Render a single page
        
        Returns:
            matplotlib.figure.Figure: The rendered figure
        """
        pass
    
    def save(self) -> None:
        """Save the complete report"""
        self.pages = self.generate_pages()
        with PdfPages(self.output_path) as pdf:
            for page in self.pages:
                fig = self.render_page(page)
                if fig is not None:
                    pdf.savefig(fig)
                plt.close(fig)

class ModelComparisonReport(PassReport):
    def __init__(
        self,
        original_model,
        modified_model,
        X_test,
        y_test,
        original_history,
        modified_history,
        output_path: str
    ):
        super().__init__(output_path)
        self.original_model = original_model
        self.modified_model = modified_model
        self.X_test = X_test
        self.y_test = y_test
        self.original_history_df = self._original_history_to_df(original_history)
        self.modified_history_df = self._modified_history_to_df(modified_history)
        
    def _original_history_to_df(self, history: Union[tf.keras.callbacks.History, Dict]) -> pd.DataFrame:
        """Convert original model's history to DataFrame."""
        hist_dict = history.history if isinstance(history, tf.keras.callbacks.History) else history
        df = pd.DataFrame(hist_dict)
        df.index = range(1, len(df) + 1)
        df.index.name = 'epoch'
        return df
    
    def _modified_history_to_df(self, history: Dict) -> pd.DataFrame:
        """Convert modified model's iterative history to DataFrame."""
        records = []
        
        if 'iterative' in history:
            for entry in history['iterative']:
                record = {
                    'layer_name': entry['layer_name'],
                    'reverted': entry['reverted']
                }
                # Get final values from each metric in the history
                for metric, values in entry['history'].items():
                    record[metric] = values[-1]
                records.append(record)
        
        df = pd.DataFrame(records)
        df.index = range(1, len(df) + 1)
        df.index.name = 'iteration'
        return df

    def generate_pages(self) -> List[ReportPage]:
        """Generate all pages for the report."""
        pages = []
        
        # Training History Page
        pages.append(ReportPage(
            title="Training History Comparison",
            content={
                'original': self.original_history_df,
                'modified': self.modified_history_df
            },
            type='training_history'
        ))
        
        # Get predictions for performance metrics
        y_pred_orig = self.original_model.model.predict(self.X_test)
        y_pred_mod = self.modified_model.model.predict(self.X_test)
        y_true = np.argmax(self.y_test, axis=1)
        
        # Confusion Matrix Page
        pages.append(ReportPage(
            title="Confusion Matrices",
            content={
                'original': confusion_matrix(y_true, np.argmax(y_pred_orig, axis=1)),
                'modified': confusion_matrix(y_true, np.argmax(y_pred_mod, axis=1))
            },
            type='confusion_matrix'
        ))
        
        # Classification Report Page
        pages.append(ReportPage(
            title="Classification Reports",
            content={
                'original': classification_report(y_true, np.argmax(y_pred_orig, axis=1), output_dict=True),
                'modified': classification_report(y_true, np.argmax(y_pred_mod, axis=1), output_dict=True)
            },
            type='classification_report'
        ))
        
        return pages

    def render_page(self, page: ReportPage) -> plt.Figure:
        """Render a single page."""
        if page.type == 'training_history':
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            self._plot_training_histories(axs, page.content)
            plt.tight_layout()
            return fig
            
        elif page.type == 'confusion_matrix':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            self._plot_confusion_matrix(ax1, page.content['original'], "Original Model")
            self._plot_confusion_matrix(ax2, page.content['modified'], "Modified Model")
            plt.tight_layout()
            return fig
            
        elif page.type == 'classification_report':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 8))
            self._plot_classification_report(ax1, page.content['original'], "Original Model")
            self._plot_classification_report(ax2, page.content['modified'], "Modified Model")
            plt.tight_layout()
            return fig
            
        return None

    def _plot_training_histories(self, axs, content: Dict):
        """Plot training metrics comparison."""
        # Plot original model metrics
        self._plot_metric_pair(
            axs[0, 0], 
            content['original'],
            'loss', 
            'val_loss',
            'Original Model Loss'
        )
        
        # Plot modified model metrics
        self._plot_metric_pair(
            axs[0, 1],
            content['modified'],
            'loss',
            'val_loss',
            'Modified Model Loss'
        )
        
        # Plot accuracies
        metrics = ['accuracy', 'val_accuracy'] if 'accuracy' in content['original'] else ['categorical_accuracy', 'val_categorical_accuracy']
        
        self._plot_metric_pair(
            axs[1, 0],
            content['original'],
            metrics[0],
            metrics[1],
            'Original Model Accuracy'
        )
        
        self._plot_metric_pair(
            axs[1, 1],
            content['modified'],
            metrics[0],
            metrics[1],
            'Modified Model Accuracy'
        )

    def _plot_confusion_matrix(self, ax, cm: np.ndarray, title: str):
        """Plot a single confusion matrix."""
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

    def _plot_classification_report(self, ax, report: Dict, title: str):
        """Plot a single classification report."""
        df = pd.DataFrame(report).transpose()
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(
            cellText=np.round(df.values, 3),
            rowLabels=df.index,
            colLabels=df.columns,
            cellLoc='center',
            loc='center'
        )
        table.scale(1, 1.5)
        ax.set_title(title)
        plt.tight_layout()

    def _plot_metric_pair(self, ax, df: pd.DataFrame, train_metric: str, val_metric: str, title: str):
        """Helper method to plot training/validation metric pairs."""
        if train_metric in df.columns:
            ax.plot(df.index, df[train_metric], 'o-', label=f'Training {train_metric}')
        if val_metric in df.columns:
            ax.plot(df.index, df[val_metric], 'x--', label=f'Validation {val_metric}')
        
        ax.set_title(title)
        ax.set_xlabel('Epoch/Iteration')
        ax.set_ylabel(train_metric.capitalize())
        ax.legend()
        ax.grid(True)

class SingleModelReport(PassReport):
    def __init__(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        history: Union[tf.keras.callbacks.History, Dict],
        output_path: str
    ):
        super().__init__(output_path)
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.history = history
        
    def generate_pages(self) -> List[ReportPage]:
        pages = []
        
        # Training History
        if self.history is not None:
            history_df = pd.DataFrame(self.history.history if isinstance(self.history, tf.keras.callbacks.History) else self.history)
            pages.append(ReportPage(
                title="Training History",
                content=history_df,
                type='training_history'
            ))
        
        # Model Performance
        y_pred = self.model.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        pages.append(ReportPage(
            title="Confusion Matrix",
            content=cm,
            type='confusion_matrix'
        ))
        
        # Classification Report
        report = pd.DataFrame(
            classification_report(y_true, y_pred_classes, output_dict=True)
        ).transpose()
        pages.append(ReportPage(
            title="Classification Report",
            content=report,
            type='classification_report'
        ))
        
        return pages
    
    def render_page(self, page: ReportPage) -> plt.Figure:
        if page.type == 'training_history':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            self._plot_training_history(page.content)
            return fig
        elif page.type == 'confusion_matrix':
            fig, ax = plt.subplots(figsize=(8, 6))
            self._plot_confusion_matrix(page.content)
            return fig
        elif page.type == 'classification_report':
            fig, ax = plt.subplots(figsize=(8.5, len(page.content) * 0.5))
            self._plot_classification_report(page.content)
            return fig
    
    def _plot_training_history(self, df: pd.DataFrame) -> plt.Figure:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot loss
        df[['loss', 'val_loss']].plot(ax=ax1, marker='o')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        ax1.legend()
        
        # Plot accuracy
        acc_cols = [col for col in df.columns if 'acc' in col.lower()]
        if acc_cols:
            df[acc_cols].plot(ax=ax2, marker='o')
            ax2.set_title('Model Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.grid(True)
            ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def _plot_confusion_matrix(self, cm: np.ndarray) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        return fig
    
    def _plot_classification_report(self, report_df: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8.5, len(report_df) * 0.5))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(
            cellText=np.round(report_df.values, 2),
            rowLabels=report_df.index,
            colLabels=report_df.columns,
            cellLoc='center',
            loc='center'
        )
        table.scale(1, 1.5)
        ax.set_title('Classification Report')
        plt.tight_layout()
        return fig