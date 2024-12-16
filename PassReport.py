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
        self._set_default_theme()
    
    def _set_default_theme(self):
        """Set default plotting theme."""
        self.theme = {
            'style': 'dark_background',
            'palette': 'Pastel1',
            'background_color': '#1C1C1C',
            'text_color': 'white',
            'grid_color': 'gray',
            'grid_alpha': 0.2,
        }
    
    def set_theme(self, **kwargs):
        """Update theme settings."""
        self.theme.update(kwargs)
    
    def _apply_theme_to_axis(self, ax):
        """Apply theme settings to a matplotlib axis."""
        plt.style.use(self.theme['style'])
        
        ax.set_facecolor(self.theme['background_color'])
        ax.grid(True, alpha=self.theme['grid_alpha'], color=self.theme['grid_color'])
        
        # Set text colors
        ax.title.set_color(self.theme['text_color'])
        ax.xaxis.label.set_color(self.theme['text_color'])
        ax.yaxis.label.set_color(self.theme['text_color'])
        ax.tick_params(colors=self.theme['text_color'])
        
        # Set spine colors
        for spine in ax.spines.values():
            spine.set_color(self.theme['text_color'])
            
        # If legend exists, update its colors
        if ax.get_legend() is not None:
            legend = ax.get_legend()
            legend.get_frame().set_facecolor(self.theme['background_color'])
            for text in legend.get_texts():
                text.set_color(self.theme['text_color'])
    
    def _convert_history_to_df(self, history: Union[tf.keras.callbacks.History, Dict]) -> pd.DataFrame:
        """Convert standard Keras history to DataFrame with consistent format."""
        hist_dict = history.history if isinstance(history, tf.keras.callbacks.History) else history
        df = pd.DataFrame(hist_dict)
        df['epoch'] = range(1, len(df) + 1)
        return df

    def _plot_confusion_matrix(self, ax, cm: np.ndarray, title: str = 'Confusion Matrix'):
        """Plot confusion matrix with consistent styling."""
        sns.heatmap(cm, annot=True, fmt='d', 
                   cmap=self.theme['palette'], 
                   cbar=False, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        self._apply_theme_to_axis(ax)
    
    def _plot_classification_report(self, ax, report_df: pd.DataFrame, title: str = 'Classification Report'):
        """Plot classification report with consistent styling."""
        ax.axis('tight')
        ax.axis('off')
        
        # Convert the DataFrame to numeric values where possible
        numeric_df = report_df.apply(pd.to_numeric, errors='ignore')
        
        # Round only the numeric columns
        rounded_values = numeric_df.applymap(lambda x: round(x, 3) if isinstance(x, (int, float)) else x)
        
        table = ax.table(
            cellText=rounded_values.values,
            rowLabels=report_df.index,
            colLabels=report_df.columns,
            cellLoc='center',
            loc='center'
        )
        
        self._apply_table_theme(table)
        ax.set_title(title)
        self._apply_theme_to_axis(ax)
    
    def _apply_table_theme(self, table):
        """Apply theme to table elements."""
        # Set colors for all cells
        for cell in table._cells.values():
            cell.set_facecolor(self.theme['background_color'])
            cell.set_text_props(color=self.theme['text_color'])
            cell.set_edgecolor(self.theme['text_color'])
        
        # Adjust table scale for better visibility
        table.scale(1, 1.5)
        
        # Set auto column widths
        table.auto_set_column_width([i for i in range(len(table._cells))])
    
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

    def _apply_theme_to_figure(self, fig):
        """Apply theme settings to entire figure and all its subplots."""
        fig.patch.set_facecolor(self.theme['background_color'])
        for ax in fig.get_axes():
            self._apply_theme_to_axis(ax)

    def _configure_plot_layout(self, fig):
        """Configure common plot layout settings."""
        self._apply_theme_to_figure(fig)
        plt.tight_layout()
    
    def _configure_legend(self, ax):
        """Apply consistent legend styling."""
        if ax.get_legend() is not None:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.get_legend().set_facecolor(self.theme['background_color'])
            for text in ax.get_legend().get_texts():
                text.set_color(self.theme['text_color'])

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
        self.original_history_df = self._convert_history_to_df(original_history)
        self.modified_history_df = self._modified_history_to_df(modified_history)
        
    def _original_history_to_df(self, history: Union[tf.keras.callbacks.History, Dict]) -> pd.DataFrame:
        """Convert original model's history to DataFrame."""
        hist_dict = history.history if isinstance(history, tf.keras.callbacks.History) else history
        df = pd.DataFrame(hist_dict)
        df['epoch'] = range(1, len(df) + 1)
        return df
    
    def _modified_history_to_df(self, history: Dict) -> pd.DataFrame:
        """Convert modified model's iterative history to DataFrame."""
        records = []
        
        if 'iterative' in history:
            # For this retraining mode, we have a list of layers, each with their own history
            for layer in history['iterative']:
                metrics = list(layer['history'].keys())
                # Check that number of epochs is the same for all metrics
                # Assume the first metric has a valid length
                number_of_epochs = len(layer['history'][metrics[0]])
                for metric in metrics:
                    if len(layer['history'][metric]) != number_of_epochs:
                        raise ValueError(f"Number of epochs for metric {metric} does not match for layer {layer['layer_name']}: {len(layer['history'][metric])} != {number_of_epochs}")

                # Create a row for each epoch, one column per metric
                for epoch in range(number_of_epochs):
                    record = {
                        'layer': layer['layer'],
                        'layer_name': layer['layer_name'],
                        'reverted': layer['reverted'],
                        'phase': layer['phase'],
                        'epoch': epoch,
                        **{metric: layer['history'][metric][epoch] for metric in metrics}
                    }
                    records.append(record)
        elif 'all' in history:
            raise ValueError("'all' retraining is not yet supported")
        elif 'batched' in history:
            raise ValueError("'batched' retraining is not yet supported")
        else:
            raise ValueError("Invalid history format")
        
        df = pd.DataFrame(records)
        # df.index = range(1, len(df) + 1)
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
            self._configure_plot_layout(fig)
            return fig
            
        elif page.type == 'confusion_matrix':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            self._plot_confusion_matrix(ax1, page.content['original'], "Original Model")
            self._plot_confusion_matrix(ax2, page.content['modified'], "Modified Model")
            self._configure_plot_layout(fig)
            return fig
            
        elif page.type == 'classification_report':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 8))
            # Convert dictionaries to DataFrames before plotting
            original_df = pd.DataFrame(page.content['original']).transpose()
            modified_df = pd.DataFrame(page.content['modified']).transpose()
            self._plot_classification_report(ax1, original_df, "Original Model")
            self._plot_classification_report(ax2, modified_df, "Modified Model")
            self._configure_plot_layout(fig)
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

    def _plot_metric_pair(self, ax, df: pd.DataFrame, train_metric: str, val_metric: str, title: str):
        """Helper method to plot training/validation metric pairs."""
        if 'layer_name' in df.columns:
            # For modified model with multiple layers
            unique_layers = df['layer_name'].unique()
            num_colors = len(unique_layers)
            colors = plt.cm.get_cmap(self.theme['palette'])(np.linspace(0, 1, num_colors))
            
            for i, layer in enumerate(unique_layers):
                layer_data = df[df['layer_name'] == layer]
                if train_metric in layer_data.columns:
                    ax.plot(layer_data['epoch'], layer_data[train_metric], 
                           'o-', color=colors[i], label=f'{layer} (train)')
                if val_metric in layer_data.columns:
                    ax.plot(layer_data['epoch'], layer_data[val_metric], 
                           'x--', color=colors[i], label=f'{layer} (val)')
        else:
            # For original model without layers
            colors = plt.cm.get_cmap(self.theme['palette'])([0, 1])
            if train_metric in df.columns:
                ax.plot(df['epoch'], df[train_metric], 'o-', 
                       color=colors[0], label=f'Training {train_metric}')
            if val_metric in df.columns:
                ax.plot(df['epoch'], df[val_metric], 'x--', 
                       color=colors[1], label=f'Validation {val_metric}')
        
        ax.set_title(title)
        ax.set_xlabel('Epoch/Iteration')
        ax.set_ylabel(train_metric.capitalize())
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        self._apply_theme_to_axis(ax)

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
        self.history_df = self._convert_history_to_df(history)

    def generate_pages(self) -> List[ReportPage]:
        pages = []
        
        # Training History
        if self.history_df is not None:
            pages.append(ReportPage(
                title="Training History",
                content=self.history_df,
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
            self._plot_training_history(page.content, ax1, ax2)
            self._configure_plot_layout(fig)
            return fig
        elif page.type == 'confusion_matrix':
            fig, ax = plt.subplots(figsize=(8, 6))
            self._plot_confusion_matrix(ax, page.content)
            self._configure_plot_layout(fig)
            return fig
        elif page.type == 'classification_report':
            fig, ax = plt.subplots(figsize=(8.5, len(page.content) * 0.5))
            self._plot_classification_report(ax, page.content)
            self._configure_plot_layout(fig)
            return fig
    
    def _plot_training_history(self, df: pd.DataFrame, ax1, ax2):
        """Plot training history on provided axes."""
        colors = plt.cm.get_cmap(self.theme['palette'])([0, 1])
        
        # Plot loss
        if 'loss' in df.columns:
            loss_cols = [col for col in df.columns if 'loss' in col.lower()]
            for i, col in enumerate(loss_cols):
                ax1.plot(df['epoch'], df[col], 'o-' if 'val' not in col else 'x--',
                        color=colors[i], label=col)
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot accuracy
        acc_cols = [col for col in df.columns if 'acc' in col.lower()]
        if acc_cols:
            for i, col in enumerate(acc_cols):
                ax2.plot(df['epoch'], df[col], 'o-' if 'val' not in col else 'x--',
                        color=colors[i], label=col)
            ax2.set_title('Model Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        self._apply_theme_to_axis(ax1)
        self._apply_theme_to_axis(ax2)