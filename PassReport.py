from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

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
    def render_page(self, page: ReportPage) -> None:
        """Render a single page"""
        pass
    
    def save(self) -> None:
        """Save the complete report"""
        self.pages = self.generate_pages()
        with PdfPages(self.output_path) as pdf:
            for page in self.pages:
                self.render_page(page)
                pdf.savefig()
                plt.close()

class ModelComparisonReport(PassReport):
    def __init__(self, original_model: ModelWrapper, 
                 modified_model: ModelWrapper,
                 X_train: np.ndarray, y_train: np.ndarray,
                 output_path: str):
        super().__init__(output_path)
        self.original_model = original_model
        self.modified_model = modified_model
        self.X_train = X_train
        self.y_train = y_train