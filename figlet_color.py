from pyfiglet import Figlet
import colorsys
from typing import List, Tuple, Optional, Dict, Any
import os
import pyfiglet

class MOXIEPrettyPrint:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with a config dictionary that can include:
        {
            'font': str,              # pyfiglet font name
            'style': str,             # 'gradient', 'rainbow', 'multi_gradient', 'per_line'
            'colors': List[Tuple],    # list of RGB tuples (each 0-1)
            'frequency': float,       # for rainbow style
            'start_color': Tuple,     # for simple gradient
            'end_color': Tuple,       # for simple gradient
            'gradient_length': int,   # Default length for fixed-length gradients
            'font_dir': str,          # Default font directory
        }
        """
        self.config = {
            'font': '3d_diagonal',
            'style': 'multi_gradient',
            'colors': [(1,0,0), (1,1,0), (0,1,0), (0,1,1), (0,0,1)],
            'frequency': 0.1,
            'start_color': (1,0,0),
            'end_color': (0,0,1),
            'gradient_length': 10,
            'width': 100,
            'font_dir': os.path.join(os.path.dirname(pyfiglet.__file__), 'fonts'),
        }
        self.config.update(config)
        
        # Ensure font directory exists
        if not os.path.exists(self.config['font_dir']):
            print(f"Warning: Font directory {self.config['font_dir']} does not exist.")
            print("You may need to create it and download some figlet fonts.")
            print("You can download fonts from: https://github.com/xero/figlet-fonts")

        self.figlet = self._get_figlet()
        
        # Map style names to their corresponding methods
        self._style_map = {
            'gradient': self._gradient_text,
            'rainbow': self._rainbow_text,
            'multi_gradient': self._multi_color_gradient,
            'per_line': self._color_per_line,
            'fixed_length_gradient': self._fixed_length_gradient,
        }

    def list_available_fonts(self):
        """Print all available fonts using a sample text to demonstrate each font"""
        # Get list of font files from the font directory
        font_files = [f[:-4] for f in os.listdir(self.config['font_dir']) 
                     if f.endswith('.flf')]
        
        # Save current configuration
        original_style = self.config['style']
        original_font = self.config['font']
        original_width = self.config['width']
        
        # Temporarily change style and width
        self.config['style'] = 'multi_gradient'
        self.config['colors'] = [(1,0,0), (1,1,0), (0,1,0), (0,1,1), (0,0,1)]  # Rainbow colors
        self.config['width'] = 200  # Increase width to prevent truncation
        
        print("\nAvailable Fonts:")
        print("================")
        
        # Print each font name using its own style
        sample_text = "Pretty Print"  # Or any other sample text you prefer
        for font_name in sorted(font_files):
            try:
                self.config['font'] = font_name
                self.figlet = self._get_figlet()  # Refresh figlet with new font
                print(f"\nFont: {font_name}")
                self.print(sample_text)
            except Exception as e:
                print(f"Error loading font {font_name}: {str(e)}")
        
        # Restore original configuration
        self.config['style'] = original_style
        self.config['font'] = original_font
        self.config['width'] = original_width
        self.figlet = self._get_figlet()

    @staticmethod
    def _rgb_to_ansi(r: float, g: float, b: float) -> str:
        return f"\033[38;2;{int(r*255)};{int(g*255)};{int(b*255)}m"

    def _color_per_line(self, lines: List[str]) -> str:
        result = []
        colors = self.config['colors']
        for line, color in zip(lines, colors):
            result.append(self._rgb_to_ansi(color[0], color[1], color[2]) + line)
        return '\n'.join(result)

    def _rainbow_text(self, lines: List[str]) -> str:
        result = []
        frequency = self.config['frequency']
        
        for line in lines:
            colored_line = ""
            for i, char in enumerate(line):
                if char != " ":
                    hue = (i * frequency) % 1.0
                    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                    colored_line += f"{self._rgb_to_ansi(r,g,b)}{char}"
                else:
                    colored_line += char
            result.append(colored_line)
        return '\n'.join(result)

    def _gradient_text(self, lines: List[str]) -> str:
        result = []
        start_color = self.config['start_color']
        end_color = self.config['end_color']
        
        for line in lines:
            colored_line = ""
            for i, char in enumerate(line):
                if char != " ":
                    progress = i / len(line) if len(line) > 1 else 0
                    r = start_color[0] + (end_color[0] - start_color[0]) * progress
                    g = start_color[1] + (end_color[1] - start_color[1]) * progress
                    b = start_color[2] + (end_color[2] - start_color[2]) * progress
                    colored_line += f"{self._rgb_to_ansi(r,g,b)}{char}"
                else:
                    colored_line += char
            result.append(colored_line)
        return '\n'.join(result)

    def _multi_color_gradient(self, lines: List[str]) -> str:
        result = []
        colors = self.config['colors']
        
        for line in lines:
            colored_line = ""
            for i, char in enumerate(line):
                if char != " ":
                    segment_length = 1.0 / (len(colors) - 1)
                    position = (i / len(line)) if len(line) > 1 else 0
                    segment = int(position / segment_length)
                    segment = min(segment, len(colors) - 2)
                    
                    segment_position = (position - segment * segment_length) / segment_length
                    
                    start_color = colors[segment]
                    end_color = colors[segment + 1]
                    r = start_color[0] + (end_color[0] - start_color[0]) * segment_position
                    g = start_color[1] + (end_color[1] - start_color[1]) * segment_position
                    b = start_color[2] + (end_color[2] - start_color[2]) * segment_position
                    
                    colored_line += f"{self._rgb_to_ansi(r,g,b)}{char}"
                else:
                    colored_line += char
            result.append(colored_line)
        return '\n'.join(result)

    def _gradient_ignore_line_length(self, lines: List[str]) -> str:
        return self._gradient_text(lines)   

    def _fixed_length_gradient(self, lines: List[str]) -> str:
        """Apply gradient over a fixed number of non-space characters, then repeat"""
        result = []
        start_color = self.config['start_color']
        end_color = self.config['end_color']
        gradient_length = self.config['gradient_length']
        
        for line in lines:
            colored_line = ""
            char_count = 0  # Count of non-space characters
            
            for char in line:
                if char != " ":
                    # Calculate position within the fixed-length gradient
                    position = (char_count % gradient_length) / (gradient_length - 1)
                    
                    # Interpolate colors
                    r = start_color[0] + (end_color[0] - start_color[0]) * position
                    g = start_color[1] + (end_color[1] - start_color[1]) * position
                    b = start_color[2] + (end_color[2] - start_color[2]) * position
                    
                    colored_line += f"{self._rgb_to_ansi(r,g,b)}{char}"
                    char_count += 1
                else:
                    colored_line += char
                    
            result.append(colored_line)
        return '\n'.join(result)

    def _load_font_from_file(self, font_name):
        """Load a font from the configured font directory"""
        font_path = os.path.join(self.config['font_dir'], f"{font_name}.flf")
        try:
            # Create Figlet object with the full path
            return Figlet(font=font_path, dir=self.config['font_dir'], width=self.config['width'])
        except Exception as e:
            print(f"Warning: Could not load font '{font_name}' from {font_path}")
            print(f"Detailed Error: {str(e)}")
            print(f"Error type: {type(e)}")
            return None

    def _get_figlet(self):
        """Get a figlet object with the configured font"""
        font_name = self.config.get('font', 'standard')
        
        try:
            # Just pass the font name now that we've added the directory to the search path
            return Figlet(font=font_name, width=self.config['width'])
        except pyfiglet.FontNotFound as e:
            print(f"Warning: Font '{font_name}' not found, falling back to 'standard'")
            print(f"Available fonts: {pyfiglet.DEFAULT_FONT}")
            return Figlet(font='standard', width=self.config['width'])

    def print(self, text: str) -> None:
        """Print the text using the configured style and colors"""
        colored_text = self.get_colored_text(text)
        print(colored_text)

    def get_colored_text(self, text: str) -> str:
        figlet_text = self.figlet.renderText(text)
        lines = figlet_text.split('\n')
        style_func = self._style_map.get(self.config['style'], self._multi_color_gradient)
        colored_text = style_func(lines)
        colored_text += '\033[0m'  # Reset color
        return colored_text

def quick_demo():
    # Example usage:
    config = {
        'style': 'multi_gradient',
        'colors': [(1,0,0), (1,0,1), (0,1,0), (0,1,1), (0,0,1)],
    }
    printer = MOXIEPrettyPrint(config)
    printer.print('jojjjajjr')

    # Rainbow
    rainbow_config = {
        'style': 'rainbow',
        'frequency': 0.1
    }
    rainbow_printer = MOXIEPrettyPrint(rainbow_config)
    rainbow_printer.print('jojjjajjr')

    # Simple gradient
    gradient_config = {
        'style': 'gradient',
        'start_color': (1.0, 0.0, 1.0), # pink
        'end_color': (0.0, 1.0, 0.0) # green
    }
    gradient_printer = MOXIEPrettyPrint(gradient_config)
    gradient_printer.print('jojjjajjr')

    # Fixed-length gradient
    fixed_gradient_config = {
        'style': 'fixed_length_gradient',
        'start_color': (1,0,0),  # Red
        'end_color': (0,1,1),    # Blue
        'gradient_length': 20     # Gradient repeats every 5 non-space characters
    }
    fixed_printer = MOXIEPrettyPrint(fixed_gradient_config)
    fixed_printer.print('jojjjajjr')

def main():
    printer = MOXIEPrettyPrint({})
    printer.list_available_fonts()

if __name__ == '__main__':
    main()