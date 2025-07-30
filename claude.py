import numpy as np
import random
import json
import os
import glob
from typing import List, Dict, Tuple, Any, Optional
from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass

# Remove PyTorch dependency for now - will implement simplified neural network
# import torch
# import torch.nn as nn
# import torch.optim as optim

# ===== PRIMITIVE LIBRARY (DSL) =====
class Primitive(ABC):
    """Base class for all primitives (suckers)"""
    @abstractmethod
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass
    
    @abstractmethod
    def get_type(self) -> str:
        """Return primitive type: transform, logical, detection, edit"""
        pass

class TileToOutputPrimitive(Primitive):
    """
    Tiles the input grid to match the output grid's shape.
    """
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        if params is None or 'target_shape' not in params:
            return grid.copy()
        target_shape = params['target_shape']
        reps = (int(np.ceil(target_shape[0] / grid.shape[0])),
                int(np.ceil(target_shape[1] / grid.shape[1])))
        tiled = np.tile(grid, reps)
        return tiled[:target_shape[0], :target_shape[1]]
    def get_name(self): return "tile_to_output"
    def get_type(self): return "size"

class ExtractAndPlacePrimitive(Primitive):
    """
    Extracts a subgrid from input and places it at a specified location in a larger grid.
    """
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        # params: {'target_shape': (rows, cols), 'position': (r, c)}
        if params is None or 'target_shape' not in params or 'position' not in params:
            return grid.copy()
        target_shape = params['target_shape']
        position = params['position']
        result = np.zeros(target_shape, dtype=grid.dtype)
        r, c = position
        rows, cols = grid.shape
        # Place the input grid at the specified position
        r_end = min(r + rows, target_shape[0])
        c_end = min(c + cols, target_shape[1])
        result[r:r_end, c:c_end] = grid[:r_end - r, :c_end - c]
        return result
    def get_name(self): return "extract_and_place"
    def get_type(self): return "size"

class GenerateFilledGridPrimitive(Primitive):
    """
    Generates a new grid of a given size, filled with a specific value or pattern.
    """
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        # params: {'target_shape': (rows, cols), 'fill': value}
        if params is None or 'target_shape' not in params:
            return grid.copy()
        target_shape = params['target_shape']
        fill = params.get('fill', 0)
        return np.full(target_shape, fill, dtype=grid.dtype)
    def get_name(self): return "generate_filled_grid"
    def get_type(self): return "size"
    
# Transform Primitives
class RotatePrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        angle = params.get('angle', 90) if params else 90
        k = angle // 90
        return np.rot90(grid, k)
    
    def get_name(self) -> str:
        return "rotate"
    
    def get_type(self) -> str:
        return "transform"

class FlipPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        axis = params.get('axis', 0) if params else 0
        return np.flip(grid, axis=axis)
    
    def get_name(self) -> str:
        return "flip"
    
    def get_type(self) -> str:
        return "transform"

class MirrorPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        axis = params.get('axis', 1) if params else 1
        return np.flip(grid, axis=axis)
    
    def get_name(self) -> str:
        return "mirror"
    
    def get_type(self) -> str:
        return "transform"
class ShiftUpPadPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        # Shift up by one, pad bottom with zeros
        result = np.zeros_like(grid)
        result[:-1, :] = grid[1:, :]
        return result
    def get_name(self): return "shift_up_pad"
    def get_type(self): return "shift"

class ShiftDownPadPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        # Shift down by one, pad top with zeros
        result = np.zeros_like(grid)
        result[1:, :] = grid[:-1, :]
        return result
    def get_name(self): return "shift_down_pad"
    def get_type(self): return "shift"

class ShiftLeftPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        # Shift left by one, wrap-around (circular shift)
        return np.roll(grid, -1, axis=1)
    def get_name(self): return "shift_left"
    def get_type(self): return "shift"

class ShiftRightPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        # Shift right by one, wrap-around (circular shift)
        return np.roll(grid, 1, axis=1)
    def get_name(self): return "shift_right"
    def get_type(self): return "shift"

class Rot90Primitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        return np.rot90(grid, -1)  # 90° clockwise
    def get_name(self): return "rot90"
    def get_type(self): return "transform"

class Rot180Primitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        return np.rot90(grid, 2)
    def get_name(self): return "rot180"
    def get_type(self): return "transform"

class Rot270Primitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        return np.rot90(grid, 1)  # 90° counterclockwise
    def get_name(self): return "rot270"
    def get_type(self): return "transform"

class FlipHPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        return np.fliplr(grid)
    def get_name(self): return "flip_h"
    def get_type(self): return "transform"

class FlipVPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        return np.flipud(grid)
    def get_name(self): return "flip_v"
    def get_type(self): return "transform"

class TransposePrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        return grid.T
    def get_name(self): return "transpose"
    def get_type(self): return "transform"

class FlipDiagPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        return np.transpose(grid)
    def get_name(self): return "flip_diag"
    def get_type(self): return "transform"

class FlipAntiDiagPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        return np.fliplr(np.flipud(np.transpose(grid)))
    def get_name(self): return "flip_antidiag"
    def get_type(self): return "transform"

# Logical Primitives
class ANDPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        if params and 'mask' in params:
            mask = params['mask']
            if mask.shape == grid.shape:
                return np.logical_and(grid > 0, mask > 0).astype(int)
        return grid
    
    def get_name(self) -> str:
        return "and"
    
    def get_type(self) -> str:
        return "logical"

class ORPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        if params and 'mask' in params:
            mask = params['mask']
            if mask.shape == grid.shape:
                return np.logical_or(grid > 0, mask > 0).astype(int)
        return grid
    
    def get_name(self) -> str:
        return "or"
    
    def get_type(self) -> str:
        return "logical"

# Detection Primitives
class ColorDetectionPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        color = params.get('color', 1) if params else 1
        return (grid == color).astype(int)
    
    def get_name(self) -> str:
        return "color_detect"
    
    def get_type(self) -> str:
        return "detection"

# Neural Network Pattern Detection Primitive (Simplified Implementation)
class PatternDetectionNN(Primitive):
    """Neural network-based pattern detection primitive"""
    def __init__(self):
        # Simplified neural network weights
        self.weights = np.random.randn(3, 3) * 0.1  # 3x3 convolution kernel
        self.bias = 0.0
        self.learning_rate = 0.01
        self.trained = False
    
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        """Apply learned pattern detection"""
        if grid.size == 0:
            return grid
        
        result = np.zeros_like(grid)
        rows, cols = grid.shape
        
        # Apply convolution with learned weights
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                patch = grid[i-1:i+2, j-1:j+2]
                activation = np.sum(patch * self.weights) + self.bias
                result[i, j] = 1 if activation > 0.5 else 0
        
        return result
    
    def train_on_pattern(self, input_grid: np.ndarray, target_grid: np.ndarray):
        """Train the neural network on a specific pattern"""
        if input_grid.shape != target_grid.shape:
            return
        
        rows, cols = input_grid.shape
        
        # Simple gradient descent training
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                patch = input_grid[i-1:i+2, j-1:j+2]
                predicted = np.sum(patch * self.weights) + self.bias
                predicted = 1 if predicted > 0.5 else 0
                target = target_grid[i, j]
                
                # Calculate gradient
                error = target - predicted
                gradient = error * patch
                
                # Update weights
                self.weights += self.learning_rate * gradient
                self.bias += self.learning_rate * error
        
        self.trained = True
    
    def get_name(self) -> str:
        return "pattern_detect_nn"
    
    def get_type(self) -> str:
        return "detection"

# Editing Primitives
class AddPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        result = grid.copy()
        
        if not params:
            return result
            
        value = params.get('value', 1)
        position = params.get('position')
        
        if position and len(position) == 2:
            r, c = position
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                result[r, c] = value
        
        return result
    
    def get_name(self) -> str:
        return "add"
    
    def get_type(self) -> str:
        return "edit"

class FillEnclosedPrimitive(Primitive):
    """Fill enclosed regions with a specific color"""
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        boundary_color = params.get('boundary_color', 3) if params else 3
        fill_color = params.get('fill_color', 4) if params else 4
        
        result = grid.copy()
        rows, cols = result.shape
        
        def is_enclosed(start_r, start_c):
            """Check if a region starting from (start_r, start_c) is enclosed"""
            if result[start_r, start_c] != 0:
                return False, []
            
            visited = set()
            queue = [(start_r, start_c)]
            region_cells = []
            touches_border = False
            
            while queue:
                r, c = queue.pop(0)
                if (r, c) in visited:
                    continue
                    
                visited.add((r, c))
                
                # Check if we're at the border
                if r == 0 or r == rows-1 or c == 0 or c == cols-1:
                    touches_border = True
                
                if result[r, c] == 0:  # Empty cell
                    region_cells.append((r, c))
                    
                    # Add unvisited neighbors
                    for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < rows and 0 <= nc < cols and 
                            (nr, nc) not in visited):
                            queue.append((nr, nc))
            
            return not touches_border and len(region_cells) > 0, region_cells
        
        # Find and fill enclosed regions
        processed = set()
        for i in range(rows):
            for j in range(cols):
                if (i, j) not in processed and result[i, j] == 0:
                    is_enc, cells = is_enclosed(i, j)
                    processed.update(cells)
                    
                    if is_enc:
                        for r, c in cells:
                            result[r, c] = fill_color
        
        return result
    
    def get_name(self) -> str:
        return "fill_enclosed"
    
    def get_type(self) -> str:
        return "edit"

# === SPATIAL TRANSFORM PRIMITIVES ===
class TranslatePrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        shift = params.get('shift', (0, 0)) if params else (0, 0)
        result = np.zeros_like(grid)
        rows, cols = grid.shape
        dr, dc = shift
        for i in range(rows):
            for j in range(cols):
                ni, nj = i + dr, j + dc
                if 0 <= ni < rows and 0 <= nj < cols:
                    result[ni, nj] = grid[i, j]
        return result
    def get_name(self): return "translate"
    def get_type(self): return "transform"

class ScalePrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        scale = params.get('scale', 1) if params else 1
        if scale == 1:
            return grid.copy()
        # Simple nearest-neighbor scaling
        rows, cols = grid.shape
        new_rows, new_cols = int(rows * scale), int(cols * scale)
        result = np.zeros((new_rows, new_cols), dtype=grid.dtype)
        for i in range(new_rows):
            for j in range(new_cols):
                src_i = min(int(i / scale), rows - 1)
                src_j = min(int(j / scale), cols - 1)
                result[i, j] = grid[src_i, src_j]
        return result
    def get_name(self): return "scale"
    def get_type(self): return "transform"
class TilePatternPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        # Placeholder: tile 2x2 input to 6x6 output
        if grid.shape == (2, 2):
            return np.tile(grid, (3, 3))[:6, :6]
        return grid.copy()
    def get_name(self): return "tile_pattern"
    def get_type(self): return "size"

class Tile3x3Primitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        return np.tile(grid, (3, 3))
    def get_name(self): return "tile_3x3"
    def get_type(self): return "size"

class Tile2x2Primitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        return np.tile(grid, (2, 2))
    def get_name(self): return "tile_2x2"
    def get_type(self): return "size"

class CropCenterHalfPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        rows, cols = grid.shape
        r0 = rows // 4
        c0 = cols // 4
        r1 = r0 + rows // 2
        c1 = c0 + cols // 2
        return grid[r0:r1, c0:c1]
    def get_name(self): return "crop_center_half"
    def get_type(self): return "size"

class CropCenterThirdPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        rows, cols = grid.shape
        r0 = rows // 3
        c0 = cols // 3
        r1 = r0 + rows // 3
        c1 = c0 + cols // 3
        return grid[r0:r1, c0:c1]
    def get_name(self): return "crop_center_third"
    def get_type(self): return "size"

class MaskC1Primitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        return (grid == 1).astype(int)
    def get_name(self): return "mask_c_1"
    def get_type(self): return "color"

class MaskC2Primitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        return (grid == 2).astype(int)
    def get_name(self): return "mask_c_2"
    def get_type(self): return "color"

class MaskC3Primitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        return (grid == 3).astype(int)
    def get_name(self): return "mask_c_3"
    def get_type(self): return "color"

class Replace0to1Primitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        result = grid.copy()
        result[grid == 0] = 1
        return result
    def get_name(self): return "replace_0_to_1"
    def get_type(self): return "color"

class Replace1to2Primitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        result = grid.copy()
        result[grid == 1] = 2
        return result
    def get_name(self): return "replace_1_to_2"
    def get_type(self): return "color"

class FloodObjectPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        # Placeholder: would flood fill from top-left non-zero pixel
        return grid.copy()
    def get_name(self): return "flood_object"
    def get_type(self): return "flood"

class FillBackground0Primitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        # Placeholder: would fill background (border-connected) with 0
        return grid.copy()
    def get_name(self): return "fill_background_0"
    def get_type(self): return "flood"

class ObjectsPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        # Placeholder: would return list of objects, here just return grid
        return grid.copy()
    def get_name(self): return "objects"
    def get_type(self): return "object"

class BBoxPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        # Placeholder: would draw bounding boxes
        return grid.copy()
    def get_name(self): return "bbox"
    def get_type(self): return "object"    

class HoleMaskPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        # Placeholder: would detect holes (enclosed regions)
        return np.zeros_like(grid)
    def get_name(self): return "hole_mask"
    def get_type(self): return "color"

class CropPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        # params: {'top': int, 'left': int, 'height': int, 'width': int}
        if not params:
            return grid.copy()
        top = params.get('top', 0)
        left = params.get('left', 0)
        height = params.get('height', grid.shape[0])
        width = params.get('width', grid.shape[1])
        return grid[top:top+height, left:left+width]
    def get_name(self): return "crop"
    def get_type(self): return "transform"

class ExtendPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        # params: {'new_shape': (rows, cols), 'fill': value}
        if not params or 'new_shape' not in params:
            return grid.copy()
        new_rows, new_cols = params['new_shape']
        fill = params.get('fill', 0)
        result = np.full((new_rows, new_cols), fill, dtype=grid.dtype)
        rows, cols = grid.shape
        min_rows, min_cols = min(rows, new_rows), min(cols, new_cols)
        result[:min_rows, :min_cols] = grid[:min_rows, :min_cols]
        return result
    def get_name(self): return "extend"
    def get_type(self): return "transform"

# === PATTERN OPERATIONS ===
class CopyPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        return grid.copy()
    def get_name(self): return "copy"
    def get_type(self): return "pattern"

class RepeatPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        reps = params.get('reps', (1, 1)) if params else (1, 1)
        return np.tile(grid, reps)
    def get_name(self): return "repeat"
    def get_type(self): return "pattern"

class TilePrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        reps = params.get('reps', (2, 2)) if params else (2, 2)
        return np.tile(grid, reps)
    def get_name(self): return "tile"
    def get_type(self): return "pattern"

# MirrorPrimitive already exists

# === LOGICAL OPERATIONS ===
class IntersectionPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        if params and 'mask' in params:
            mask = params['mask']
            if mask.shape == grid.shape:
                return np.logical_and(grid > 0, mask > 0).astype(grid.dtype)
        return grid.copy()
    def get_name(self): return "intersection"
    def get_type(self): return "logical"

class UnionPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        if params and 'mask' in params:
            mask = params['mask']
            if mask.shape == grid.shape:
                return np.logical_or(grid > 0, mask > 0).astype(grid.dtype)
        return grid.copy()
    def get_name(self): return "union"
    def get_type(self): return "logical"

class DifferencePrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        if params and 'mask' in params:
            mask = params['mask']
            if mask.shape == grid.shape:
                return np.logical_and(grid > 0, mask == 0).astype(grid.dtype)
        return grid.copy()
    def get_name(self): return "difference"
    def get_type(self): return "logical"

class ConditionalFillPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        # params: {'condition': callable, 'fill': value}
        if not params or 'condition' not in params:
            return grid.copy()
        condition = params['condition']
        fill = params.get('fill', 1)
        result = grid.copy()
        mask = condition(grid)
        result[mask] = fill
        return result
    def get_name(self): return "conditional_fill"
    def get_type(self): return "logical"

# === ADVANCED OPERATIONS ===
class FloodFillPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        # Placeholder: real flood fill would require start point and color
        return grid.copy()
    def get_name(self): return "flood_fill"
    def get_type(self): return "advanced"

class ConnectedComponentsPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        # Placeholder: would label connected components
        return grid.copy()
    def get_name(self): return "connected_components"
    def get_type(self): return "advanced"

class SymmetryDetectionPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        # Placeholder: would detect symmetry
        return grid.copy()
    def get_name(self): return "symmetry_detection"
    def get_type(self): return "advanced"

class ObjectExtractionPrimitive(Primitive):
    def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        # Placeholder: would extract objects
        return grid.copy()
    def get_name(self): return "object_extraction"
    def get_type(self): return "advanced"

# ===== ADAPTIVE PRIMITIVE GENERATOR =====
class AdaptivePrimitiveGenerator:
    """Generates new primitives when encountering fundamentally new puzzles"""
    
    def __init__(self):
        self.generated_primitives = []
        self.pattern_library = {}
    
    def analyze_puzzle_novelty(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Determine if this puzzle represents a fundamentally new pattern"""
        # Check if input and output have same dimensions (requirement)
        if input_grid.shape != output_grid.shape:
            print(f"Warning: Input shape {input_grid.shape} != Output shape {output_grid.shape}")
            return False
        
        # Analyze the transformation pattern
        diff_grid = (output_grid != input_grid).astype(int)
        transformation_signature = self._generate_transformation_signature(input_grid, output_grid)
        
        # Check if this transformation pattern is novel
        for known_signature in self.pattern_library.keys():
            if self._similarity_score(transformation_signature, known_signature) > 0.8:
                return False  # Not novel enough
        
        return True
    
    def _generate_transformation_signature(self, input_grid: np.ndarray, output_grid: np.ndarray) -> str:
        """Generate a signature describing the transformation"""
        changes = []
        
        # Analyze color changes
        unique_input_colors = set(input_grid.flatten())
        unique_output_colors = set(output_grid.flatten())
        new_colors = unique_output_colors - unique_input_colors
        
        # Analyze spatial patterns
        diff_mask = input_grid != output_grid
        change_positions = np.where(diff_mask)
        
        # Create signature
        signature = f"colors_in:{len(unique_input_colors)}_out:{len(unique_output_colors)}_new:{len(new_colors)}"
        signature += f"_changes:{len(change_positions[0])}_ratio:{len(change_positions[0])/input_grid.size:.2f}"
        
        return signature
    
    def _similarity_score(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between two transformation signatures"""
        # Simple string similarity for now
        common_parts = sum(1 for a, b in zip(sig1.split('_'), sig2.split('_')) if a == b)
        total_parts = max(len(sig1.split('_')), len(sig2.split('_')))
        return common_parts / total_parts if total_parts > 0 else 0
    
    def generate_new_primitives(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[Primitive]:
        """Generate SMARTER new primitives to solve this novel puzzle"""
        if input_grid.shape != output_grid.shape:
            print("Cannot generate primitives: mismatched grid sizes")
            return []
        
        new_primitives = []
        transformation_signature = self._generate_transformation_signature(input_grid, output_grid)
        
        # Analyze what type of transformation is needed
        diff_mask = input_grid != output_grid
        change_positions = np.where(diff_mask)
        
        print(f"Analyzing transformation: {len(change_positions[0])} cells changed")
        
        if len(change_positions[0]) > 0:
            # 1. Pattern-based primitives
            pattern_primitive = self._create_pattern_learning_primitive(input_grid, output_grid)
            if pattern_primitive:
                new_primitives.append(pattern_primitive)
            
            # 2. Color-based primitives
            color_primitive = self._create_smart_color_primitive(input_grid, output_grid)
            if color_primitive:
                new_primitives.append(color_primitive)
            
            # 3. Shape-based primitives
            shape_primitive = self._create_shape_completion_primitive(input_grid, output_grid)
            if shape_primitive:
                new_primitives.append(shape_primitive)
            
            # 4. Rule-based primitives
            rule_primitive = self._create_rule_based_primitive(input_grid, output_grid)
            if rule_primitive:
                new_primitives.append(rule_primitive)
        
        # Store the pattern for future reference
        self.pattern_library[transformation_signature] = {
            'input_example': input_grid.copy(),
            'output_example': output_grid.copy(),
            'primitives': [p.get_name() for p in new_primitives]
        }
        
        self.generated_primitives.extend(new_primitives)
        print(f"Generated {len(new_primitives)} smart adaptive primitives")
        return new_primitives
    
    def _create_pattern_learning_primitive(self, input_grid: np.ndarray, output_grid: np.ndarray):
        """Create a primitive that learns local patterns"""
        
        class SmartPatternPrimitive(Primitive):
            def __init__(self, input_ref, output_ref):
                self.input_ref = input_ref
                self.output_ref = output_ref
                self.learned_patterns = self._learn_local_patterns()
            
            def _learn_local_patterns(self):
                """Learn 3x3 local transformation patterns"""
                patterns = {}
                rows, cols = self.input_ref.shape
                
                for i in range(1, rows-1):
                    for j in range(1, cols-1):
                        # Get 3x3 neighborhood
                        input_patch = self.input_ref[i-1:i+2, j-1:j+2]
                        output_center = self.output_ref[i, j]
                        input_center = self.input_ref[i, j]
                        
                        if input_center != output_center:
                            # Convert patch to a hashable key
                            patch_key = tuple(input_patch.flatten())
                            patterns[patch_key] = output_center
                
                return patterns
            
            def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
                result = grid.copy()
                rows, cols = result.shape
                
                # Apply learned patterns
                for i in range(1, rows-1):
                    for j in range(1, cols-1):
                        patch = grid[i-1:i+2, j-1:j+2]
                        patch_key = tuple(patch.flatten())
                        
                        if patch_key in self.learned_patterns:
                            result[i, j] = self.learned_patterns[patch_key]
                
                return result
            
            def get_name(self) -> str:
                return f"smart_pattern_{len(self.learned_patterns)}_rules"
            
            def get_type(self) -> str:
                return "adaptive"
        
        primitive = SmartPatternPrimitive(input_grid, output_grid)
        if len(primitive.learned_patterns) > 0:
            return primitive
        return None
    
    def _create_smart_color_primitive(self, input_grid: np.ndarray, output_grid: np.ndarray):
        """Create smart color transformation primitive"""
        
        class SmartColorPrimitive(Primitive):
            def __init__(self, input_ref, output_ref):
                self.color_rules = self._learn_color_rules(input_ref, output_ref)
            
            def _learn_color_rules(self, input_grid, output_grid):
                """Learn context-dependent color transformation rules"""
                rules = {}
                rows, cols = input_grid.shape
                
                for i in range(rows):
                    for j in range(cols):
                        input_color = input_grid[i, j]
                        output_color = output_grid[i, j]
                        
                        if input_color != output_color:
                            # Get neighborhood context
                            neighbors = []
                            for di in [-1, 0, 1]:
                                for dj in [-1, 0, 1]:
                                    ni, nj = i + di, j + dj
                                    if 0 <= ni < rows and 0 <= nj < cols and (di != 0 or dj != 0):
                                        neighbors.append(input_grid[ni, nj])
                            
                            # Create rule based on input color and context
                            context_key = (input_color, tuple(sorted(neighbors)))
                            rules[context_key] = output_color
                
                return rules
            
            def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
                result = grid.copy()
                rows, cols = result.shape
                
                for i in range(rows):
                    for j in range(cols):
                        color = grid[i, j]
                        
                        # Get neighborhood context
                        neighbors = []
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                ni, nj = i + di, j + dj
                                if 0 <= ni < rows and 0 <= nj < cols and (di != 0 or dj != 0):
                                    neighbors.append(grid[ni, nj])
                        
                        context_key = (color, tuple(sorted(neighbors)))
                        if context_key in self.color_rules:
                            result[i, j] = self.color_rules[context_key]
                
                return result
            
            def get_name(self) -> str:
                return f"smart_color_{len(self.color_rules)}_rules"
            
            def get_type(self) -> str:
                return "adaptive"
        
        primitive = SmartColorPrimitive(input_grid, output_grid)
        if len(primitive.color_rules) > 0:
            return primitive
        return None
    
    def _create_shape_completion_primitive(self, input_grid: np.ndarray, output_grid: np.ndarray):
        """Create shape completion primitive"""
        
        class ShapeCompletionPrimitive(Primitive):
            def __init__(self, input_ref, output_ref):
                self.completion_rules = self._learn_shape_completion(input_ref, output_ref)
            
            def _learn_shape_completion(self, input_grid, output_grid):
                """Learn shape completion patterns"""
                rules = []
                rows, cols = input_grid.shape
                
                # Find cells that were added (0 -> non-zero)
                for i in range(rows):
                    for j in range(cols):
                        if input_grid[i, j] == 0 and output_grid[i, j] != 0:
                            # Analyze the pattern around this addition
                            nearby_colors = []
                            for di in [-1, 0, 1]:
                                for dj in [-1, 0, 1]:
                                    ni, nj = i + di, j + dj
                                    if 0 <= ni < rows and 0 <= nj < cols:
                                        if input_grid[ni, nj] != 0:
                                            nearby_colors.append(input_grid[ni, nj])
                            
                            if nearby_colors:
                                # Rule: if surrounded by these colors, fill with this color
                                rule = {
                                    'required_neighbors': set(nearby_colors),
                                    'fill_color': output_grid[i, j],
                                    'min_neighbors': len(nearby_colors)
                                }
                                rules.append(rule)
                
                return rules
            
            def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
                result = grid.copy()
                rows, cols = result.shape
                
                for i in range(rows):
                    for j in range(cols):
                        if result[i, j] == 0:  # Empty cell
                            nearby_colors = set()
                            for di in [-1, 0, 1]:
                                for dj in [-1, 0, 1]:
                                    ni, nj = i + di, j + dj
                                    if 0 <= ni < rows and 0 <= nj < cols:
                                        if result[ni, nj] != 0:
                                            nearby_colors.add(result[ni, nj])
                            
                            # Check if any rule applies
                            for rule in self.completion_rules:
                                if (len(nearby_colors & rule['required_neighbors']) >= rule['min_neighbors'] and
                                    len(nearby_colors) >= 2):  # Need some structure around
                                    result[i, j] = rule['fill_color']
                                    break
                
                return result
            
            def get_name(self) -> str:
                return f"shape_complete_{len(self.completion_rules)}_rules"
            
            def get_type(self) -> str:
                return "adaptive"
        
        primitive = ShapeCompletionPrimitive(input_grid, output_grid)
        if len(primitive.completion_rules) > 0:
            return primitive
        return None
    
    def _create_rule_based_primitive(self, input_grid: np.ndarray, output_grid: np.ndarray):
        """Create rule-based primitive that learns logical transformations"""
        
        class RuleBasedPrimitive(Primitive):
            def __init__(self, input_ref, output_ref):
                self.rules = self._discover_rules(input_ref, output_ref)
            
            def _discover_rules(self, input_grid, output_grid):
                """Discover logical rules"""
                rules = []
                
                # Rule 1: Corner detection and replacement
                corners_changed = self._check_corner_rule(input_grid, output_grid)
                if corners_changed:
                    rules.append(('corners', corners_changed))
                
                # Rule 2: Line extension
                line_rule = self._check_line_extension(input_grid, output_grid)
                if line_rule:
                    rules.append(('line_extension', line_rule))
                
                # Rule 3: Symmetry completion
                symmetry_rule = self._check_symmetry_completion(input_grid, output_grid)
                if symmetry_rule:
                    rules.append(('symmetry', symmetry_rule))
                
                return rules
            
            def _check_corner_rule(self, input_grid, output_grid):
                """Check if corners follow a specific rule"""
                rows, cols = input_grid.shape
                corner_changes = {}
                
                corners = [(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)]
                for r, c in corners:
                    if input_grid[r, c] != output_grid[r, c]:
                        corner_changes[(r, c)] = (input_grid[r, c], output_grid[r, c])
                
                return corner_changes if corner_changes else None
            
            def _check_line_extension(self, input_grid, output_grid):
                """Check if lines are being extended"""
                diff = output_grid - input_grid
                
                # Check for vertical lines
                for j in range(input_grid.shape[1]):
                    col_diff = diff[:, j]
                    if np.sum(col_diff != 0) > 1:  # Multiple changes in column
                        return {'type': 'vertical', 'column': j, 'changes': col_diff}
                
                # Check for horizontal lines
                for i in range(input_grid.shape[0]):
                    row_diff = diff[i, :]
                    if np.sum(row_diff != 0) > 1:  # Multiple changes in row
                        return {'type': 'horizontal', 'row': i, 'changes': row_diff}
                
                return None
            
            def _check_symmetry_completion(self, input_grid, output_grid):
                """Check if symmetry is being completed"""
                # Check if output is more symmetric than input
                rows, cols = output_grid.shape
                
                # Check horizontal symmetry
                h_symmetric = True
                for i in range(rows):
                    for j in range(cols//2):
                        if output_grid[i, j] != output_grid[i, cols-1-j]:
                            h_symmetric = False
                            break
                
                if h_symmetric and not self._is_symmetric(input_grid, 'horizontal'):
                    return {'type': 'horizontal_symmetry'}
                
                return None
            
            def _is_symmetric(self, grid, sym_type):
                """Check if grid is symmetric"""
                if sym_type == 'horizontal':
                    return np.array_equal(grid, np.flip(grid, axis=1))
                return False
            
            def execute(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
                result = grid.copy()
                
                for rule_type, rule_data in self.rules:
                    if rule_type == 'corners':
                        for (r, c), (old_val, new_val) in rule_data.items():
                            if r < result.shape[0] and c < result.shape[1]:
                                if result[r, c] == old_val:
                                    result[r, c] = new_val
                    
                    elif rule_type == 'line_extension':
                        if rule_data['type'] == 'vertical':
                            j = rule_data['column']
                            if j < result.shape[1]:
                                # Extend the most common non-zero value in column
                                col_values = result[:, j]
                                non_zero = col_values[col_values != 0]
                                if len(non_zero) > 0:
                                    fill_value = np.bincount(non_zero).argmax()
                                    result[:, j] = fill_value
                    
                    elif rule_type == 'symmetry':
                        if rule_data['type'] == 'horizontal_symmetry':
                            # Make grid horizontally symmetric
                            rows, cols = result.shape
                            for i in range(rows):
                                for j in range(cols//2):
                                    if result[i, j] != 0:
                                        result[i, cols-1-j] = result[i, j]
                                    elif result[i, cols-1-j] != 0:
                                        result[i, j] = result[i, cols-1-j]
                
                return result
            
            def get_name(self) -> str:
                return f"rule_based_{len(self.rules)}_rules"
            
            def get_type(self) -> str:
                return "adaptive"
        
        primitive = RuleBasedPrimitive(input_grid, output_grid)
        if len(primitive.rules) > 0:
            return primitive
        return None

# ===== COST FUNCTION IMPLEMENTATION =====
class CostFunctionCalculator:
    """Implements J(θ) = Σ(y - ŷ)² / Total for micro-level learning"""
    
    @staticmethod
    def calculate_cost(predicted: np.ndarray, target: np.ndarray) -> float:
        """Calculate J(θ) = Σ(y - ŷ)² / Total"""
        if predicted.shape != target.shape:
            return float('inf')  # Penalty for mismatched shapes
        
        squared_errors = (predicted - target) ** 2
        total_error = np.sum(squared_errors)
        total_cells = target.size
        
        return total_error / total_cells if total_cells > 0 else float('inf')
    
    @staticmethod
    def calculate_gradient(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Calculate gradient for learning"""
        if predicted.shape != target.shape:
            return np.zeros_like(predicted)
        
        return 2 * (predicted - target) / target.size

# ===== TENTACLE (Modular Component) =====
class Tentacle:
    """A tentacle groups primitives to focus on one aspect of the problem"""
    def __init__(self, name: str, primitives: List[Primitive], specialization: str):
        self.name = name
        self.primitives = primitives
        self.specialization = specialization
        self.parameters = {
            'color': random.randint(0, 9),
            'angle': random.choice([90, 180, 270]),
            'axis': random.choice([0, 1]),
            'boundary_color': 3,
            'fill_color': 4
        }
        self.success_rate = 0.0
        self.usage_count = 0
        self.cost_function_value = 0.0  # J(θ) value for this tentacle
        self.weight = 1.0  # Neural network weight
    
    def execute(self, grid: np.ndarray) -> np.ndarray:
        """Execute all primitives in sequence"""
        result = grid.copy()
        
        for primitive in self.primitives:
            try:
                result = primitive.execute(result, self.parameters)
            except Exception as e:
                continue
        
        self.usage_count += 1
        return result
    
    def calculate_cost(self, input_grid: np.ndarray, target_grid: np.ndarray) -> float:
        """Calculate cost function for this tentacle"""
        try:
            output = self.execute(input_grid)
            cost = CostFunctionCalculator.calculate_cost(output, target_grid)
            self.cost_function_value = cost
            return cost
        except Exception as e:
            print(f"Error calculating cost for tentacle {self.name}: {e}")
            self.cost_function_value = float('inf')
            return float('inf')
    
    def update_success_rate(self, success: bool):
        """Update success rate based on performance"""
        if self.usage_count <= 1:
            self.success_rate = 1.0 if success else 0.0
        else:
            alpha = 0.1  # Learning rate
            self.success_rate = (1 - alpha) * self.success_rate + alpha * (1.0 if success else 0.0)
    
    def update_weight(self, cost: float, learning_rate: float = 0.01):
        """Update tentacle weight based on cost (neural network optimization)"""
        # Decrease weight if cost is high, increase if cost is low
        if cost < 1.0:  # Good performance
            self.weight += learning_rate * (1.0 - cost)
        else:  # Poor performance
            self.weight -= learning_rate * min(cost, 1.0)
        
        # Keep weight positive
        self.weight = max(0.1, self.weight)

# ===== OCTOPUS (Complete Solution) =====
class Octopus:
    """An octopus is a complete solution composed of tentacles"""
    def __init__(self, octopus_id: int):
        self.id = octopus_id
        self.tentacles = []
        self.fitness = 0.0
        self.completeness_scores = []
        self.generation = 0
        
        # RL components (FIXED: Now properly integrated)
        self.q_values = {}
        self.learning_rate = 0.1
        self.epsilon = 0.1  # Exploration rate
        self.discount_factor = 0.9
        self.last_state = None
        self.last_action = None
    
    def add_tentacle(self, tentacle: Tentacle):
        self.tentacles.append(tentacle)
    
    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        """Execute the octopus solution using RL-based tentacle selection"""
        if len(self.tentacles) == 0:
            return input_grid
        
        result = input_grid.copy()
        
        # FIXED: Now uses RL for tentacle selection
        state_key = self._generate_state_key(input_grid)
        
        # Select tentacles using RL policy
        for step in range(min(len(self.tentacles), 3)):  # Limit steps to avoid infinite loops
            action = self.select_tentacle_rl(state_key)
            
            if 0 <= action < len(self.tentacles):
                try:
                    old_result = result.copy()
                    result = self.tentacles[action].execute(result)
                    
                    # Calculate immediate reward based on change
                    reward = self._calculate_step_reward(old_result, result, input_grid)
                    
                    # Update Q-value
                    if self.last_state is not None and self.last_action is not None:
                        self.update_q_value(self.last_state, self.last_action, reward)
                    
                    self.last_state = state_key
                    self.last_action = action
                    
                except Exception as e:
                    continue
        
        return result
    
    def _generate_state_key(self, grid: np.ndarray) -> str:
        """Generate a state key for RL"""
        # Simple state representation: grid size, non-zero cells, unique colors
        non_zero_count = np.sum(grid != 0)
        unique_colors = len(np.unique(grid))
        grid_sum = np.sum(grid)
        
        return f"size_{grid.shape[0]}x{grid.shape[1]}_nz_{non_zero_count}_colors_{unique_colors}_sum_{grid_sum}"
    
    def _calculate_step_reward(self, old_grid: np.ndarray, new_grid: np.ndarray, original_input: np.ndarray) -> float:
        """Calculate reward for a single step"""
        # Reward based on how much the grid changed meaningfully
        change_count = np.sum(old_grid != new_grid)
        
        if change_count == 0:
            return -0.1  # Small penalty for no change
        elif change_count > original_input.size * 0.5:
            return -0.5  # Penalty for too much change
        else:
            return 0.1 * change_count / original_input.size  # Reward proportional to meaningful change
    
    def calculate_completeness(self, output_grid: np.ndarray, target_grid: np.ndarray) -> float:
        """Calculate completeness score"""
        if output_grid.shape != target_grid.shape:
            return 0.0
        
        matching_cells = np.sum(output_grid == target_grid)
        total_cells = target_grid.size
        return matching_cells / total_cells if total_cells > 0 else 0.0
    
    def update_fitness(self, puzzles_solved: int, total_puzzles: int, avg_completeness: float):
        """Update fitness using the refined formula"""
        w1, w2 = 0.7, 0.3  # Weights
        if total_puzzles > 0:
            self.fitness = w1 * (puzzles_solved / total_puzzles) + w2 * avg_completeness
        else:
            self.fitness = 0.0
    
    def select_tentacle_rl(self, state_key: str) -> int:
        """RL-based tentacle selection (FIXED: Now properly used)"""
        if len(self.tentacles) == 0:
            return 0
            
        if random.random() < self.epsilon:  # Exploration
            return random.randint(0, len(self.tentacles) - 1)
        
        # Exploitation: choose tentacle with highest Q-value
        if state_key not in self.q_values:
            self.q_values[state_key] = [0.0] * len(self.tentacles)
        
        return np.argmax(self.q_values[state_key])
    
    def update_q_value(self, state_key: str, action: int, reward: float):
        """Update Q-value for RL learning (FIXED: Now properly integrated)"""
        if len(self.tentacles) == 0:
            return
            
        if state_key not in self.q_values:
            self.q_values[state_key] = [0.0] * len(self.tentacles)
        
        if 0 <= action < len(self.q_values[state_key]):
            old_value = self.q_values[state_key][action]
            # Q-learning update rule
            self.q_values[state_key][action] = old_value + self.learning_rate * (reward - old_value)
    
    def train_neural_network_primitives(self, input_grid: np.ndarray, target_grid: np.ndarray):
        """Train neural network primitives (FIXED: Now implemented)"""
        for tentacle in self.tentacles:
            for primitive in tentacle.primitives:
                if isinstance(primitive, PatternDetectionNN) and not primitive.trained:
                    primitive.train_on_pattern(input_grid, target_grid)
    
    def apply_cost_function_penalties(self, input_grid: np.ndarray, target_grid: np.ndarray, threshold: float = 2.0):
        """Apply cost function penalties and remove poor tentacles (FIXED: Now implemented)"""
        tentacles_to_keep = []
        
        for tentacle in self.tentacles:
            cost = tentacle.calculate_cost(input_grid, target_grid)
            
            # Update tentacle weight based on cost
            tentacle.update_weight(cost)
            
            # Keep tentacle if cost is below threshold
            if cost < threshold:
                tentacles_to_keep.append(tentacle)
        
        # Ensure at least one tentacle remains
        if len(tentacles_to_keep) == 0 and len(self.tentacles) > 0:
            # Keep the best tentacle
            best_tentacle = min(self.tentacles, key=lambda t: t.cost_function_value)
            tentacles_to_keep.append(best_tentacle)
        
        self.tentacles = tentacles_to_keep

# ===== PUZZLE DATA LOADER =====
class PuzzleDataLoader:
    """Loads puzzle data from JSON files and training.txt"""
    
    @staticmethod
    def load_puzzle_file(file_path: str) -> Dict:
        """Load a single puzzle JSON file"""
        try:
            with open(file_path, 'r') as f:
                import json
                data = json.load(f)
            
            # Validate that all grids have matching input/output dimensions
            PuzzleDataLoader._validate_grid_dimensions(data, file_path)
            return data
        except Exception as e:
            print(f"Error loading puzzle file {file_path}: {e}")
            return {}
    
    @staticmethod
    def load_training_txt(file_path: str = "training.txt") -> Dict:
        """Load training data from training.txt file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Parse the training.txt content
            puzzles = PuzzleDataLoader._parse_training_txt(content)
            
            # Format as standard puzzle data structure
            formatted_data = {
                "train": puzzles,
                "test": []  # training.txt typically only has training data
            }
            
            print(f"Loaded {len(puzzles)} puzzles from {file_path}")
            return formatted_data
            
        except Exception as e:
            print(f"Error loading training.txt: {e}")
            return {}
    
    @staticmethod
    def _parse_training_txt(content: str) -> List[Dict]:
        """Parse training.txt content and extract puzzles"""
        puzzles = []
        
        # Split content by puzzle boundaries (usually marked by specific patterns)
        # This is a generic parser - you may need to adjust based on actual format
        
        try:
            # Method 1: Try JSON format first
            if content.strip().startswith('{') or content.strip().startswith('['):
                import json
                data = json.loads(content)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'train' in data:
                    return data.get('train', [])
            
            # Method 2: Parse custom text format
            lines = content.strip().split('\n')
            current_puzzle = {}
            current_grid = []
            in_input = False
            in_output = False
            
            for line in lines:
                line = line.strip()
                
                if not line:
                    continue
                
                # Look for puzzle separators
                if 'Input:' in line or 'input:' in line:
                    if current_puzzle and 'input' in current_puzzle and 'output' in current_puzzle:
                        puzzles.append(current_puzzle)
                    current_puzzle = {}
                    current_grid = []
                    in_input = True
                    in_output = False
                    continue
                
                if 'Output:' in line or 'output:' in line:
                    if in_input and current_grid:
                        current_puzzle['input'] = current_grid
                        current_grid = []
                    in_input = False
                    in_output = True
                    continue
                
                # Parse grid lines (assuming space or comma separated numbers)
                if in_input or in_output:
                    try:
                        # Try different parsing methods
                        if ',' in line:
                            row = [int(x.strip()) for x in line.split(',') if x.strip().isdigit()]
                        elif ' ' in line:
                            row = [int(x) for x in line.split() if x.isdigit()]
                        else:
                            # Single characters
                            row = [int(char) for char in line if char.isdigit()]
                        
                        if row:  # Only add non-empty rows
                            current_grid.append(row)
                            
                    except ValueError:
                        continue
            
            # Add the last puzzle
            if in_output and current_grid:
                current_puzzle['output'] = current_grid
            if current_puzzle and 'input' in current_puzzle and 'output' in current_puzzle:
                puzzles.append(current_puzzle)
                
        except Exception as e:
            print(f"Error parsing training.txt: {e}")
            
            # Fallback: Create some example puzzles if parsing fails
            puzzles = PuzzleDataLoader._get_fallback_training_data()
        
        return puzzles
    
    @staticmethod
    def _get_fallback_training_data() -> List[Dict]:
        """Provide comprehensive fallback training data with various puzzle types"""
        return [
            # Pattern Recognition - Fill enclosed regions
            {
                "input": [
                    [0, 0, 0, 0, 0],
                    [0, 3, 3, 3, 0],
                    [0, 3, 0, 3, 0],
                    [0, 3, 3, 3, 0],
                    [0, 0, 0, 0, 0]
                ],
                "output": [
                    [0, 0, 0, 0, 0],
                    [0, 3, 3, 3, 0],
                    [0, 3, 4, 3, 0],
                    [0, 3, 3, 3, 0],
                    [0, 0, 0, 0, 0]
                ]
            },
            # Color Replacement
            {
                "input": [
                    [1, 1, 2, 2],
                    [1, 1, 2, 2],
                    [3, 3, 4, 4],
                    [3, 3, 4, 4]
                ],
                "output": [
                    [5, 5, 2, 2],
                    [5, 5, 2, 2],
                    [3, 3, 6, 6],
                    [3, 3, 6, 6]
                ]
            },
            # Rotation Pattern
            {
                "input": [
                    [1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 2]
                ],
                "output": [
                    [0, 1, 0],
                    [0, 0, 0],
                    [2, 0, 0]
                ]
            },
            # Line Detection and Enhancement
            {
                "input": [
                    [0, 0, 0, 0],
                    [1, 1, 1, 0],
                    [0, 0, 0, 0],
                    [0, 1, 0, 2]
                ],
                "output": [
                    [0, 0, 0, 0],
                    [1, 1, 1, 0],
                    [0, 0, 0, 0],
                    [0, 2, 0, 2]
                ]
            },
            # Enclosed Shape Filling
            {
                "input": [
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]
                ],
                "output": [
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 1, 4, 4, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]
                ]
            },
            # Symmetry Pattern
            {
                "input": [
                    [1, 0, 2],
                    [0, 3, 0],
                    [0, 0, 0]
                ],
                "output": [
                    [1, 0, 2],
                    [0, 3, 0],
                    [2, 0, 1]
                ]
            },
            # Corner Pattern
            {
                "input": [
                    [0, 0, 0, 0],
                    [0, 1, 1, 0],
                    [0, 1, 1, 0],
                    [0, 0, 0, 0]
                ],
                "output": [
                    [2, 0, 0, 2],
                    [0, 1, 1, 0],
                    [0, 1, 1, 0],
                    [2, 0, 0, 2]
                ]
            },
            # Color Context Pattern
            {
                "input": [
                    [1, 2, 1, 2],
                    [2, 0, 2, 0],
                    [1, 2, 1, 2],
                    [2, 0, 2, 0]
                ],
                "output": [
                    [1, 2, 1, 2],
                    [2, 3, 2, 3],
                    [1, 2, 1, 2],
                    [2, 3, 2, 3]
                ]
            },
            # Shape Extension Pattern
            {
                "input": [
                    [0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]
                ],
                "output": [
                    [2, 1, 2],
                    [1, 1, 1],
                    [2, 1, 2]
                ]
            },
            # Diagonal Pattern
            {
                "input": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 1]
                ],
                "output": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ]
            }
        ]
    
    @staticmethod
    def load_all_puzzles_from_directory(directory_path: str) -> List[Dict]:
        """Load all JSON puzzle files from a directory"""
        all_puzzle_files = []
        
        try:
            json_files = glob.glob(os.path.join(directory_path, "*.json"))
            print(f"Found {len(json_files)} JSON files in {directory_path}")
            
            for file_path in json_files:
                try:
                    data = PuzzleDataLoader.load_puzzle_file(file_path)
                    if data:  # Only add if data was loaded successfully
                        all_puzzle_files.append({
                            'file_name': os.path.basename(file_path),
                            'file_path': file_path,
                            'data': data
                        })
                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error reading directory {directory_path}: {e}")
        
        return all_puzzle_files
    
    @staticmethod
    def combine_training_data(puzzle_files: List[Dict]) -> List[Dict]:
        """Combine training data from multiple puzzle files"""
        combined_training = []
        
        for puzzle_file in puzzle_files:
            data = puzzle_file['data']
            file_name = puzzle_file['file_name']
            
            # Get training examples
            training_examples = data.get('train', [])
            
            # Add source file information to each example
            for example in training_examples:
                example['source_file'] = file_name
                combined_training.append(example)
        
        print(f"Combined {len(combined_training)} training examples from {len(puzzle_files)} files")
        return combined_training
    
    @staticmethod
    def _validate_grid_dimensions(data: Dict, file_name: str):
        """Validate that input and output grids have matching dimensions"""
        if 'train' in data:
            for i, example in enumerate(data['train']):
                if 'input' in example and 'output' in example:
                    input_shape = np.array(example['input']).shape
                    output_shape = np.array(example['output']).shape
                    if input_shape != output_shape:
                        print(f"Warning: Training example {i} in {file_name} has mismatched dimensions: {input_shape} vs {output_shape}")
        
        if 'test' in data:
            for i, example in enumerate(data['test']):
                if 'input' in example and 'output' in example:
                    input_shape = np.array(example['input']).shape
                    output_shape = np.array(example['output']).shape
                    if input_shape != output_shape:
                        print(f"Warning: Test example {i} in {file_name} has mismatched dimensions: {input_shape} vs {output_shape}")

# ===== GRID TYPE CLASSIFIER =====
class GridTypeClassifier:
    """Meta-learning classifier to determine puzzle type and appropriate strategy"""
    
    def __init__(self):
        self.same_size_patterns = {}
        self.diff_size_patterns = {}
        self.pattern_weights = {}
        self.classification_history = []
    
    def classify_puzzle(self, input_grid: np.ndarray, output_grid: np.ndarray) -> str:
        """Classify puzzle type: 'same_size', 'diff_size', or 'complex'"""
        input_shape = input_grid.shape
        output_shape = output_grid.shape
        
        # Primary classification based on grid dimensions
        if input_shape == output_shape:
            puzzle_type = "same_size"
            sub_type = self._classify_same_size_puzzle(input_grid, output_grid)
        else:
            puzzle_type = "diff_size"
            sub_type = self._classify_diff_size_puzzle(input_grid, output_grid)
        
        # Store classification for learning
        classification_key = f"{puzzle_type}_{sub_type}"
        self.classification_history.append(classification_key)
        
        return puzzle_type, sub_type
    
    def _classify_same_size_puzzle(self, input_grid: np.ndarray, output_grid: np.ndarray) -> str:
        """Classify same-size puzzle subtypes"""
        diff_mask = input_grid != output_grid
        change_ratio = np.sum(diff_mask) / input_grid.size
        
        # Analyze transformation patterns
        if change_ratio < 0.1:
            return "local_transformation"  # Few cells changed
        elif change_ratio < 0.3:
            return "pattern_modification"  # Moderate changes
        elif change_ratio < 0.7:
            return "color_transformation"  # Many color changes
        else:
            return "structural_change"  # Major structural changes
    
    def _classify_diff_size_puzzle(self, input_grid: np.ndarray, output_grid: np.ndarray) -> str:
        """Classify different-size puzzle subtypes"""
        input_size = input_grid.size
        output_size = output_grid.size
        size_ratio = output_size / input_size
        
        if size_ratio > 4:
            return "pattern_repetition"  # Output much larger - likely tiling
        elif size_ratio > 1.5:
            return "pattern_extension"  # Output moderately larger
        elif size_ratio < 0.5:
            return "pattern_extraction"  # Output smaller - extraction/cropping
        else:
            return "rule_generation"  # Similar sizes but different - rule-based

# ===== SPECIALIZED TRAINERS =====
class SameSizeSpecialist:
    """Specialist for training on same-size grid puzzles"""
    
    def __init__(self):
        self.name = "SameSizeSpecialist"
        self.specialized_primitives = []
        self.local_pattern_cache = {}
        self.transformation_rules = {}
        self.success_metrics = {
            'local_transformation': 0.0,
            'pattern_modification': 0.0,
            'color_transformation': 0.0,
            'structural_change': 0.0
        }
    
    def initialize_primitives(self):
        """Initialize primitives optimized for same-size transformations"""
        self.specialized_primitives = [
            # Local transformation primitives
            RotatePrimitive(),
            FlipPrimitive(),
            MirrorPrimitive(),
            Rot90Primitive(),
            Rot180Primitive(),
            Rot270Primitive(),
            FlipHPrimitive(),
            FlipVPrimitive(),
            TransposePrimitive(),
            
            # Pattern modification primitives
            ColorDetectionPrimitive(),
            MaskC1Primitive(),
            MaskC2Primitive(),
            MaskC3Primitive(),
            Replace0to1Primitive(),
            Replace1to2Primitive(),
            
            # Advanced local operations
            FillEnclosedPrimitive(),
            ConditionalFillPrimitive(),
            PatternDetectionNN(),
            
            # Shift operations for local changes
            ShiftUpPadPrimitive(),
            ShiftDownPadPrimitive(),
            ShiftLeftPrimitive(),
            ShiftRightPrimitive()
        ]
        print(f"SameSizeSpecialist initialized with {len(self.specialized_primitives)} primitives")
    
    def train_on_puzzle(self, input_grid: np.ndarray, output_grid: np.ndarray, sub_type: str) -> List[Tentacle]:
        """Train specialist on a same-size puzzle"""
        if input_grid.shape != output_grid.shape:
            print("Warning: SameSizeSpecialist received mismatched grid sizes")
            return []
        
        tentacles = []
        
        # Create specialized tentacles based on sub-type
        if sub_type == "local_transformation":
            tentacles.extend(self._create_local_transformation_tentacles(input_grid, output_grid))
        elif sub_type == "pattern_modification":
            tentacles.extend(self._create_pattern_modification_tentacles(input_grid, output_grid))
        elif sub_type == "color_transformation":
            tentacles.extend(self._create_color_transformation_tentacles(input_grid, output_grid))
        elif sub_type == "structural_change":
            tentacles.extend(self._create_structural_change_tentacles(input_grid, output_grid))
        
        # Update success metrics
        for tentacle in tentacles:
            cost = tentacle.calculate_cost(input_grid, output_grid)
            success = cost < 1.0
            tentacle.update_success_rate(success)
            
            # Update specialist metrics
            if sub_type in self.success_metrics:
                current_metric = self.success_metrics[sub_type]
                self.success_metrics[sub_type] = 0.9 * current_metric + 0.1 * (1.0 if success else 0.0)
        
        return tentacles
    
    def _create_local_transformation_tentacles(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[Tentacle]:
        """Create tentacles for local transformations"""
        tentacles = []
        
        # Rotation-based tentacles
        rotation_prims = [RotatePrimitive(), Rot90Primitive(), Rot180Primitive(), Rot270Primitive()]
        tentacles.append(Tentacle("local_rotation", rotation_prims, "local_transformation"))
        
        # Flip-based tentacles
        flip_prims = [FlipPrimitive(), FlipHPrimitive(), FlipVPrimitive(), MirrorPrimitive()]
        tentacles.append(Tentacle("local_flip", flip_prims, "local_transformation"))
        
        # Shift-based tentacles
        shift_prims = [ShiftUpPadPrimitive(), ShiftDownPadPrimitive(), ShiftLeftPrimitive(), ShiftRightPrimitive()]
        tentacles.append(Tentacle("local_shift", shift_prims, "local_transformation"))
        
        return tentacles
    
    def _create_pattern_modification_tentacles(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[Tentacle]:
        """Create tentacles for pattern modifications"""
        tentacles = []
        
        # Pattern detection and modification
        pattern_prims = [PatternDetectionNN(), ColorDetectionPrimitive(), FillEnclosedPrimitive()]
        tentacles.append(Tentacle("pattern_detect_modify", pattern_prims, "pattern_modification"))
        
        # Color-based pattern modification
        color_prims = [MaskC1Primitive(), MaskC2Primitive(), MaskC3Primitive(), Replace0to1Primitive()]
        tentacles.append(Tentacle("color_pattern_modify", color_prims, "pattern_modification"))
        
        return tentacles
    
    def _create_color_transformation_tentacles(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[Tentacle]:
        """Create tentacles for color transformations"""
        tentacles = []
        
        # Color replacement tentacles
        color_replace_prims = [Replace0to1Primitive(), Replace1to2Primitive()]
        tentacles.append(Tentacle("color_replacement", color_replace_prims, "color_transformation"))
        
        # Color masking tentacles
        color_mask_prims = [MaskC1Primitive(), MaskC2Primitive(), MaskC3Primitive()]
        tentacles.append(Tentacle("color_masking", color_mask_prims, "color_transformation"))
        
        return tentacles
    
    def _create_structural_change_tentacles(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[Tentacle]:
        """Create tentacles for structural changes"""
        tentacles = []
        
        # Complex transformation tentacles
        struct_prims = [TransposePrimitive(), FlipDiagPrimitive(), FlipAntiDiagPrimitive()]
        tentacles.append(Tentacle("structural_transform", struct_prims, "structural_change"))
        
        # Fill and modify tentacles
        fill_prims = [FillEnclosedPrimitive(), ConditionalFillPrimitive()]
        tentacles.append(Tentacle("structural_fill", fill_prims, "structural_change"))
        
        return tentacles

class DiffSizeSpecialist:
    """Specialist for training on different-size grid puzzles"""
    
    def __init__(self):
        self.name = "DiffSizeSpecialist"
        self.specialized_primitives = []
        self.size_patterns = {}
        self.extraction_rules = {}
        self.success_metrics = {
            'pattern_repetition': 0.0,
            'pattern_extension': 0.0,
            'pattern_extraction': 0.0,
            'rule_generation': 0.0
        }
    
    def initialize_primitives(self):
        """Initialize primitives optimized for different-size transformations"""
        self.specialized_primitives = [
            # Size manipulation primitives
            TileToOutputPrimitive(),
            ExtractAndPlacePrimitive(),
            GenerateFilledGridPrimitive(),
            TilePatternPrimitive(),
            Tile3x3Primitive(),
            Tile2x2Primitive(),
            
            # Cropping and scaling
            CropCenterHalfPrimitive(),
            CropCenterThirdPrimitive(),
            CropPrimitive(),
            ExtendPrimitive(),
            ScalePrimitive(),
            
            # Pattern operations
            CopyPrimitive(),
            RepeatPrimitive(),
            TilePrimitive(),
            
            # Object-based operations
            ObjectsPrimitive(),
            BBoxPrimitive(),
            ObjectExtractionPrimitive()
        ]
        print(f"DiffSizeSpecialist initialized with {len(self.specialized_primitives)} primitives")
    
    def train_on_puzzle(self, input_grid: np.ndarray, output_grid: np.ndarray, sub_type: str) -> List[Tentacle]:
        """Train specialist on a different-size puzzle"""
        tentacles = []
        
        # Create specialized tentacles based on sub-type
        if sub_type == "pattern_repetition":
            tentacles.extend(self._create_repetition_tentacles(input_grid, output_grid))
        elif sub_type == "pattern_extension":
            tentacles.extend(self._create_extension_tentacles(input_grid, output_grid))
        elif sub_type == "pattern_extraction":
            tentacles.extend(self._create_extraction_tentacles(input_grid, output_grid))
        elif sub_type == "rule_generation":
            tentacles.extend(self._create_rule_generation_tentacles(input_grid, output_grid))
        
        # Update success metrics
        for tentacle in tentacles:
            # For different sizes, we need to be more flexible with cost calculation
            try:
                test_output = tentacle.execute(input_grid)
                if test_output.shape == output_grid.shape:
                    cost = tentacle.calculate_cost(input_grid, output_grid)
                    success = cost < 2.0  # More lenient threshold for different sizes
                else:
                    # Shape mismatch - partial success if shapes are reasonable
                    success = self._evaluate_shape_reasonableness(test_output.shape, output_grid.shape)
                    cost = 5.0 if not success else 1.5
                
                tentacle.update_success_rate(success)
                
                # Update specialist metrics
                if sub_type in self.success_metrics:
                    current_metric = self.success_metrics[sub_type]
                    self.success_metrics[sub_type] = 0.9 * current_metric + 0.1 * (1.0 if success else 0.0)
            except Exception as e:
                print(f"Error evaluating tentacle: {e}")
                tentacle.update_success_rate(False)
        
        return tentacles
    
    def _evaluate_shape_reasonableness(self, actual_shape: tuple, target_shape: tuple) -> bool:
        """Evaluate if actual shape is reasonably close to target"""
        actual_ratio = actual_shape[0] / actual_shape[1] if actual_shape[1] > 0 else 1
        target_ratio = target_shape[0] / target_shape[1] if target_shape[1] > 0 else 1
        
        # Check if aspect ratios are similar
        ratio_similarity = min(actual_ratio, target_ratio) / max(actual_ratio, target_ratio)
        
        # Check if sizes are in reasonable range
        actual_size = actual_shape[0] * actual_shape[1]
        target_size = target_shape[0] * target_shape[1]
        size_similarity = min(actual_size, target_size) / max(actual_size, target_size)
        
        return ratio_similarity > 0.5 and size_similarity > 0.25
    
    def _create_repetition_tentacles(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[Tentacle]:
        """Create tentacles for pattern repetition"""
        tentacles = []
        
        # Calculate repetition factors
        input_size = input_grid.size
        output_size = output_grid.size
        size_ratio = output_size / input_size
        
        # Tiling tentacles
        if size_ratio >= 4:
            tile_prims = [Tile2x2Primitive(), TilePatternPrimitive()]
            tentacles.append(Tentacle("tile_2x2", tile_prims, "pattern_repetition"))
        
        if size_ratio >= 9:
            tile_prims = [Tile3x3Primitive(), TilePatternPrimitive()]
            tentacles.append(Tentacle("tile_3x3", tile_prims, "pattern_repetition"))
        
        # Adaptive tiling
        tile_adaptive_prims = [TileToOutputPrimitive(), RepeatPrimitive()]
        tentacles.append(Tentacle("tile_adaptive", tile_adaptive_prims, "pattern_repetition"))
        
        return tentacles
    
    def _create_extension_tentacles(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[Tentacle]:
        """Create tentacles for pattern extension"""
        tentacles = []
        
        # Extension tentacles
        extend_prims = [ExtendPrimitive(), GenerateFilledGridPrimitive()]
        tentacles.append(Tentacle("pattern_extend", extend_prims, "pattern_extension"))
        
        # Scale-based extension
        scale_prims = [ScalePrimitive(), ExtendPrimitive()]
        tentacles.append(Tentacle("scale_extend", scale_prims, "pattern_extension"))
        
        # Extract and place extension
        extract_prims = [ExtractAndPlacePrimitive(), CopyPrimitive()]
        tentacles.append(Tentacle("extract_place", extract_prims, "pattern_extension"))
        
        return tentacles
    
    def _create_extraction_tentacles(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[Tentacle]:
        """Create tentacles for pattern extraction"""
        tentacles = []
        
        # Cropping tentacles
        crop_prims = [CropPrimitive(), CropCenterHalfPrimitive()]
        tentacles.append(Tentacle("crop_extract", crop_prims, "pattern_extraction"))
        
        crop_prims2 = [CropCenterThirdPrimitive(), ObjectExtractionPrimitive()]
        tentacles.append(Tentacle("crop_center", crop_prims2, "pattern_extraction"))
        
        # Object-based extraction
        obj_prims = [ObjectsPrimitive(), BBoxPrimitive(), ObjectExtractionPrimitive()]
        tentacles.append(Tentacle("object_extract", obj_prims, "pattern_extraction"))
        
        return tentacles
    
    def _create_rule_generation_tentacles(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[Tentacle]:
        """Create tentacles for rule-based generation"""
        tentacles = []
        
        # Rule-based generation tentacles
        rule_prims = [GenerateFilledGridPrimitive(), ConditionalFillPrimitive()]
        tentacles.append(Tentacle("rule_generate", rule_prims, "rule_generation"))
        
        # Pattern-based rules
        pattern_rule_prims = [PatternDetectionNN(), GenerateFilledGridPrimitive()]
        tentacles.append(Tentacle("pattern_rule", pattern_rule_prims, "rule_generation"))
        
        return tentacles

# ===== META-LEARNING TRAINER =====
class MetaLearningTrainer:
    """Meta-learning approach that coordinates specialists and learns from their performance"""
    
    def __init__(self):
        self.classifier = GridTypeClassifier()
        self.same_size_specialist = SameSizeSpecialist()
        self.diff_size_specialist = DiffSizeSpecialist()
        self.meta_knowledge = {}
        self.specialist_performance = {
            'same_size': {'puzzles_solved': 0, 'total_puzzles': 0, 'avg_performance': 0.0},
            'diff_size': {'puzzles_solved': 0, 'total_puzzles': 0, 'avg_performance': 0.0}
        }
        self.cross_specialist_learning = True
        
        # Initialize specialists
        self.same_size_specialist.initialize_primitives()
        self.diff_size_specialist.initialize_primitives()
    
    def train_on_puzzle_set(self, puzzle_data: List[Dict]) -> Dict:
        """Train on a set of puzzles using meta-learning approach"""
        results = {
            'same_size_results': [],
            'diff_size_results': [],
            'meta_insights': {},
            'specialist_performance': {}
        }
        
        print(f"Starting meta-learning training on {len(puzzle_data)} puzzles")
        
        for i, puzzle in enumerate(puzzle_data):
            try:
                # Extract training examples
                train_examples = puzzle.get('train', [])
                
                for example in train_examples:
                    input_grid = np.array(example['input'])
                    output_grid = np.array(example['output'])
                    
                    # Classify puzzle type
                    puzzle_type, sub_type = self.classifier.classify_puzzle(input_grid, output_grid)
                    
                    print(f"Puzzle {i+1}: {puzzle_type} - {sub_type}")
                    
                    # Train appropriate specialist
                    if puzzle_type == "same_size":
                        tentacles = self.same_size_specialist.train_on_puzzle(input_grid, output_grid, sub_type)
                        results['same_size_results'].append({
                            'tentacles': tentacles,
                            'sub_type': sub_type,
                            'performance': self._evaluate_tentacles(tentacles, input_grid, output_grid)
                        })
                        self.specialist_performance['same_size']['total_puzzles'] += 1
                        
                    elif puzzle_type == "diff_size":
                        tentacles = self.diff_size_specialist.train_on_puzzle(input_grid, output_grid, sub_type)
                        results['diff_size_results'].append({
                            'tentacles': tentacles,
                            'sub_type': sub_type,
                            'performance': self._evaluate_tentacles(tentacles, input_grid, output_grid)
                        })
                        self.specialist_performance['diff_size']['total_puzzles'] += 1
                    
                    # Cross-specialist learning
                    if self.cross_specialist_learning:
                        self._apply_cross_specialist_learning(puzzle_type, sub_type, input_grid, output_grid)
                        
            except Exception as e:
                print(f"Error processing puzzle {i+1}: {e}")
                continue
        
        # Update meta-knowledge
        self._update_meta_knowledge(results)
        
        # Calculate final performance metrics
        results['specialist_performance'] = self.specialist_performance.copy()
        
        print("Meta-learning training completed")
        print(f"Same-size specialist: {self.specialist_performance['same_size']['total_puzzles']} puzzles")
        print(f"Diff-size specialist: {self.specialist_performance['diff_size']['total_puzzles']} puzzles")
        
        return results
    
    def _evaluate_tentacles(self, tentacles: List[Tentacle], input_grid: np.ndarray, target_grid: np.ndarray) -> Dict:
        """Evaluate tentacle performance"""
        performance = {
            'total_tentacles': len(tentacles),
            'successful_tentacles': 0,
            'avg_cost': 0.0,
            'best_cost': float('inf')
        }
        
        total_cost = 0.0
        for tentacle in tentacles:
            try:
                output = tentacle.execute(input_grid)
                if output.shape == target_grid.shape:
                    cost = CostFunctionCalculator.calculate_cost(output, target_grid)
                    total_cost += cost
                    performance['best_cost'] = min(performance['best_cost'], cost)
                    
                    if cost < 2.0:  # Success threshold
                        performance['successful_tentacles'] += 1
                else:
                    total_cost += 10.0  # Penalty for shape mismatch
            except Exception:
                total_cost += 10.0  # Penalty for execution error
        
        performance['avg_cost'] = total_cost / len(tentacles) if tentacles else float('inf')
        return performance
    
    def _apply_cross_specialist_learning(self, puzzle_type: str, sub_type: str, input_grid: np.ndarray, output_grid: np.ndarray):
        """Apply insights from one specialist to another"""
        try:
            if puzzle_type == "same_size":
                # See if diff-size specialist techniques can help
                diff_tentacles = self.diff_size_specialist.train_on_puzzle(input_grid, output_grid, "rule_generation")
                for tentacle in diff_tentacles[:2]:  # Try first 2 tentacles
                    output = tentacle.execute(input_grid)
                    if output.shape == input_grid.shape:  # If it maintains size
                        cost = CostFunctionCalculator.calculate_cost(output, output_grid)
                        if cost < 1.5:  # If it performs well
                            # Add successful cross-specialist tentacle to same-size specialist
                            tentacle.specialization = "cross_specialist"
                            print(f"Cross-learning: Added diff-size technique to same-size specialist")
            
            elif puzzle_type == "diff_size":
                # See if same-size transformations can be part of the solution
                same_tentacles = self.same_size_specialist.train_on_puzzle(input_grid, input_grid, "local_transformation")
                # Apply transformations then size operations
                for tentacle in same_tentacles[:2]:
                    try:
                        transformed = tentacle.execute(input_grid)
                        # Now try size operations on transformed grid
                        size_tentacles = self.diff_size_specialist.train_on_puzzle(transformed, output_grid, "pattern_extension")
                        for size_tentacle in size_tentacles[:1]:
                            final_output = size_tentacle.execute(transformed)
                            if final_output.shape == output_grid.shape:
                                cost = CostFunctionCalculator.calculate_cost(final_output, output_grid)
                                if cost < 2.0:
                                    print(f"Cross-learning: Combined same-size + diff-size approach successful")
                    except Exception:
                        continue
                        
        except Exception as e:
            print(f"Cross-specialist learning error: {e}")
    
    def _update_meta_knowledge(self, results: Dict):
        """Update meta-knowledge based on training results"""
        # Analyze successful patterns
        successful_same_size = [r for r in results['same_size_results'] if r['performance']['successful_tentacles'] > 0]
        successful_diff_size = [r for r in results['diff_size_results'] if r['performance']['successful_tentacles'] > 0]
        
        # Update specialist performance
        if results['same_size_results']:
            avg_success_rate = len(successful_same_size) / len(results['same_size_results'])
            self.specialist_performance['same_size']['puzzles_solved'] = len(successful_same_size)
            self.specialist_performance['same_size']['avg_performance'] = avg_success_rate
        
        if results['diff_size_results']:
            avg_success_rate = len(successful_diff_size) / len(results['diff_size_results'])
            self.specialist_performance['diff_size']['puzzles_solved'] = len(successful_diff_size)
            self.specialist_performance['diff_size']['avg_performance'] = avg_success_rate
        
        # Store meta-insights
        self.meta_knowledge['best_same_size_subtypes'] = self._find_best_subtypes(successful_same_size)
        self.meta_knowledge['best_diff_size_subtypes'] = self._find_best_subtypes(successful_diff_size)
        
        print(f"Meta-knowledge updated: {len(self.meta_knowledge)} insights stored")
    
    def _find_best_subtypes(self, successful_results: List[Dict]) -> Dict:
        """Find which subtypes perform best"""
        subtype_performance = {}
        
        for result in successful_results:
            sub_type = result['sub_type']
            performance = result['performance']['successful_tentacles'] / result['performance']['total_tentacles']
            
            if sub_type not in subtype_performance:
                subtype_performance[sub_type] = []
            subtype_performance[sub_type].append(performance)
        
        # Calculate average performance per subtype
        avg_performance = {}
        for sub_type, performances in subtype_performance.items():
            avg_performance[sub_type] = sum(performances) / len(performances)
        
        return avg_performance
    
    def create_optimized_octopus(self, puzzle_type: str = None, sub_type: str = None) -> 'Octopus':
        """Create an octopus optimized for specific puzzle types"""
        import time
        octopus = Octopus(octopus_id=hash(f"{puzzle_type}_{sub_type}_{time.time()}") % 10000)
        
        if puzzle_type == "same_size":
            # Add best same-size tentacles
            specialist = self.same_size_specialist
            for prim_set in [specialist.specialized_primitives[i:i+3] for i in range(0, len(specialist.specialized_primitives), 3)]:
                tentacle = Tentacle(f"same_size_{len(octopus.tentacles)}", prim_set, puzzle_type)
                octopus.add_tentacle(tentacle)
                
        elif puzzle_type == "diff_size":
            # Add best diff-size tentacles
            specialist = self.diff_size_specialist
            for prim_set in [specialist.specialized_primitives[i:i+3] for i in range(0, len(specialist.specialized_primitives), 3)]:
                tentacle = Tentacle(f"diff_size_{len(octopus.tentacles)}", prim_set, puzzle_type)
                octopus.add_tentacle(tentacle)
        
        else:
            # Create hybrid octopus with best from both specialists
            same_prims = self.same_size_specialist.specialized_primitives[:6]
            diff_prims = self.diff_size_specialist.specialized_primitives[:6]
            
            for i, prim_set in enumerate([same_prims[i:i+2] for i in range(0, len(same_prims), 2)]):
                tentacle = Tentacle(f"hybrid_same_{i}", prim_set, "hybrid")
                octopus.add_tentacle(tentacle)
            
            for i, prim_set in enumerate([diff_prims[i:i+2] for i in range(0, len(diff_prims), 2)]):
                tentacle = Tentacle(f"hybrid_diff_{i}", prim_set, "hybrid")
                octopus.add_tentacle(tentacle)
        
        print(f"Created optimized octopus for {puzzle_type} with {len(octopus.tentacles)} tentacles")
        return octopus

# ===== MOTHER OCTOPUS (Enhanced Evolutionary Engine) =====
# ===== MOTHER OCTOPUS (Enhanced Evolutionary Engine) =====
class MotherOctopus:
    """Enhanced evolutionary engine with meta-learning and specialized training"""
    
    def __init__(self, population_size: int = 20, verbose: bool = True):
        self.population_size = population_size
        self.population = []
        self.generation = 0
        self.training_history = []
        self.best_octopus = None
        self.mutation_rate = 0.1
        self.elite_ratio = 0.2
        self.verbose = verbose  # Add verbose parameter
        
        # Enhanced components
        self.meta_trainer = MetaLearningTrainer()
        self.adaptive_generator = AdaptivePrimitiveGenerator()
        
        # All available primitives
        self.all_primitives = self._initialize_all_primitives()
        
        # Training performance tracking
        self.performance_tracker = {
            'same_size_success_rate': 0.0,
            'diff_size_success_rate': 0.0,
            'overall_improvement': 0.0,
            'generations_trained': 0
        }
        
        # Add compatibility attributes for the old interface
        self.training_data_same_shape = []
        self.training_data_diff_shape = []
        self.test_data = []
        self.data_loader = PuzzleDataLoader()
        
        if self.verbose:
            print(f"MotherOctopus initialized with meta-learning capabilities")
            print(f"Population size: {population_size}")
            print(f"Total primitives available: {len(self.all_primitives)}")
    
    def _initialize_all_primitives(self) -> List[Primitive]:
        """Initialize comprehensive primitive library"""
        primitives = []
        
        # Transform Primitives
        primitives.extend([
            RotatePrimitive(), FlipPrimitive(), MirrorPrimitive(),
            Rot90Primitive(), Rot180Primitive(), Rot270Primitive(),
            FlipHPrimitive(), FlipVPrimitive(), TransposePrimitive(),
            FlipDiagPrimitive(), FlipAntiDiagPrimitive()
        ])
        
        # Size Manipulation Primitives
        primitives.extend([
            TileToOutputPrimitive(), ExtractAndPlacePrimitive(), GenerateFilledGridPrimitive(),
            TilePatternPrimitive(), Tile3x3Primitive(), Tile2x2Primitive(),
            CropCenterHalfPrimitive(), CropCenterThirdPrimitive(), CropPrimitive(),
            ExtendPrimitive(), ScalePrimitive()
        ])
        
        # Color and Pattern Primitives
        primitives.extend([
            ColorDetectionPrimitive(), MaskC1Primitive(), MaskC2Primitive(), MaskC3Primitive(),
            Replace0to1Primitive(), Replace1to2Primitive(), HoleMaskPrimitive()
        ])
        
        # Advanced Primitives
        primitives.extend([
            PatternDetectionNN(), FillEnclosedPrimitive(), ConditionalFillPrimitive(),
            FloodFillPrimitive(), ConnectedComponentsPrimitive(), SymmetryDetectionPrimitive(),
            ObjectExtractionPrimitive()
        ])
        
        # Logical Primitives
        primitives.extend([
            ANDPrimitive(), ORPrimitive(), IntersectionPrimitive(),
            UnionPrimitive(), DifferencePrimitive()
        ])
        
        # Movement Primitives
        primitives.extend([
            ShiftUpPadPrimitive(), ShiftDownPadPrimitive(),
            ShiftLeftPrimitive(), ShiftRightPrimitive(), TranslatePrimitive()
        ])
        
        # Pattern Operation Primitives
        primitives.extend([
            CopyPrimitive(), RepeatPrimitive(), TilePrimitive()
        ])
        
        return primitives
    
    def train_with_meta_learning(self, puzzle_data: List[Dict], generations: int = 10) -> Dict:
        """Enhanced training using meta-learning approach"""
        if self.verbose:
            print(f"Starting meta-learning training for {generations} generations")
        
        # Phase 1: Meta-learning training to understand puzzle types
        if self.verbose:
            print("Phase 1: Meta-learning analysis...")
        meta_results = self.meta_trainer.train_on_puzzle_set(puzzle_data)
        
        # Phase 2: Create specialized population based on meta-insights
        if self.verbose:
            print("Phase 2: Creating specialized population...")
        self._create_specialized_population(meta_results)
        
        # Phase 3: Evolutionary training with specialization awareness
        if self.verbose:
            print("Phase 3: Evolutionary training with specialization...")
        evolution_results = self._evolutionary_training_with_specialization(puzzle_data, generations)
        
        # Phase 4: Final optimization and cross-training
        if self.verbose:
            print("Phase 4: Final optimization...")
        final_results = self._final_optimization(puzzle_data)
        
        # Compile comprehensive results
        training_results = {
            'meta_learning_results': meta_results,
            'evolution_results': evolution_results,
            'final_results': final_results,
            'performance_tracker': self.performance_tracker.copy(),
            'best_octopus': self.best_octopus,
            'population_summary': self._get_population_summary()
        }
        
        self.performance_tracker['generations_trained'] = generations
        
        if self.verbose:
            print("Meta-learning training completed!")
            self._print_training_summary(training_results)
        
        return training_results
    
    def _create_specialized_population(self, meta_results: Dict):
        """Create population with specialized octopi based on meta-learning insights"""
        self.population = []
        
        # Calculate distribution based on puzzle types encountered
        same_size_count = len(meta_results['same_size_results'])
        diff_size_count = len(meta_results['diff_size_results'])
        total_puzzles = same_size_count + diff_size_count
        
        if total_puzzles > 0:
            same_size_ratio = same_size_count / total_puzzles
            diff_size_ratio = diff_size_count / total_puzzles
        else:
            same_size_ratio = diff_size_ratio = 0.5
        
        # Create specialized octopi
        same_size_octopi = int(self.population_size * same_size_ratio * 0.6)  # 60% for dominant type
        diff_size_octopi = int(self.population_size * diff_size_ratio * 0.6)
        hybrid_octopi = self.population_size - same_size_octopi - diff_size_octopi
        
        if self.verbose:
            print(f"Creating specialized population:")
            print(f"  Same-size specialists: {same_size_octopi}")
            print(f"  Diff-size specialists: {diff_size_octopi}")
            print(f"  Hybrid octopi: {hybrid_octopi}")
        
        # Create same-size specialists
        for i in range(same_size_octopi):
            octopus = self.meta_trainer.create_optimized_octopus("same_size")
            octopus.specialization = "same_size"
            self.population.append(octopus)
        
        # Create diff-size specialists
        for i in range(diff_size_octopi):
            octopus = self.meta_trainer.create_optimized_octopus("diff_size")
            octopus.specialization = "diff_size"
            self.population.append(octopus)
        
        # Create hybrid octopi
        for i in range(hybrid_octopi):
            octopus = self.meta_trainer.create_optimized_octopus()
            octopus.specialization = "hybrid"
            self.population.append(octopus)
        
        if self.verbose:
            print(f"Specialized population created: {len(self.population)} octopi")
    
    def _evolutionary_training_with_specialization(self, puzzle_data: List[Dict], generations: int) -> Dict:
        """Evolutionary training that respects specializations"""
        evolution_results = {
            'generation_scores': [],
            'specialization_performance': {
                'same_size': [],
                'diff_size': [],
                'hybrid': []
            },
            'best_performers': []
        }
        
        for gen in range(generations):
            print(f"\nGeneration {gen + 1}/{generations}")
            
            # Evaluate population on appropriate puzzle types
            generation_scores = self._evaluate_population_specialized(puzzle_data)
            evolution_results['generation_scores'].append(generation_scores)
            
            # Track specialization performance
            self._track_specialization_performance(evolution_results, generation_scores)
            
            # Evolve population with specialization awareness
            self._evolve_population_specialized()
            
            # Update best octopus
            best_octopus = max(self.population, key=lambda o: o.fitness)
            if self.best_octopus is None or best_octopus.fitness > self.best_octopus.fitness:
                self.best_octopus = best_octopus
                evolution_results['best_performers'].append({
                    'generation': gen + 1,
                    'fitness': best_octopus.fitness,
                    'specialization': getattr(best_octopus, 'specialization', 'unknown')
                })
            
            # Adaptive mutation rate
            if gen > 0:
                improvement = generation_scores['avg_fitness'] - evolution_results['generation_scores'][-2]['avg_fitness']
                if improvement < 0.01:  # Stagnation
                    self.mutation_rate = min(0.3, self.mutation_rate * 1.1)
                else:
                    self.mutation_rate = max(0.05, self.mutation_rate * 0.95)
            
            print(f"  Avg fitness: {generation_scores['avg_fitness']:.3f}")
            print(f"  Best fitness: {generation_scores['best_fitness']:.3f}")
            print(f"  Mutation rate: {self.mutation_rate:.3f}")
        
        return evolution_results
    
    def _evaluate_population_specialized(self, puzzle_data: List[Dict]) -> Dict:
        """Evaluate population considering their specializations"""
        fitness_scores = []
        specialization_scores = {'same_size': [], 'diff_size': [], 'hybrid': []}
        
        for octopus in self.population:
            octopus_fitness = 0.0
            octopus_completeness = []
            puzzles_solved = 0
            total_puzzles = 0
            
            specialization = getattr(octopus, 'specialization', 'hybrid')
            
            for puzzle in puzzle_data:
                train_examples = puzzle.get('train', [])
                
                for example in train_examples:
                    try:
                        input_grid = np.array(example['input'])
                        output_grid = np.array(example['output'])
                        
                        # Classify puzzle type
                        puzzle_type, sub_type = self.meta_trainer.classifier.classify_puzzle(input_grid, output_grid)
                        
                        # Skip if octopus specialization doesn't match (for specialists)
                        if (specialization == "same_size" and puzzle_type != "same_size") or \
                           (specialization == "diff_size" and puzzle_type != "diff_size"):
                            continue
                        
                        total_puzzles += 1
                        
                        # Execute octopus
                        output = octopus.execute(input_grid)
                        
                        # Calculate completeness
                        completeness = octopus.calculate_completeness(output, output_grid)
                        octopus_completeness.append(completeness)
                        
                        # Check if solved (high completeness threshold)
                        if completeness > 0.9:
                            puzzles_solved += 1
                        
                        # Train neural network primitives
                        octopus.train_neural_network_primitives(input_grid, output_grid)
                        
                        # Apply cost function penalties
                        octopus.apply_cost_function_penalties(input_grid, output_grid)
                        
                    except Exception as e:
                        total_puzzles += 1  # Count failed attempts
                        octopus_completeness.append(0.0)
                        continue
            
            # Calculate fitness
            avg_completeness = np.mean(octopus_completeness) if octopus_completeness else 0.0
            octopus.update_fitness(puzzles_solved, total_puzzles, avg_completeness)
            
            fitness_scores.append(octopus.fitness)
            specialization_scores[specialization].append(octopus.fitness)
        
        return {
            'avg_fitness': np.mean(fitness_scores),
            'best_fitness': max(fitness_scores) if fitness_scores else 0.0,
            'specialization_avg': {k: np.mean(v) if v else 0.0 for k, v in specialization_scores.items()}
        }
    
    def _track_specialization_performance(self, evolution_results: Dict, generation_scores: Dict):
        """Track performance by specialization type"""
        for spec_type, avg_score in generation_scores['specialization_avg'].items():
            evolution_results['specialization_performance'][spec_type].append(avg_score)
    
    def _evolve_population_specialized(self):
        """Evolve population while maintaining specialization diversity with rank selection"""
        # Sort by fitness
        self.population.sort(key=lambda o: o.fitness, reverse=True)
        
        # Keep elite from each specialization
        elite_count = max(1, int(self.population_size * self.elite_ratio))
        
        # Group by specialization
        spec_groups = {'same_size': [], 'diff_size': [], 'hybrid': []}
        for octopus in self.population:
            spec = getattr(octopus, 'specialization', 'hybrid')
            spec_groups[spec].append(octopus)
        
        # Keep top performers from each group using rank selection
        new_population = []
        for spec_type, octopi in spec_groups.items():
            if octopi:
                octopi.sort(key=lambda o: o.fitness, reverse=True)
                # Use rank selection instead of just taking top third
                selected = self._rank_selection(octopi, max(1, len(octopi) // 2))
                new_population.extend(selected)
        
        # Fill remaining slots with mutations and crossovers using rank selection
        while len(new_population) < self.population_size:
            if random.random() < 0.7:  # 70% mutation, 30% crossover
                # Rank-based parent selection for mutation
                parent = self._rank_selection(self.population[:elite_count], 1)[0]
                child = self._mutate_octopus_specialized(parent)
            else:
                # Rank-based parent selection for crossover
                parents = self._rank_selection(self.population[:elite_count], 2)
                if len(parents) >= 2:
                    child = self._crossover_octopi_specialized(parents[0], parents[1])
                else:
                    # Fallback to mutation if we can't get 2 parents
                    parent = parents[0] if parents else self.population[0]
                    child = self._mutate_octopus_specialized(parent)
            
            new_population.append(child)
        
        self.population = new_population[:self.population_size]
        self.generation += 1
        
        if self.verbose:
            fitness_stats = self._get_population_fitness_stats()
            print(f"  Generation {self.generation}: Avg={fitness_stats['avg']:.3f}, "
                  f"Best={fitness_stats['best']:.3f}, Diversity={fitness_stats['diversity']:.3f}")
    
    def _rank_selection(self, octopi: List['Octopus'], num_select: int) -> List['Octopus']:
        """Select octopi using rank-based selection to reduce domination"""
        if not octopi:
            return []
        
        if len(octopi) <= num_select:
            return octopi.copy()
        
        # Sort by fitness (descending)
        sorted_octopi = sorted(octopi, key=lambda o: o.fitness, reverse=True)
        
        # Calculate rank-based probabilities
        n = len(sorted_octopi)
        rank_probs = []
        
        for i in range(n):
            # Linear ranking: P(rank_i) = (2 - s + 2*(s-1)*(n-i-1)/(n-1)) / n
            # where s is selection pressure (1.5 for moderate pressure)
            s = 1.5
            rank_prob = (2 - s + 2 * (s - 1) * (n - i - 1) / (n - 1)) / n
            rank_probs.append(rank_prob)
        
        # Normalize probabilities
        total_prob = sum(rank_probs)
        if total_prob > 0:
            rank_probs = [p / total_prob for p in rank_probs]
        else:
            rank_probs = [1/n] * n  # Equal probability fallback
        
        # Select based on rank probabilities
        selected = []
        for _ in range(num_select):
            # Roulette wheel selection based on rank probabilities
            rand_val = random.random()
            cumulative_prob = 0
            
            for i, prob in enumerate(rank_probs):
                cumulative_prob += prob
                if rand_val <= cumulative_prob:
                    selected.append(sorted_octopi[i])
                    break
            else:
                # Fallback: select last octopus
                selected.append(sorted_octopi[-1])
        
        return selected
    
    def _get_population_fitness_stats(self) -> Dict:
        """Get fitness statistics for the population"""
        if not self.population:
            return {'avg': 0.0, 'best': 0.0, 'worst': 0.0, 'diversity': 0.0}
        
        fitness_values = [o.fitness for o in self.population]
        
        avg_fitness = np.mean(fitness_values)
        best_fitness = max(fitness_values)
        worst_fitness = min(fitness_values)
        
        # Calculate diversity as standard deviation of fitness
        diversity = np.std(fitness_values) if len(fitness_values) > 1 else 0.0
        
        return {
            'avg': avg_fitness,
            'best': best_fitness,
            'worst': worst_fitness,
            'diversity': diversity
        }
    
    def _mutate_octopus_specialized(self, parent: 'Octopus') -> 'Octopus':
        """Mutate octopus while preserving specialization with improved diversity"""
        child = Octopus(octopus_id=random.randint(1000, 9999))
        child.specialization = getattr(parent, 'specialization', 'hybrid')
        
        # Copy and mutate tentacles with dynamic mutation rate
        mutation_strength = random.uniform(0.5, 1.5)  # Variable mutation strength
        
        for tentacle in parent.tentacles:
            # Higher chance of mutation for poor-performing tentacles
            base_mutation_rate = self.mutation_rate
            if tentacle.success_rate < 0.3:
                effective_mutation_rate = min(0.8, base_mutation_rate * 2.0)
            else:
                effective_mutation_rate = base_mutation_rate
            
            if random.random() < effective_mutation_rate:
                # Mutate this tentacle
                new_primitives = tentacle.primitives.copy()
                
                # Multiple mutation operations
                for _ in range(int(mutation_strength)):
                    mutation_type = random.choice(['remove', 'add', 'replace', 'reorder'])
                    
                    if mutation_type == 'remove' and len(new_primitives) > 1:
                        new_primitives.pop(random.randint(0, len(new_primitives) - 1))
                    
                    elif mutation_type == 'add':
                        # Add a primitive from appropriate specialist
                        if child.specialization == "same_size":
                            candidates = self.meta_trainer.same_size_specialist.specialized_primitives
                        elif child.specialization == "diff_size":
                            candidates = self.meta_trainer.diff_size_specialist.specialized_primitives
                        else:
                            candidates = self.all_primitives
                        
                        new_primitive = random.choice(candidates)
                        new_primitives.append(new_primitive)
                    
                    elif mutation_type == 'replace' and new_primitives:
                        idx = random.randint(0, len(new_primitives) - 1)
                        if child.specialization == "same_size":
                            candidates = self.meta_trainer.same_size_specialist.specialized_primitives
                        elif child.specialization == "diff_size":
                            candidates = self.meta_trainer.diff_size_specialist.specialized_primitives
                        else:
                            candidates = self.all_primitives
                        
                        new_primitives[idx] = random.choice(candidates)
                    
                    elif mutation_type == 'reorder' and len(new_primitives) > 1:
                        # Shuffle the order of primitives
                        random.shuffle(new_primitives)
                
                # Create new tentacle with mutated primitives
                new_tentacle = Tentacle(f"mutated_{tentacle.name}_{random.randint(100,999)}", 
                                      new_primitives, tentacle.specialization)
                # Inherit some parameters but add noise
                new_tentacle.parameters = tentacle.parameters.copy()
                for key, value in new_tentacle.parameters.items():
                    if isinstance(value, (int, float)):
                        noise = random.uniform(-0.2, 0.2) * value
                        new_tentacle.parameters[key] = max(0, value + noise)
                
                child.add_tentacle(new_tentacle)
            else:
                # Keep tentacle as is but create a copy
                copied_tentacle = Tentacle(tentacle.name, tentacle.primitives.copy(), tentacle.specialization)
                copied_tentacle.parameters = tentacle.parameters.copy()
                copied_tentacle.success_rate = tentacle.success_rate
                child.add_tentacle(copied_tentacle)
        
        # Ensure child has at least one tentacle
        if len(child.tentacles) == 0:
            # Add a random tentacle based on specialization
            if child.specialization == "same_size":
                primitives = random.sample(self.meta_trainer.same_size_specialist.specialized_primitives, 
                                         min(3, len(self.meta_trainer.same_size_specialist.specialized_primitives)))
            elif child.specialization == "diff_size":
                primitives = random.sample(self.meta_trainer.diff_size_specialist.specialized_primitives,
                                         min(3, len(self.meta_trainer.diff_size_specialist.specialized_primitives)))
            else:
                primitives = random.sample(self.all_primitives, min(3, len(self.all_primitives)))
            
            emergency_tentacle = Tentacle("emergency", primitives, child.specialization)
            child.add_tentacle(emergency_tentacle)
        
        return child
    
    def _crossover_octopi_specialized(self, parent1: 'Octopus', parent2: 'Octopus') -> 'Octopus':
        """Crossover while maintaining specialization"""
        child = Octopus(octopus_id=random.randint(1000, 9999))
        child.specialization = getattr(parent1, 'specialization', 'hybrid')
        
        # Combine tentacles from both parents
        all_tentacles = parent1.tentacles + parent2.tentacles
        random.shuffle(all_tentacles)
        
        # Select subset of tentacles
        max_tentacles = min(8, len(all_tentacles))
        selected_count = random.randint(max(1, max_tentacles // 2), max_tentacles)
        
        for tentacle in all_tentacles[:selected_count]:
            child.add_tentacle(tentacle)
        
        return child
    
    def _final_optimization(self, puzzle_data: List[Dict]) -> Dict:
        """Final optimization phase with cross-training"""
        print("Starting final optimization with cross-training...")
        
        final_results = {
            'cross_training_improvements': 0,
            'adaptive_primitives_generated': 0,
            'final_population_performance': {}
        }
        
        # Apply cross-training between specialists
        improvements = 0
        for octopus in self.population:
            if hasattr(octopus, 'specialization'):
                old_fitness = octopus.fitness
                
                # Try techniques from other specialists
                if octopus.specialization == "same_size":
                    # Add some diff-size techniques
                    diff_tentacles = self.meta_trainer.diff_size_specialist.train_on_puzzle(
                        np.ones((3, 3)), np.ones((3, 3)), "rule_generation"
                    )
                    if diff_tentacles:
                        octopus.add_tentacle(diff_tentacles[0])
                
                elif octopus.specialization == "diff_size":
                    # Add some same-size techniques
                    same_tentacles = self.meta_trainer.same_size_specialist.train_on_puzzle(
                        np.ones((3, 3)), np.ones((3, 3)), "local_transformation"
                    )
                    if same_tentacles:
                        octopus.add_tentacle(same_tentacles[0])
                
                # Re-evaluate
                new_fitness = self._quick_evaluate_octopus(octopus, puzzle_data[:5])  # Quick eval on subset
                if new_fitness > old_fitness:
                    improvements += 1
                    octopus.fitness = new_fitness
        
        final_results['cross_training_improvements'] = improvements
        
        # Generate adaptive primitives for novel patterns
        for puzzle in puzzle_data[:10]:  # Check first 10 puzzles for novel patterns
            for example in puzzle.get('train', []):
                try:
                    input_grid = np.array(example['input'])
                    output_grid = np.array(example['output'])
                    
                    if self.adaptive_generator.analyze_puzzle_novelty(input_grid, output_grid):
                        new_primitives = self.adaptive_generator.generate_new_primitives(input_grid, output_grid)
                        final_results['adaptive_primitives_generated'] += len(new_primitives)
                        
                        # Add adaptive primitives to best octopi
                        for octopus in self.population[:3]:  # Top 3 octopi
                            if new_primitives:
                                adaptive_tentacle = Tentacle("adaptive", new_primitives[:2], "adaptive")
                                octopus.add_tentacle(adaptive_tentacle)
                
                except Exception as e:
                    continue
        
        # Final population performance assessment
        final_results['final_population_performance'] = self._evaluate_population_specialized(puzzle_data)
        
        print(f"Final optimization completed:")
        print(f"  Cross-training improvements: {improvements}")
        print(f"  Adaptive primitives generated: {final_results['adaptive_primitives_generated']}")
        
        return final_results
    
    def _quick_evaluate_octopus(self, octopus: 'Octopus', puzzle_subset: List[Dict]) -> float:
        """Quick evaluation of octopus on puzzle subset"""
        completeness_scores = []
        
        for puzzle in puzzle_subset:
            for example in puzzle.get('train', [])[:2]:  # Max 2 examples per puzzle
                try:
                    input_grid = np.array(example['input'])
                    output_grid = np.array(example['output'])
                    
                    output = octopus.execute(input_grid)
                    completeness = octopus.calculate_completeness(output, output_grid)
                    completeness_scores.append(completeness)
                    
                except Exception:
                    completeness_scores.append(0.0)
        
        return np.mean(completeness_scores) if completeness_scores else 0.0
    
    def _get_population_summary(self) -> Dict:
        """Get summary of current population"""
        summary = {
            'total_octopi': len(self.population),
            'specialization_counts': {'same_size': 0, 'diff_size': 0, 'hybrid': 0},
            'avg_fitness': 0.0,
            'best_fitness': 0.0,
            'avg_tentacles_per_octopus': 0.0
        }
        
        fitness_scores = []
        tentacle_counts = []
        
        for octopus in self.population:
            spec = getattr(octopus, 'specialization', 'hybrid')
            summary['specialization_counts'][spec] += 1
            fitness_scores.append(octopus.fitness)
            tentacle_counts.append(len(octopus.tentacles))
        
        summary['avg_fitness'] = np.mean(fitness_scores) if fitness_scores else 0.0
        summary['best_fitness'] = max(fitness_scores) if fitness_scores else 0.0
        summary['avg_tentacles_per_octopus'] = np.mean(tentacle_counts) if tentacle_counts else 0.0
        
        return summary
    
    def _print_training_summary(self, training_results: Dict):
        """Print comprehensive training summary"""
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        meta_results = training_results['meta_learning_results']
        print(f"Meta-learning Phase:")
        print(f"  Same-size puzzles processed: {len(meta_results['same_size_results'])}")
        print(f"  Diff-size puzzles processed: {len(meta_results['diff_size_results'])}")
        
        evolution_results = training_results['evolution_results']
        print(f"\nEvolutionary Training:")
        print(f"  Generations completed: {len(evolution_results['generation_scores'])}")
        if evolution_results['generation_scores']:
            final_gen = evolution_results['generation_scores'][-1]
            print(f"  Final avg fitness: {final_gen['avg_fitness']:.3f}")
            print(f"  Final best fitness: {final_gen['best_fitness']:.3f}")
        
        final_results = training_results['final_results']
        print(f"\nFinal Optimization:")
        print(f"  Cross-training improvements: {final_results['cross_training_improvements']}")
        print(f"  Adaptive primitives generated: {final_results['adaptive_primitives_generated']}")
        
        pop_summary = training_results['population_summary']
        print(f"\nFinal Population:")
        print(f"  Total octopi: {pop_summary['total_octopi']}")
        print(f"  Same-size specialists: {pop_summary['specialization_counts']['same_size']}")
        print(f"  Diff-size specialists: {pop_summary['specialization_counts']['diff_size']}")
        print(f"  Hybrid octopi: {pop_summary['specialization_counts']['hybrid']}")
        print(f"  Best octopus fitness: {pop_summary['best_fitness']:.3f}")
        
        print("="*60)

    def solve_puzzle(self, input_grid: np.ndarray, use_best_octopus: bool = True) -> np.ndarray:
        """Solve a puzzle using the trained population"""
        if use_best_octopus and self.best_octopus:
            return self.best_octopus.execute(input_grid)
        
        # If no best octopus, try to classify and use appropriate specialist
        if len(self.population) > 0:
            # Try to determine puzzle type
            # For solving, we don't have the output, so we make educated guesses
            best_candidates = []
            
            # Try specialists first
            for octopus in self.population:
                if hasattr(octopus, 'specialization'):
                    # Simple heuristic: smaller grids often benefit from diff-size specialists
                    if input_grid.size < 20 and octopus.specialization == "diff_size":
                        best_candidates.append(octopus)
                    elif input_grid.size >= 20 and octopus.specialization == "same_size":
                        best_candidates.append(octopus)
                    elif octopus.specialization == "hybrid":
                        best_candidates.append(octopus)
            
            # If no good candidates, use top performers
            if not best_candidates:
                sorted_pop = sorted(self.population, key=lambda o: o.fitness, reverse=True)
                best_candidates = sorted_pop[:3]
            
            # Try each candidate and return the most "reasonable" result
            for octopus in best_candidates[:3]:
                try:
                    result = octopus.execute(input_grid)
                    if result is not None and result.size > 0:
                        return result
                except Exception:
                    continue
        
        # Fallback: return input if all else fails
        return input_grid.copy()
    
    # ===== COMPATIBILITY METHODS FOR OLD INTERFACE =====
    def load_all_puzzles_from_directory(self, directory_path: str):
        """Load all puzzles from directory (compatibility method)"""
        if self.verbose:
            print(f"Loading all puzzles from directory: {directory_path}")
        
        # Load puzzle files using the data loader
        puzzle_files = self.data_loader.load_all_puzzles_from_directory(directory_path)
        
        # Convert to the format expected by the meta-learning system
        puzzle_data = []
        for puzzle_file in puzzle_files:
            data = puzzle_file.get('data', {})
            train_examples = data.get('train', [])
            
            if train_examples:
                # Convert each training example
                for example in train_examples:
                    input_grid = np.array(example['input'])
                    output_grid = np.array(example['output'])
                    
                    # Categorize by grid size
                    if input_grid.shape == output_grid.shape:
                        self.training_data_same_shape.append(example)
                    else:
                        self.training_data_diff_shape.append(example)
                
                # Also add to puzzle_data for meta-learning
                puzzle_data.append(data)
        
        if self.verbose:
            print(f"Loaded {len(self.training_data_same_shape)} same-shape and {len(self.training_data_diff_shape)} diff-shape puzzles")
        
        return puzzle_data
    
    def load_puzzle_data(self, puzzle_data: Dict):
        """Load puzzle data (compatibility method)"""
        train_examples = puzzle_data.get('train', [])
        test_examples = puzzle_data.get('test', [])
        
        for example in train_examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            if input_grid.shape == output_grid.shape:
                self.training_data_same_shape.append(example)
            else:
                self.training_data_diff_shape.append(example)
        
        self.test_data.extend(test_examples)
    
    def evolve_generation(self, training_data: List[Dict]):
        """Evolve one generation (compatibility method)"""
        # Convert training data to the format expected by meta-learning
        puzzle_data = [{"train": training_data}]
        
        # Use meta-learning training for one generation
        try:
            results = self.train_with_meta_learning(puzzle_data, generations=1)
            return results
        except Exception as e:
            if self.verbose:
                print(f"Evolution error: {e}")
            return {}
    
    def train(self, generations: int = 10):
        """Train the system (compatibility method)"""
        if not self.training_data_same_shape and not self.training_data_diff_shape:
            if self.verbose:
                print("No training data available!")
            return {}
        
        # Combine training data
        all_training = self.training_data_same_shape + self.training_data_diff_shape
        puzzle_data = [{"train": all_training}]
        
        if self.verbose:
            print(f"Training for {generations} generations on {len(all_training)} puzzles...")
        
        # Use meta-learning training
        return self.train_with_meta_learning(puzzle_data, generations=generations)
    
    def test_on_puzzles(self, test_on_training: bool = False):
        """Test the system on puzzles (compatibility method)"""
        data_to_test = self.training_data_same_shape if test_on_training else self.test_data
        
        if not data_to_test:
            return {"accuracy": 0.0, "total_tested": 0, "correct": 0}
        
        correct = 0
        total = len(data_to_test)
        
        for puzzle in data_to_test:
            try:
                input_grid = np.array(puzzle['input'])
                target_grid = np.array(puzzle['output'])
                
                result = self.solve_puzzle(input_grid)
                
                if result.shape == target_grid.shape:
                    accuracy = np.mean(result == target_grid)
                    if accuracy > 0.9:
                        correct += 1
            except Exception:
                continue
        
        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "total_tested": total,
            "correct": correct
        }
    
    def get_system_statistics(self):
        """Get system statistics (compatibility method)"""
        return {
            'population_size': self.population_size,
            'generation': self.generation,
            'best_fitness': self.best_octopus.fitness if self.best_octopus else 0.0,
            'total_training_puzzles': len(self.training_data_same_shape) + len(self.training_data_diff_shape),
            'same_shape_puzzles': len(self.training_data_same_shape),
            'diff_shape_puzzles': len(self.training_data_diff_shape),
            'performance_tracker': self.performance_tracker.copy()
        }
    
    def count_puzzles_solved(self, test_on_training: bool = False):
        """Count puzzles solved (compatibility method)"""
        data_to_test = self.training_data_same_shape if test_on_training else self.test_data
        
        if not data_to_test:
            return {"solved": 0, "total": 0, "solve_rate": 0.0}
        
        solved = 0
        total = len(data_to_test)
        
        for puzzle in data_to_test:
            try:
                input_grid = np.array(puzzle['input'])
                target_grid = np.array(puzzle['output'])
                
                result = self.solve_puzzle(input_grid)
                
                if result.shape == target_grid.shape:
                    accuracy = np.mean(result == target_grid)
                    if accuracy > 0.95:  # 95% accuracy threshold for "solved"
                        solved += 1
            except Exception:
                continue
        
        solve_rate = solved / total if total > 0 else 0.0
        
        return {
            "solved": solved,
            "total": total,
            "solve_rate": solve_rate
        }

    def get_performance_report(self):
        """Get training performance report (compatibility method)"""
        return {
            'generations_trained': getattr(self, 'generation_count', 0),
            'best_fitness': getattr(self.best_octopus, 'fitness', 0) if self.best_octopus else 0,
            'performance_history': getattr(self, 'performance_history', []),
            'total_training_puzzles': len(self.training_data_same_shape) + len(self.training_data_diff_shape),
            'total_test_cases': len(self.test_data),
            'primitive_library_size': len(self.primitive_library),
            'generated_primitives': 0,  # Traditional approach doesn't generate primitives
            'pattern_library_size': 0   # Traditional approach doesn't have pattern library
        }

# ===== DEMONSTRATION AND TESTING =====
def demonstrate_meta_learning_training():
    """Demonstrate the meta-learning training system with comprehensive examples"""
    print("🐙 ARC Meta-Learning Training System Demo 🐙")
    print("=" * 60)
    
    # Create training data with both same-size and different-size puzzles
    training_data = create_comprehensive_training_data()
    
    print(f"Created {len(training_data)} training puzzles:")
    same_size_count = sum(1 for p in training_data if check_puzzle_type(p) == "same_size")
    diff_size_count = len(training_data) - same_size_count
    print(f"  Same-size puzzles: {same_size_count}")
    print(f"  Different-size puzzles: {diff_size_count}")
    
    # Initialize Mother Octopus with meta-learning
    mother = MotherOctopus(population_size=15)
    
    # Train using meta-learning approach
    print("\n🧠 Starting Meta-Learning Training...")
    training_results = mother.train_with_meta_learning(training_data, generations=5)
    
    # Test the trained system
    print("\n🧪 Testing Trained System...")
    test_results = test_meta_learning_system(mother, training_data)
    
    # Demonstrate puzzle solving
    print("\n🎯 Demonstrating Puzzle Solving...")
    demonstrate_puzzle_solving(mother, training_data)
    
    return training_results, test_results

def create_comprehensive_training_data() -> List[Dict]:
    """Create comprehensive training data with various puzzle types"""
    training_puzzles = []
    
    # Same-size puzzle 1: Color transformation
    same_size_1 = {
        "train": [
            {
                "input": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                "output": [[1, 2, 1], [2, 1, 2], [1, 2, 1]]
            },
            {
                "input": [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                "output": [[2, 1, 2], [1, 2, 1], [2, 1, 2]]
            }
        ],
        "test": [{"input": [[0, 1, 1], [1, 0, 0], [1, 1, 0]], "output": [[1, 2, 2], [2, 1, 1], [2, 2, 1]]}]
    }
    training_puzzles.append(same_size_1)
    
    # Same-size puzzle 2: Rotation
    same_size_2 = {
        "train": [
            {
                "input": [[1, 0, 0], [0, 0, 0], [0, 0, 2]],
                "output": [[0, 0, 1], [0, 0, 0], [2, 0, 0]]
            },
            {
                "input": [[3, 0, 0], [0, 1, 0], [0, 0, 0]],
                "output": [[0, 0, 3], [0, 1, 0], [0, 0, 0]]
            }
        ],
        "test": [{"input": [[1, 2, 0], [0, 0, 0], [0, 0, 3]], "output": [[0, 0, 1], [0, 0, 2], [3, 0, 0]]}]
    }
    training_puzzles.append(same_size_2)
    
    # Different-size puzzle 1: Pattern repetition (2x2 -> 4x4)
    diff_size_1 = {
        "train": [
            {
                "input": [[1, 2], [3, 4]],
                "output": [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]
            },
            {
                "input": [[5, 0], [0, 6]],
                "output": [[5, 0, 5, 0], [0, 6, 0, 6], [5, 0, 5, 0], [0, 6, 0, 6]]
            }
        ],
        "test": [{"input": [[7, 8], [9, 1]], "output": [[7, 8, 7, 8], [9, 1, 9, 1], [7, 8, 7, 8], [9, 1, 9, 1]]}]
    }
    training_puzzles.append(diff_size_1)
    
    # Different-size puzzle 2: Pattern extraction (4x4 -> 2x2)
    diff_size_2 = {
        "train": [
            {
                "input": [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 5, 6], [3, 4, 7, 8]],
                "output": [[5, 6], [7, 8]]
            },
            {
                "input": [[0, 0, 9, 1], [0, 0, 2, 3], [5, 5, 4, 4], [6, 6, 7, 7]],
                "output": [[9, 1], [2, 3]]
            }
        ],
        "test": [{"input": [[1, 1, 8, 9], [2, 2, 7, 6], [3, 3, 5, 4], [4, 4, 3, 2]], "output": [[8, 9], [7, 6]]}]
    }
    training_puzzles.append(diff_size_2)
    
    # Same-size puzzle 3: Pattern modification
    same_size_3 = {
        "train": [
            {
                "input": [[1, 0, 1], [0, 2, 0], [1, 0, 1]],
                "output": [[1, 2, 1], [2, 2, 2], [1, 2, 1]]
            },
            {
                "input": [[0, 1, 0], [1, 3, 1], [0, 1, 0]],
                "output": [[3, 1, 3], [1, 3, 1], [3, 1, 3]]
            }
        ],
        "test": [{"input": [[2, 0, 2], [0, 1, 0], [2, 0, 2]], "output": [[2, 1, 2], [1, 1, 1], [2, 1, 2]]}]
    }
    training_puzzles.append(same_size_3)
    
    # Different-size puzzle 3: Rule-based generation (3x3 -> 3x3 but different logic)
    diff_size_3 = {
        "train": [
            {
                "input": [[1, 0, 0], [0, 0, 0], [0, 0, 2]],
                "output": [[1, 1, 1], [1, 1, 1], [2, 2, 2]]
            },
            {
                "input": [[3, 0, 0], [0, 4, 0], [0, 0, 0]],
                "output": [[3, 3, 3], [4, 4, 4], [0, 0, 0]]
            }
        ],
        "test": [{"input": [[5, 0, 0], [0, 0, 6], [0, 0, 0]], "output": [[5, 5, 5], [6, 6, 6], [0, 0, 0]]}]
    }
    training_puzzles.append(diff_size_3)
    
    return training_puzzles

def check_puzzle_type(puzzle: Dict) -> str:
    """Check if puzzle is same-size or different-size"""
    for example in puzzle.get('train', []):
        input_shape = np.array(example['input']).shape
        output_shape = np.array(example['output']).shape
        if input_shape != output_shape:
            return "diff_size"
    return "same_size"

def test_meta_learning_system(mother: MotherOctopus, training_data: List[Dict]) -> Dict:
    """Test the meta-learning system performance"""
    test_results = {
        'same_size_accuracy': 0.0,
        'diff_size_accuracy': 0.0,
        'overall_accuracy': 0.0,
        'detailed_results': []
    }
    
    same_size_correct = 0
    same_size_total = 0
    diff_size_correct = 0
    diff_size_total = 0
    
    print("Testing trained system on puzzles...")
    
    for i, puzzle in enumerate(training_data):
        puzzle_type = check_puzzle_type(puzzle)
        
        for example in puzzle.get('train', []):
            input_grid = np.array(example['input'])
            target_grid = np.array(example['output'])
            
            # Get prediction from best octopus
            predicted = mother.solve_puzzle(input_grid)
            
            # Calculate accuracy
            if predicted.shape == target_grid.shape:
                accuracy = np.mean(predicted == target_grid)
            else:
                accuracy = 0.0
            
            # Record results
            is_correct = accuracy > 0.9
            test_results['detailed_results'].append({
                'puzzle_id': i,
                'puzzle_type': puzzle_type,
                'accuracy': accuracy,
                'correct': is_correct,
                'input_shape': input_grid.shape,
                'target_shape': target_grid.shape,
                'predicted_shape': predicted.shape
            })
            
            # Update counters
            if puzzle_type == "same_size":
                same_size_total += 1
                if is_correct:
                    same_size_correct += 1
            else:
                diff_size_total += 1
                if is_correct:
                    diff_size_correct += 1
            
            print(f"  Puzzle {i+1} ({puzzle_type}): {accuracy:.2f} accuracy {'✓' if is_correct else '✗'}")
    
    # Calculate final metrics
    test_results['same_size_accuracy'] = same_size_correct / same_size_total if same_size_total > 0 else 0.0
    test_results['diff_size_accuracy'] = diff_size_correct / diff_size_total if diff_size_total > 0 else 0.0
    test_results['overall_accuracy'] = (same_size_correct + diff_size_correct) / (same_size_total + diff_size_total)
    
    print(f"\nTest Results:")
    print(f"  Same-size accuracy: {test_results['same_size_accuracy']:.2f}")
    print(f"  Diff-size accuracy: {test_results['diff_size_accuracy']:.2f}")
    print(f"  Overall accuracy: {test_results['overall_accuracy']:.2f}")
    
    return test_results

def demonstrate_puzzle_solving(mother: MotherOctopus, training_data: List[Dict]):
    """Demonstrate puzzle solving with visualization"""
    print("Demonstrating puzzle solving...")
    
    for i, puzzle in enumerate(training_data[:3]):  # Show first 3 puzzles
        print(f"\n--- Puzzle {i+1} ---")
        
        example = puzzle['train'][0]  # Use first training example
        input_grid = np.array(example['input'])
        target_grid = np.array(example['output'])
        
        print("Input:")
        print(input_grid)
        
        print("Target:")
        print(target_grid)
        
        # Solve with different approaches
        print("Solutions:")
        
        # Best octopus solution
        if mother.best_octopus:
            solution = mother.best_octopus.execute(input_grid)
            print("Best Octopus:")
            print(solution)
            accuracy = np.mean(solution == target_grid) if solution.shape == target_grid.shape else 0.0
            print(f"Accuracy: {accuracy:.2f}")
        
        # Specialist solutions
        puzzle_type = check_puzzle_type(puzzle)
        specialists = [o for o in mother.population if getattr(o, 'specialization', None) == puzzle_type]
        
        if specialists:
            specialist_solution = specialists[0].execute(input_grid)
            print(f"{puzzle_type.title()} Specialist:")
            print(specialist_solution)
            specialist_accuracy = np.mean(specialist_solution == target_grid) if specialist_solution.shape == target_grid.shape else 0.0
            print(f"Accuracy: {specialist_accuracy:.2f}")

# Example usage function
def run_comprehensive_demo():
    """Run the complete demonstration"""
    print("🚀 Starting Comprehensive ARC Meta-Learning Demo")
    
    try:
        training_results, test_results = demonstrate_meta_learning_training()
        
        print("\n📊 Final Summary:")
        print(f"Training completed successfully!")
        print(f"Overall test accuracy: {test_results['overall_accuracy']:.2%}")
        print(f"Same-size puzzle accuracy: {test_results['same_size_accuracy']:.2%}")
        print(f"Different-size puzzle accuracy: {test_results['diff_size_accuracy']:.2%}")
        
        return training_results, test_results
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    # Run the demonstration
    run_comprehensive_demo()
    """Manages the evolution and training of the octopus population"""
    
    def __init__(self, population_size: int = 20, verbose: bool = False):
        self.population_size = population_size
        self.population = []
        self.generation = 0
        self.best_octopus = None
        self.performance_history = []
        self.verbose = verbose
        
        # Training and test data
        self.training_data = []
        self.test_data = []
        self.all_puzzle_files = []
        self.training_data_same_shape = []
        self.training_data_diff_shape = []
        
        # Initialize components
        self.primitive_library = self._initialize_primitives()
        self.adaptive_generator = AdaptivePrimitiveGenerator()
        self.cost_calculator = CostFunctionCalculator()
        self.data_loader = PuzzleDataLoader()
    
    def _initialize_primitives(self) -> List[Primitive]:
        primitives = [
             # Geometric Transformations
        TileToOutputPrimitive(), ExtractAndPlacePrimitive(), GenerateFilledGridPrimitive(),
        Rot90Primitive(), Rot180Primitive(), Rot270Primitive(),
        FlipHPrimitive(), FlipVPrimitive(), TransposePrimitive(),
        FlipDiagPrimitive(), FlipAntiDiagPrimitive(),
        # Positional Shifts
        ShiftUpPadPrimitive(), ShiftDownPadPrimitive(), ShiftLeftPrimitive(), ShiftRightPrimitive(),
        # Size Transformations
        TilePatternPrimitive(), Tile3x3Primitive(), Tile2x2Primitive(),
        CropCenterHalfPrimitive(), CropCenterThirdPrimitive(),
        # Color Operations
        MaskC1Primitive(), MaskC2Primitive(), MaskC3Primitive(),
        Replace0to1Primitive(), Replace1to2Primitive(), HoleMaskPrimitive(),
        # Flood Fill Operations
        FloodObjectPrimitive(), FillBackground0Primitive(),
        # Object Detection & Analysis
        ObjectsPrimitive(), BBoxPrimitive(),
        RotatePrimitive(),
        FlipPrimitive(),
        MirrorPrimitive(),
        TranslatePrimitive(),
        ScalePrimitive(),
        CropPrimitive(),
        ExtendPrimitive(),
        CopyPrimitive(),
        RepeatPrimitive(),
        TilePrimitive(),
        IntersectionPrimitive(),
        UnionPrimitive(),
        DifferencePrimitive(),
        ConditionalFillPrimitive(),
        FloodFillPrimitive(),
        ConnectedComponentsPrimitive(),
        SymmetryDetectionPrimitive(),
        ObjectExtractionPrimitive(),
        ColorDetectionPrimitive(),
        AddPrimitive(),
        FillEnclosedPrimitive()
        ]
        return primitives
    
    def load_puzzle_data(self, puzzle_data: Dict):
        """Load puzzle data from JSON format"""
        self.training_data_same_shape = []
        self.training_data_diff_shape = []
        for puzzle in puzzle_data.get('train', []):
            if 'input' in puzzle and 'output' in puzzle:
                in_shape = np.array(puzzle['input']).shape
                out_shape = np.array(puzzle['output']).shape
                if in_shape == out_shape:
                    self.training_data_same_shape.append(puzzle)
                else:
                    self.training_data_diff_shape.append(puzzle)

        for puzzle in puzzle_data.get('test', []):
            if 'input' in puzzle and 'output' in puzzle:
                in_shape = np.array(puzzle['input']).shape
                out_shape = np.array(puzzle['output']).shape
                if in_shape == out_shape:
                    self.test_data.append(puzzle)
                else:
                    self.test_data.append(puzzle)

    
    def load_all_puzzles_from_directory(self, directory_path: str):
        """Load all puzzle files from directory for comprehensive learning"""
        print(f"Loading all puzzles from directory: {directory_path}")
        self.all_puzzle_files = self.data_loader.load_all_puzzles_from_directory(directory_path)
        if self.all_puzzle_files:
            json_training_data = self.data_loader.combine_training_data(self.all_puzzle_files)
            for puzzle in json_training_data:
                if 'input' in puzzle and 'output' in puzzle:
                    in_shape = np.array(puzzle['input']).shape
                    out_shape = np.array(puzzle['output']).shape
                    if in_shape == out_shape:
                        self.training_data_same_shape.append(puzzle)
                    else:
                        self.training_data_diff_shape.append(puzzle)
            for puzzle_file in self.all_puzzle_files:
                test_data = puzzle_file['data'].get('test', [])
                for test_case in test_data:
                    if 'input' in test_case and 'output' in test_case:
                        in_shape = np.array(test_case['input']).shape
                        out_shape = np.array(test_case['output']).shape
                        if in_shape == out_shape:
                            test_case['source_file'] = puzzle_file['file_name']
                            self.test_data.append(test_case)
                        else:
                            test_case['source_file'] = puzzle_file['file_name']
                            self.test_data.append(test_case)
        print(f"Total training puzzles (same shape): {len(self.training_data_same_shape)}")
        print(f"Total training puzzles (diff shape): {len(self.training_data_diff_shape)}")
        print(f"Total test cases: {len(self.test_data)}")
    
    def initialize_population(self):
        """Create initial population with diverse initialization"""
        self.population = []
        for i in range(self.population_size):
            octopus = Octopus(i)
            num_tentacles = random.randint(2, 5)
            for j in range(num_tentacles):
                num_primitives = random.randint(1, 3)
                selected_primitives = random.sample(
                    self.primitive_library,
                    min(num_primitives, len(self.primitive_library))
                )
                specializations = ["transform", "detect", "edit", "logical"]
                tentacle = Tentacle(
                    name=f"tentacle_{j}",
                    primitives=selected_primitives,
                    specialization=random.choice(specializations)
                )
                octopus.add_tentacle(tentacle)
            self.population.append(octopus)
    
    def evaluate_population(self, training_data: List[Dict]):
        """Evaluate entire population on training puzzles"""
        if not training_data:
            return
        for octopus in self.population:
            total_completeness = 0.0
            puzzles_solved = 0
            octopus.completeness_scores = []
            for puzzle in training_data:
                try:
                    input_grid = np.array(puzzle['input'])
                    target_grid = np.array(puzzle['output'])
                    if input_grid.shape != target_grid.shape:
                        continue
                    octopus.train_neural_network_primitives(input_grid, target_grid)
                    output_grid = octopus.execute(input_grid)
                    completeness = octopus.calculate_completeness(output_grid, target_grid)
                    octopus.apply_cost_function_penalties(input_grid, target_grid)
                    octopus.completeness_scores.append(completeness)
                    total_completeness += completeness
                    if completeness > 0.95:
                        puzzles_solved += 1
                        state_key = octopus._generate_state_key(input_grid)
                        if octopus.last_action is not None:
                            octopus.update_q_value(state_key, octopus.last_action, 1.0)
                except Exception as e:
                    octopus.completeness_scores.append(0.0)
            avg_completeness = total_completeness / len(training_data) if training_data else 0.0
            octopus.update_fitness(puzzles_solved, len(training_data), avg_completeness)
    
    def adapt_to_novel_puzzles(self, training_data: List[Dict]):
        """Adaptive learning for novel puzzles (DISABLED for now)"""
        # To re-enable, set allow_adaptive = True
        allow_adaptive = False
        if not allow_adaptive:
            return
        for puzzle in training_data:
            try:
                input_grid = np.array(puzzle['input'])
                target_grid = np.array(puzzle['output'])
                if input_grid.shape != target_grid.shape:
                    continue
                if self.adaptive_generator.analyze_puzzle_novelty(input_grid, target_grid):
                    print(f"Novel puzzle detected from {puzzle.get('source_file', 'training.txt')}")
                    new_primitives = self.adaptive_generator.generate_new_primitives(input_grid, target_grid)
                    if new_primitives:
                        print(f"Generated {len(new_primitives)} new primitives")
                        self.primitive_library.extend(new_primitives)
                        self._create_specialized_octopus(new_primitives)
            except Exception as e:
                if self.verbose:
                    print(f"Error in adaptive learning: {e}")
    
    def _create_specialized_octopus(self, new_primitives: List[Primitive]):
        """Create a specialized octopus using new primitives"""
        if len(self.population) < self.population_size:
            octopus_id = len(self.population)
        else:
            worst_octopus = min(self.population, key=lambda x: x.fitness)
            octopus_id = worst_octopus.id
            self.population.remove(worst_octopus)
        
        specialized_octopus = Octopus(octopus_id)
        
        tentacle = Tentacle(
            name="adaptive_tentacle",
            primitives=new_primitives,
            specialization="adaptive"
        )
        specialized_octopus.add_tentacle(tentacle)
        
        if len(self.primitive_library) > len(new_primitives):
            standard_primitives = random.sample(self.primitive_library[:-len(new_primitives)], min(2, len(self.primitive_library) - len(new_primitives)))
            standard_tentacle = Tentacle(
                name="standard_tentacle",
                primitives=standard_primitives,
                specialization="edit"
            )
            specialized_octopus.add_tentacle(standard_tentacle)
        
        self.population.append(specialized_octopus)
        print(f"Created specialized octopus {octopus_id} with adaptive primitives")
    
    def selection(self) -> List[Octopus]:
        """Select elite octopuses for reproduction with better diversity"""
        if not self.population:
            return []
        
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Keep more elites but ensure diversity
        elite_size = max(2, int(0.3 * self.population_size))  # Increased from 0.1 to 0.3
        elites = self.population[:elite_size]
        
        # Add some random selections for diversity
        non_elites = self.population[elite_size:]
        if non_elites:
            random_count = min(2, len(non_elites))
            random_selections = random.sample(non_elites, random_count)
            elites.extend(random_selections)
        
        return elites
    
    def crossover(self, parent1: Octopus, parent2: Octopus) -> Octopus:
        """Create offspring through tentacle exchange"""
        child = Octopus(len(self.population))
        child.generation = self.generation + 1
        
        all_tentacles = parent1.tentacles + parent2.tentacles
        
        if not all_tentacles:
            selected_primitives = random.sample(
                self.primitive_library, 
                min(2, len(self.primitive_library))
            )
            tentacle = Tentacle(
                name="default_tentacle",
                primitives=selected_primitives,
                specialization="edit"
            )
            child.add_tentacle(tentacle)
        else:
            num_tentacles = random.randint(1, min(5, len(all_tentacles)))
            selected_tentacles = random.sample(all_tentacles, num_tentacles)
            
            for tentacle in selected_tentacles:
                new_tentacle = Tentacle(
                    tentacle.name + "_copy",
                    tentacle.primitives.copy(),
                    tentacle.specialization
                )
                new_tentacle.parameters = tentacle.parameters.copy()
                child.add_tentacle(new_tentacle)
        
        return child
    
    def mutate(self, octopus: Octopus, mutation_rate: float = 0.2):
        """Apply enhanced mutations to an octopus"""
        # More aggressive mutation for better exploration
        if random.random() < mutation_rate and len(octopus.tentacles) < 8:
            # Add new tentacle with higher chance
            num_primitives = random.randint(2, 4)  # More primitives per tentacle
            selected_primitives = random.sample(
                self.primitive_library, 
                min(num_primitives, len(self.primitive_library))
            )
            
            specializations = ["transform", "detect", "edit", "logical", "color", "size"]
            new_tentacle = Tentacle(
                name=f"mutated_tentacle_{random.randint(0, 1000)}",
                primitives=selected_primitives,
                specialization=random.choice(specializations)
            )
            octopus.add_tentacle(new_tentacle)
        
        # Remove poor performing tentacles
        if random.random() < mutation_rate * 0.5 and len(octopus.tentacles) > 2:
            # Keep at least 2 tentacles
            octopus.tentacles.pop(random.randint(0, len(octopus.tentacles) - 1))
        
        # Mutate existing tentacles more aggressively
        for tentacle in octopus.tentacles:
            if random.random() < mutation_rate * 1.5:
                # Parameter mutations
                tentacle.parameters['color'] = random.randint(0, 9)
                tentacle.parameters['angle'] = random.choice([90, 180, 270])
                tentacle.parameters['axis'] = random.choice([0, 1])
                
                # Sometimes replace primitives
                if random.random() < 0.3 and self.primitive_library:
                    new_primitive = random.choice(self.primitive_library)
                    if tentacle.primitives:
                        idx = random.randint(0, len(tentacle.primitives) - 1)
                        tentacle.primitives[idx] = new_primitive
                    else:
                        tentacle.primitives.append(new_primitive)
    
    def evolve_generation(self, training_data: List[Dict]):
        """Execute one generation of evolution"""
        print(f"=== Generation {self.generation + 1} ===")
        self.adapt_to_novel_puzzles(training_data)
        self.evaluate_population(training_data)
        if self.population:
            best_current = max(self.population, key=lambda x: x.fitness)
            if self.best_octopus is None or best_current.fitness > self.best_octopus.fitness:
                self.best_octopus = copy.deepcopy(best_current)
        elite = self.selection()
        if not elite:
            return
        new_population = elite.copy()
        while len(new_population) < self.population_size:
            parent1 = random.choice(elite)
            parent2 = random.choice(elite)
            child = self.crossover(parent1, parent2)
            self.mutate(child, mutation_rate=0.3)  # Increased mutation rate
            new_population.append(child)
        self.population = new_population
        self.generation += 1
        if self.population:
            avg_fitness = np.mean([oct.fitness for oct in self.population])
            max_fitness = max([oct.fitness for oct in self.population])
            self.performance_history.append({
                'generation': self.generation,
                'avg_fitness': avg_fitness,
                'max_fitness': max_fitness
            })
            if self.generation % 5 == 0 or self.generation == 1:
                print(f"  [Progress] Generation {self.generation}: Avg={avg_fitness:.3f}, Best={max_fitness:.3f}, Max={max_fitness:.3f}")
                print(f"  [Progress] Primitive library size: {len(self.primitive_library)}")
            else:
                print(f"  Generation {self.generation}: Avg={avg_fitness:.3f}, Best={max_fitness:.3f}")
    
    def train(self, generations: int = 50, explore_rate: float = 0.1):
        try:
            """Train the system"""
            if not self.training_data_same_shape and not self.training_data_diff_shape:
                print("No training data available. Please load puzzle data first.")
                return
            print("Initializing population...")
            self.initialize_population()
            print(f"Training for {generations} generations on {len(self.training_data_same_shape)} same-shape and {len(self.training_data_diff_shape)} diff-shape puzzles...")
            for gen in range(generations):
                # With probability explore_rate, use a diff-shape puzzle
                if self.training_data_diff_shape and random.random() < explore_rate:
                    print(f"[Exploration] Training on different-shape puzzles (gen {gen+1})")
                    self.evolve_generation(self.training_data_diff_shape)
                else:
                    self.evolve_generation(self.training_data_same_shape)
                if (gen + 1) % 5 == 0 or gen == generations - 1:
                    print(f"\n[Summary] Generation {gen + 1} completed")
                    if self.best_octopus:
                        solve_stats = self.count_puzzles_solved(test_on_training=True)
                        print(f"  [Summary] Current best octopus solves {solve_stats['solved']}/{solve_stats['total']} training puzzles ({solve_stats['solve_rate']:.1%})")
        except Exception as e:
            print(f"Error in generation {gen}: {e}")
            import traceback
            traceback.print_exc()
            return
    
    def solve_puzzle(self, input_grid: np.ndarray) -> np.ndarray:
        """Use the best octopus to solve a puzzle"""
        if self.best_octopus is not None:
            return self.best_octopus.execute(input_grid)
        else:
            return input_grid
    
    def count_puzzles_solved(self, test_on_training: bool = False) -> Dict:
        """Count how many puzzles the best octopus actually solves"""
        if self.best_octopus is None:
            return {"solved": 0, "total": 0, "solve_rate": 0.0}
        
        # Choose which dataset to test on
        if test_on_training:
            # For training data, combine both same and different size puzzles
            data_to_test = self.training_data_same_shape + self.training_data_diff_shape
            test_type = "TRAINING"
        else:
            data_to_test = self.test_data
            test_type = "TEST"
        
        if not data_to_test:
            print(f"\n=== NO {test_type} DATA AVAILABLE ===")
            return {"solved": 0, "total": 0, "solve_rate": 0.0}
        
        puzzles_solved = 0
        total_puzzles = 0
        
        print(f"\n=== TESTING PUZZLE SOLVING ABILITY ===")
        print(f"Testing on {test_type} data ({len(data_to_test)} puzzles)...")
        
        for i, puzzle in enumerate(data_to_test):
            try:
                input_grid = np.array(puzzle['input'])
                if 'output' in puzzle:  # Only test if we have expected output
                    target_grid = np.array(puzzle['output'])
                    predicted_grid = self.best_octopus.execute(input_grid)
                    
                    # Check if puzzle is solved (exact match)
                    if np.array_equal(predicted_grid, target_grid):
                        puzzles_solved += 1
                        print(f"✓ Puzzle {i+1}: SOLVED")
                    else:
                        # Calculate how close we got
                        accuracy = np.sum(predicted_grid == target_grid) / target_grid.size
                        print(f"✗ Puzzle {i+1}: {accuracy:.1%} correct")
                    
                    total_puzzles += 1
                else:
                    # For test data without expected output, just run the solver
                    predicted_grid = self.best_octopus.execute(input_grid)
                    print(f"? Puzzle {i+1}: Generated solution (no expected output to compare)")
                    total_puzzles += 1
                    
            except Exception as e:
                print(f"✗ Puzzle {i+1}: ERROR - {e}")
                total_puzzles += 1
        
        solve_rate = puzzles_solved / total_puzzles if total_puzzles > 0 else 0.0
        
        result = {
            "solved": puzzles_solved,
            "total": total_puzzles,
            "solve_rate": solve_rate
        }
        
        print(f"\n=== PUZZLE SOLVING RESULTS ===")
        print(f"Puzzles solved: {puzzles_solved}/{total_puzzles}")
        print(f"Success rate: {solve_rate:.1%}")
        
        return result
    
    def get_performance_report(self) -> Dict:
        """Get training performance report"""
        return {
            'generations_trained': self.generation,
            'best_fitness': self.best_octopus.fitness if self.best_octopus else 0,
            'performance_history': self.performance_history,
            'total_training_puzzles': len(self.training_data_same_shape) + len(self.training_data_diff_shape),
            'total_test_cases': len(self.test_data),
            'primitive_library_size': len(self.primitive_library),
            'generated_primitives': len(self.adaptive_generator.generated_primitives),
            'pattern_library_size': len(self.adaptive_generator.pattern_library)
        }

# ===== MAIN FUNCTION =====
def main():
    """Enhanced main function to demonstrate both training approaches"""
    print("=== Enhanced Octopus ARC Solver ===")
    
    # First run the meta-learning demo
    print("\n🚀 Running Meta-Learning Demo First...")
    try:
        run_comprehensive_demo()
    except Exception as e:
        print(f"Meta-learning demo error: {e}")
    
    print("\n" + "="*60)
    print("Now running traditional training approach...")
    print("="*60)
    
    mother_octopus = MotherOctopus(population_size=10, verbose=False)
    
    training_dir = os.path.join("data", "training")
    test_dir = os.path.join("data", "evaluation")

    # Load training data
    print(f"Loading training data from: {training_dir}")
    if os.path.exists(training_dir):
        mother_octopus.load_all_puzzles_from_directory(training_dir)
        print(f"Training puzzles loaded: {len(mother_octopus.training_data_same_shape)} same-shape, {len(mother_octopus.training_data_diff_shape)} diff-shape")
    else:
        print(f"Training directory {training_dir} does not exist! Using fallback data.")
        fallback_data = {
            "train": mother_octopus.data_loader._get_fallback_training_data(),
            "test": []
        }
        mother_octopus.load_puzzle_data(fallback_data)

    # Load test data (evaluation set)
    print(f"Loading test data from: {test_dir}")
    if os.path.exists(test_dir):
        # Only load test data from evaluation folder, don't add to training
        test_files = mother_octopus.data_loader.load_all_puzzles_from_directory(test_dir)
        test_cases = []
        for puzzle_file in test_files:
            file_name = puzzle_file['file_name']
            test_data = puzzle_file['data'].get('test', [])
            for test_case in test_data:
                test_case['source_file'] = file_name
                test_cases.append(test_case)
            # Some evaluation sets may have 'input' only (no 'test' key), so also check for 'input' at top level
            if 'input' in puzzle_file['data']:
                test_cases.append({
                    'input': puzzle_file['data']['input'],
                    'source_file': file_name
                })
        mother_octopus.test_data = test_cases
        print(f"Test cases loaded: {len(mother_octopus.test_data)}")
    else:
        print(f"Test directory {test_dir} does not exist! Using fallback test data.")
        fallback_test_data = mother_octopus._create_fallback_test_data()
        for test_case in fallback_test_data:
            test_case['source_file'] = 'fallback_generator'
        mother_octopus.test_data = fallback_test_data
        print(f"Test cases loaded: {len(mother_octopus.test_data)} (fallback)")

    print(f"\n=== DATA SUMMARY ===")
    print(f"Training puzzles loaded: {len(mother_octopus.training_data_same_shape)} same-shape, {len(mother_octopus.training_data_diff_shape)} diff-shape")
    print(f"Test cases loaded: {len(mother_octopus.test_data)}")
    
    print("\n=== STARTING TRAINING ===")
    mother_octopus.train(generations=20)
    
    print("\n=== FINAL TRAINING RESULTS ===")
    if mother_octopus.training_data_same_shape or mother_octopus.training_data_diff_shape:
        final_training_stats = mother_octopus.count_puzzles_solved(test_on_training=True)
        print(f"Training Results: {final_training_stats['solved']}/{final_training_stats['total']} solved ({final_training_stats['solve_rate']:.1%})")
    else:
        print("No training data was available for testing.")
    
    print("\n=== TESTING ON TEST DATA ===")
    if mother_octopus.test_data:
        test_stats = mother_octopus.count_puzzles_solved(test_on_training=False)
        print(f"Test Results: {test_stats['solved']}/{test_stats['total']} solved ({test_stats['solve_rate']:.1%})")
    else:
        print("No test data available")
    
    # Performance summary
    report = mother_octopus.get_performance_report()
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Generations trained: {report['generations_trained']}")
    print(f"Best fitness achieved: {report['best_fitness']:.3f}")
    if mother_octopus.training_data_same_shape or mother_octopus.training_data_diff_shape:
        print(f"Training puzzles solved: {final_training_stats['solved']}/{final_training_stats['total']} ({final_training_stats['solve_rate']:.1%})")
    if mother_octopus.test_data:
        print(f"Test puzzles solved: {test_stats['solved']}/{test_stats['total']} ({test_stats['solve_rate']:.1%})")
    print(f"Primitive library size: {report['primitive_library_size']}")
    print(f"Generated adaptive primitives: {report['generated_primitives']}")

if __name__ == "__main__":
    main()