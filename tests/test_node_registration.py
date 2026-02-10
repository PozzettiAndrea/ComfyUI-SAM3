"""
Test that all nodes are properly registered and importable
"""
import pytest
import sys

# Import directly from nodes submodule to bypass __init__.py
# This is necessary because __init__.py skips imports during pytest
sys.path.insert(0, '.')


@pytest.mark.unit
def test_import_torch():
    """Test that PyTorch is available"""
    import torch
    assert torch.__version__ is not None
    print(f"PyTorch version: {torch.__version__}")


@pytest.mark.unit
def test_import_nodes():
    """Test that node mappings can be imported"""
    # Import directly from nodes module, not from __init__.py
    from nodes.load_model import NODE_CLASS_MAPPINGS as LOAD_MAPPINGS
    from nodes.segmentation import NODE_CLASS_MAPPINGS as SEG_MAPPINGS
    from nodes.sam3_video_nodes import NODE_CLASS_MAPPINGS as VIDEO_MAPPINGS

    # Combine all mappings
    all_mappings = {}
    all_mappings.update(LOAD_MAPPINGS)
    all_mappings.update(SEG_MAPPINGS)
    all_mappings.update(VIDEO_MAPPINGS)

    assert isinstance(all_mappings, dict)
    assert len(all_mappings) > 0
    print(f"Loaded {len(all_mappings)} nodes")


@pytest.mark.unit
def test_required_nodes_registered():
    """Test that all required nodes are registered"""
    # Import directly from nodes module
    from nodes.load_model import NODE_CLASS_MAPPINGS as LOAD_MAPPINGS
    from nodes.segmentation import NODE_CLASS_MAPPINGS as SEG_MAPPINGS
    from nodes.sam3_video_nodes import NODE_CLASS_MAPPINGS as VIDEO_MAPPINGS

    # Combine all mappings
    all_mappings = {}
    all_mappings.update(LOAD_MAPPINGS)
    all_mappings.update(SEG_MAPPINGS)
    all_mappings.update(VIDEO_MAPPINGS)

    required_nodes = [
        'LoadSAM3Model',
        'SAM3Segmentation',
        'SAM3Grounding',
        'SAM3VideoSegmentation',
        'SAM3Propagate',
        'SAM3VideoOutput',
    ]

    for node in required_nodes:
        assert node in all_mappings, f'Missing node: {node}'

    print(f'[OK] All {len(required_nodes)} required nodes registered successfully')


@pytest.mark.unit
def test_node_display_names():
    """Test that all nodes have display names"""
    from nodes.load_model import NODE_CLASS_MAPPINGS as LOAD_MAPPINGS
    from nodes.load_model import NODE_DISPLAY_NAME_MAPPINGS as LOAD_DISPLAY
    from nodes.segmentation import NODE_CLASS_MAPPINGS as SEG_MAPPINGS
    from nodes.segmentation import NODE_DISPLAY_NAME_MAPPINGS as SEG_DISPLAY
    from nodes.sam3_video_nodes import NODE_CLASS_MAPPINGS as VIDEO_MAPPINGS
    from nodes.sam3_video_nodes import NODE_DISPLAY_NAME_MAPPINGS as VIDEO_DISPLAY

    # Combine all mappings
    all_mappings = {}
    all_mappings.update(LOAD_MAPPINGS)
    all_mappings.update(SEG_MAPPINGS)
    all_mappings.update(VIDEO_MAPPINGS)

    all_display = {}
    all_display.update(LOAD_DISPLAY)
    all_display.update(SEG_DISPLAY)
    all_display.update(VIDEO_DISPLAY)

    for node_name in all_mappings.keys():
        assert node_name in all_display, f'Missing display name for: {node_name}'

    print('[OK] All nodes have display names')


@pytest.mark.unit
def test_video_segmentation_node():
    """Test that SAM3VideoSegmentation node has expected parameters"""
    from nodes.sam3_video_nodes import NODE_CLASS_MAPPINGS

    video_seg = NODE_CLASS_MAPPINGS['SAM3VideoSegmentation']
    inputs = video_seg.INPUT_TYPES()

    # Check required inputs exist
    required = inputs.get('required', {})
    assert 'video_frames' in required, 'Missing video_frames input'
    assert 'prompt_mode' in required, 'Missing prompt_mode input'

    print(f'[OK] SAM3VideoSegmentation has {len(required)} required parameters')
    print(f'[OK] SAM3VideoSegmentation has {len(inputs.get("optional", {}))} optional parameters')


@pytest.mark.unit
def test_package_structure():
    """Test that required package files exist"""
    import os
    from pathlib import Path

    required_files = [
        '__init__.py',
        'nodes/__init__.py',
        'nodes/load_model.py',
        'nodes/segmentation.py',
        'nodes/sam3_video_nodes.py',
        'nodes/sam3_lib/model_builder.py',
    ]

    for file in required_files:
        path = Path(file)
        assert path.exists(), f'Missing required file: {file}'

    print('[OK] Package structure is valid')


@pytest.mark.unit
def test_node_categories():
    """Test that nodes are properly categorized"""
    from nodes import NODE_CLASS_MAPPINGS

    category_counts = {}

    for node_name, node_class in NODE_CLASS_MAPPINGS.items():
        category = getattr(node_class, 'CATEGORY', 'unknown')
        category_counts[category] = category_counts.get(category, 0) + 1

    # Verify we have SAM3 categories
    assert 'SAM3' in category_counts or any('SAM3' in c for c in category_counts.keys()), \
        'No SAM3 category found'

    print(f'[OK] Nodes organized in {len(category_counts)} categories:')
    for cat, count in sorted(category_counts.items()):
        print(f'  - {cat}: {count} nodes')
