"""
Test that all nodes are properly registered and importable
"""
import pytest


@pytest.mark.unit
def test_import_torch():
    """Test that PyTorch is available"""
    import torch
    assert torch.__version__ is not None
    print(f"PyTorch version: {torch.__version__}")


@pytest.mark.unit
def test_import_nodes():
    """Test that node mappings can be imported"""
    from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    assert isinstance(NODE_CLASS_MAPPINGS, dict)
    assert isinstance(NODE_DISPLAY_NAME_MAPPINGS, dict)
    assert len(NODE_CLASS_MAPPINGS) > 0
    print(f"Loaded {len(NODE_CLASS_MAPPINGS)} nodes")


@pytest.mark.unit
def test_required_nodes_registered():
    """Test that all required nodes are registered"""
    from nodes import NODE_CLASS_MAPPINGS

    required_nodes = [
        'LoadSAM3Model',
        'SAM3Segmentation',
        'SAM3VideoModelLoader',
        'SAM3InitVideoSession',
        'SAM3InitVideoSessionAdvanced',
        'SAM3AddVideoPrompt',
        'SAM3PropagateVideo',
        'SAM3VideoOutput',
        'SAM3CloseVideoSession',
    ]

    for node in required_nodes:
        assert node in NODE_CLASS_MAPPINGS, f'Missing node: {node}'

    print(f'✓ All {len(required_nodes)} required nodes registered successfully')


@pytest.mark.unit
def test_node_display_names():
    """Test that all nodes have display names"""
    from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    for node_name in NODE_CLASS_MAPPINGS.keys():
        assert node_name in NODE_DISPLAY_NAME_MAPPINGS, f'Missing display name for: {node_name}'

    print('✓ All nodes have display names')


@pytest.mark.unit
def test_simplified_vs_advanced_session_params():
    """Test that simplified session node has fewer parameters than advanced"""
    from nodes import NODE_CLASS_MAPPINGS

    simplified = NODE_CLASS_MAPPINGS['SAM3InitVideoSession']
    advanced = NODE_CLASS_MAPPINGS['SAM3InitVideoSessionAdvanced']

    simplified_inputs = simplified.INPUT_TYPES()
    advanced_inputs = advanced.INPUT_TYPES()

    simplified_optional = len(simplified_inputs.get('optional', {}))
    advanced_optional = len(advanced_inputs.get('optional', {}))

    assert simplified_optional < advanced_optional, \
        f'Simplified node should have fewer parameters than advanced (got {simplified_optional} vs {advanced_optional})'

    print(f'✓ Simplified node has {simplified_optional} optional parameters')
    print(f'✓ Advanced node has {advanced_optional} optional parameters')


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
        'sam3_lib/model_builder.py',
    ]

    for file in required_files:
        path = Path(file)
        assert path.exists(), f'Missing required file: {file}'

    print('✓ Package structure is valid')


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

    print(f'✓ Nodes organized in {len(category_counts)} categories:')
    for cat, count in sorted(category_counts.items()):
        print(f'  - {cat}: {count} nodes')
