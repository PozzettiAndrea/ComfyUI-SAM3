"""
Pytest configuration and fixtures for ComfyUI-SAM3 tests
"""
import sys
import os
from pathlib import Path
import pytest
from unittest.mock import MagicMock, Mock


# Add the custom node directory to Python path so we can import modules
custom_nodes_dir = Path(__file__).parent.parent
sys.path.insert(0, str(custom_nodes_dir))

# Mock ComfyUI modules at module level BEFORE pytest starts
# This prevents import errors when pytest tries to load __init__.py files

# Mock folder_paths module (ComfyUI path management)
test_models_dir = os.environ.get("TEST_MODELS_DIR", str(Path.home() / ".cache" / "test_models"))
test_base_path = os.environ.get("TEST_BASE_PATH", str(Path.home() / ".cache" / "comfy_test"))
mock_folder_paths = type("folder_paths", (), {})()
mock_folder_paths.models_dir = test_models_dir
mock_folder_paths.base_path = test_base_path
mock_folder_paths.get_folder_paths = lambda x: [test_models_dir]
mock_folder_paths.get_temp_directory = lambda: "/tmp/comfy_temp"
mock_folder_paths.get_output_directory = lambda: "/tmp/comfy_output"
mock_folder_paths.get_input_directory = lambda: "/tmp/comfy_input"
mock_folder_paths.add_model_folder_path = lambda name, path: None
sys.modules["folder_paths"] = mock_folder_paths

# Mock comfy modules
mock_comfy = type("comfy", (), {})()
mock_comfy_utils = type("utils", (), {})()
mock_comfy_utils.load_torch_file = lambda x: {}
mock_comfy_utils.ProgressBar = MagicMock()
mock_comfy.utils = mock_comfy_utils

# Mock model_management
mock_comfy_mm = type("model_management", (), {})()
mock_comfy_mm.get_torch_device = lambda: "cpu"
mock_comfy_mm.soft_empty_cache = lambda: None
mock_comfy_mm.load_models_gpu = lambda x: None
mock_comfy_mm.unet_offload_device = lambda: "cpu"
mock_comfy_mm.is_device_mps = lambda x: False
mock_comfy_mm.get_autocast_device = lambda x: "cpu"
mock_comfy_mm.module_size = lambda x: 1000000  # 1MB mock size
mock_comfy.model_management = mock_comfy_mm

# Mock model_patcher with ModelPatcher base class
class MockModelPatcher:
    """Mock ModelPatcher for testing"""
    def __init__(self, model=None, load_device="cpu", offload_device="cpu", size=0, weight_inplace_update=False):
        self.model = model
        self.load_device = load_device
        self.offload_device = offload_device
        self.size = size
        self.weight_inplace_update = weight_inplace_update
        self.patches = {}
        self.object_patches = {}
        self.model_options = {"transformer_options": {}}

    def model_size(self):
        return self.size

    def cleanup(self):
        pass

mock_comfy_model_patcher = type("model_patcher", (), {})()
mock_comfy_model_patcher.ModelPatcher = MockModelPatcher
mock_comfy.model_patcher = mock_comfy_model_patcher

sys.modules["comfy"] = mock_comfy
sys.modules["comfy.utils"] = mock_comfy_utils
sys.modules["comfy.model_management"] = mock_comfy_mm
sys.modules["comfy.model_patcher"] = mock_comfy_model_patcher

# Mock server module (ComfyUI server)
mock_prompt_server_instance = MagicMock()
mock_prompt_server_instance.app = MagicMock()
mock_prompt_server_instance.app.add_routes = MagicMock()

mock_prompt_server = type("PromptServer", (), {})()
mock_prompt_server.instance = mock_prompt_server_instance

mock_server = type("server", (), {})()
mock_server.PromptServer = type("PromptServer", (), {"instance": mock_prompt_server_instance})()

sys.modules["server"] = mock_server

# Mock aiohttp.web
mock_aiohttp_web = type("web", (), {})()
mock_aiohttp_web.static = lambda *args: None

mock_aiohttp = type("aiohttp", (), {})()
mock_aiohttp.web = mock_aiohttp_web

sys.modules["aiohttp"] = mock_aiohttp
sys.modules["aiohttp.web"] = mock_aiohttp_web


def pytest_ignore_collect(collection_path, config):
    """Ignore __init__.py files during collection"""
    if collection_path.name == "__init__.py":
        return True
    return False


@pytest.fixture(scope="session", autouse=True)
def setup_test_directories():
    """Ensure test directories exist"""
    test_models_dir = os.environ.get("TEST_MODELS_DIR", str(Path.home() / ".cache" / "test_models"))
    temp_dirs = [
        test_models_dir,
        "/tmp/comfy_temp",
        "/tmp/comfy_output",
        "/tmp/comfy_input"
    ]

    for dir_path in temp_dirs:
        os.makedirs(dir_path, exist_ok=True)

    yield


@pytest.fixture(scope="session", autouse=True)
def setup_mock_comfy():
    """Set up mock ComfyUI modules for testing - runs once per session"""
    return True


@pytest.fixture
def mock_comfy_environment():
    """Provide access to mocked ComfyUI environment"""
    return sys.modules["folder_paths"]


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, no model loading)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests with mocked models"
    )
