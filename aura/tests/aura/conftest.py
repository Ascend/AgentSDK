import sys
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


@pytest.fixture(autouse=False, scope="function")
def mock_requests():
    """Mock requests module for testing"""
    mock_requests = MagicMock()
    
    def mock_get(url, **kwargs):
        return MagicMock(status_code=200, json=lambda: {})
    
    def mock_post(url, **kwargs):
        return MagicMock(status_code=200, json=lambda: {})
    
    mock_requests.get = mock_get
    mock_requests.post = mock_post

    with patch.dict(sys.modules, {"requests": mock_requests}):
        yield


@pytest.fixture(autouse=False, scope="function")
def mock_aiohttp():
    """Mock aiohttp module for testing"""
    mock_aiohttp = MagicMock()
    
    class MockClientSession:
        def __init__(self):
            pass
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, *args):
            pass
        
        async def get(self, url):
            return MagicMock()
        
        async def post(self, url, json=None):
            return MagicMock()
    
    mock_aiohttp.ClientSession = MockClientSession

    with patch.dict(sys.modules, {"aiohttp": mock_aiohttp}):
        yield


@pytest.fixture(autouse=False, scope="function")
def mock_pydantic():
    """Mock pydantic module for testing"""
    mock_pydantic = MagicMock()
    
    class MockBaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def dict(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
        def json(self):
            import json
            return json.dumps(self.dict())
    
    mock_pydantic.BaseModel = MockBaseModel
    mock_pydantic.Field = lambda default=None, **kwargs: default

    with patch.dict(sys.modules, {"pydantic": mock_pydantic}):
        yield


@pytest.fixture(autouse=False, scope="function")
def mock_torch():
    """Mock torch module for testing"""
    mock_torch = MagicMock()
    mock_torch.cuda = MagicMock()
    mock_torch.cuda.is_available = MagicMock(return_value=False)
    mock_torch.distributed = MagicMock()

    with patch.dict(
        sys.modules,
        {
            "torch": mock_torch,
            "torch.distributed": mock_torch.distributed,
        },
    ):
        yield


@pytest.fixture(autouse=False, scope="function")
def mock_fastapi():
    """Mock fastapi module for testing"""
    mock_fastapi = MagicMock()
    
    class MockHTTPException(Exception):
        def __init__(self, status_code, detail=""):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)
    
    class MockRequest:
        def __init__(self):
            self.json = AsyncMock(return_value={})
    
    class MockFastAPI:
        def __init__(self):
            self.routes = []
            self.include_router = MagicMock()
        
        def add_api_route(self, path, endpoint, methods=None):
            self.routes.append(MagicMock(path=path, methods=methods or set()))
        
        def get(self, path):
            def decorator(func):
                self.routes.append(MagicMock(path=path, methods={"GET"}))
                return func
            return decorator
        
        def post(self, path):
            def decorator(func):
                self.routes.append(MagicMock(path=path, methods={"POST"}))
                return func
            return decorator
    
    class MockAPIRouter:
        def __init__(self, prefix=""):
            self.prefix = prefix
            self.routes = []
        
        def add_api_route(self, path, endpoint, methods=None):
            self.routes.append(MagicMock(path=path, methods=methods or set()))
        
        def post(self, path):
            def decorator(func):
                self.routes.append(MagicMock(path=path, methods={"POST"}))
                return func
            return decorator
    
    mock_fastapi.Request = MockRequest
    mock_fastapi.FastAPI = MockFastAPI
    mock_fastapi.APIRouter = MockAPIRouter
    mock_fastapi.HTTPException = MockHTTPException

    with patch.dict(
        sys.modules,
        {
            "fastapi": mock_fastapi,
            "fastapi.testclient": MagicMock(),
        },
    ):
        yield


@pytest.fixture(autouse=False, scope="function")
def mock_sse_starlette():
    """Mock sse_starlette module for testing"""
    mock_sse = MagicMock()
    
    class MockEventSourceResponse:
        def __init__(self, content):
            self.content = content
    
    mock_sse.EventSourceResponse = MockEventSourceResponse

    with patch.dict(sys.modules, {"sse_starlette": mock_sse}):
        yield