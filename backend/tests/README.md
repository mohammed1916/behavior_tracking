# Backend Tests

Tests are organized into the following categories:

## 📂 Folder Structure

### `/streaming`
Tests for SSE (Server-Sent Events) and streaming functionality:
- `test_sse_quick.py` - Quick SSE streaming tests
- `test_sse_segment_events.py` - SSE segment event tests
- `test_direct_stream.py` - Direct streaming tests
- `test_e2e_vlm_stream.py` - End-to-end VLM streaming tests

### `/server`
API endpoint and server functionality tests:
- `test_server.py` - Server API tests

### `/aggregation`
Caption aggregation and timing-related tests:
- `test_aggregation.py` - Caption aggregation tests
- `test_timing.py` - Timing and temporal window tests

### `/models`
VLM (Vision-Language Model) and model loading tests:
- `test_vlm_local_models.py` - Local VLM model tests

### `/integration`
Integration and end-to-end workflow tests:
- `test_small.py` - Small integration test suite

## Running Tests

Run all tests:
```bash
pytest backend/tests/
```

Run specific category:
```bash
pytest backend/tests/streaming/
pytest backend/tests/aggregation/
pytest backend/tests/server/
pytest backend/tests/models/
pytest backend/tests/integration/
```

Run specific test file:
```bash
pytest backend/tests/streaming/test_sse_quick.py
```
