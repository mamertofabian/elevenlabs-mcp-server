import pytest

from elevenlabs_mcp.server import ElevenLabsServer


def test_parse_script_valid_input():
    server = ElevenLabsServer(enable_revival=False)
    valid_json = """
    {
        "script": [
            {
                "text": "Hello world",
                "voice_id": "voice1",
                "actor": "narrator"
            }
        ]
    }
    """

    script_parts, _debug_info = server.parse_script(valid_json)

    assert len(script_parts) == 1
    assert script_parts[0]["text"] == "Hello world"
    assert script_parts[0]["voice_id"] == "voice1"
    assert script_parts[0]["actor"] == "narrator"


# Valid JSON with script array containing multiple parts with text, voice_id, and actor
def test_parse_valid_script_json():
    server = ElevenLabsServer(enable_revival=False)
    script_json = """
    {
        "script": [
            {"text": "Hello", "voice_id": "voice1", "actor": "Bob"},
            {"text": "World", "voice_id": "voice2", "actor": "Alice"}
        ]
    }"""

    script_parts, debug_info = server.parse_script(script_json)

    assert len(script_parts) == 2
    assert script_parts[0] == {"text": "Hello", "voice_id": "voice1", "actor": "Bob"}
    assert script_parts[1] == {"text": "World", "voice_id": "voice2", "actor": "Alice"}
    assert len(debug_info) > 0


def test_parse_script_invalid_json():
    server = ElevenLabsServer(enable_revival=False)
    invalid_json = "{ invalid json }"

    with pytest.raises(Exception) as exc_info:
        server.parse_script(invalid_json)

    assert "Invalid JSON format" in str(exc_info.value)


def test_parse_script_only_text():
    server = ElevenLabsServer(enable_revival=False)
    json_with_only_text = """
    {
        "script": [
            {"text": "Hello world"}
        ]
    }
    """

    script_parts, _debug_info = server.parse_script(json_with_only_text)

    assert len(script_parts) == 1
    assert script_parts[0]["text"] == "Hello world"
    assert script_parts[0]["voice_id"] is None
    assert script_parts[0]["actor"] is None


def test_parse_script_missing_voice():
    server = ElevenLabsServer(enable_revival=False)
    json_missing_voice = """
    {
        "script": [
            {
                "text": "Hello world",
                "actor": "narrator"
            }
        ]
    }
    """

    script_parts, _debug_info = server.parse_script(json_missing_voice)

    assert len(script_parts) == 1
    assert script_parts[0]["text"] == "Hello world"
    assert script_parts[0]["voice_id"] is None
    assert script_parts[0]["actor"] == "narrator"


def test_parse_script_missing_actor():
    server = ElevenLabsServer(enable_revival=False)
    json_missing_actor = """
    {
        "script": [
            {
                "text": "Hello world",
                "voice_id": "voice1"
            }
        ]
    }
    """

    script_parts, _debug_info = server.parse_script(json_missing_actor)

    assert len(script_parts) == 1
    assert script_parts[0]["text"] == "Hello world"
    assert script_parts[0]["voice_id"] == "voice1"
    assert script_parts[0]["actor"] is None


def test_parse_script_empty_array():
    server = ElevenLabsServer(enable_revival=False)
    empty_script = """
    {
        "script": []
    }
    """

    script_parts, _debug_info = server.parse_script(empty_script)

    assert len(script_parts) == 0


def test_parse_script_empty_object():
    server = ElevenLabsServer(enable_revival=False)
    empty_object = """
    {
        "script": [
            {}
        ]
    }
    """

    with pytest.raises(Exception) as exc_info:
        server.parse_script(empty_object)

    assert "Missing required field 'text'" in str(exc_info.value)


def test_parse_plain_text():
    server = ElevenLabsServer(enable_revival=False)
    plain_text = "Hello, this is plain text"

    script_parts, _debug_info = server.parse_script(plain_text)

    assert len(script_parts) == 1
    assert script_parts[0]["text"] == "Hello, this is plain text"
    assert script_parts[0]["voice_id"] is None
    assert script_parts[0]["actor"] is None


def test_parse_mixed_input():
    server = ElevenLabsServer(enable_revival=False)
    # Valid JSON with some parts having only text
    mixed_json = """
    {
        "script": [
            {"text": "Part 1"},
            {"text": "Part 2", "voice_id": "voice1"},
            {"text": "Part 3", "actor": "Bob"},
            {"text": "Part 4", "voice_id": "voice2", "actor": "Alice"}
        ]
    }
    """

    script_parts, _debug_info = server.parse_script(mixed_json)

    assert len(script_parts) == 4
    assert script_parts[0] == {"text": "Part 1", "voice_id": None, "actor": None}
    assert script_parts[1] == {"text": "Part 2", "voice_id": "voice1", "actor": None}
    assert script_parts[2] == {"text": "Part 3", "voice_id": None, "actor": "Bob"}
    assert script_parts[3] == {"text": "Part 4", "voice_id": "voice2", "actor": "Alice"}
