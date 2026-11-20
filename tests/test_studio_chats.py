"""Saved conversations: titles, ordering, validation and safe identifiers."""

from __future__ import annotations

import json

import pytest

from mariana.studio.chats import ChatStore
from mariana.studio.errors import BadRequestError, NotFoundError, TooLargeError


def test_conversations_are_created_updated_listed_and_deleted(tmp_path):
    store = ChatStore(tmp_path)
    first = store.save({
        "model": "trained-model/a.pt", "model_name": "a.pt", "system_prompt": "Be brief.",
        "settings": {"temperature": 0.2},
        "messages": [
            {"role": "system", "content": "ignored for the title"},
            {"role": "user", "content": "  What is the tallest mountain on Earth, measured from sea level?\nDetails"},
            {"role": "assistant", "content": "Everest.", "kind": "DRV-LM", "usage": {"completion_tokens": 2},
             "secret": "dropped"},
            {"role": "tool", "content": "dropped"}, {"role": "user", "content": 5}, "junk",
        ],
    })
    assert first["id"].startswith("chat-") and first["title"].endswith("…") and len(first["title"]) == 61
    assert [message["role"] for message in first["messages"]] == ["system", "user", "assistant"]
    assert first["messages"][2] == {"role": "assistant", "content": "Everest.", "kind": "DRV-LM",
                                    "usage": {"completion_tokens": 2}}
    assert (tmp_path / "agent-workspace" / "chats" / f"{first['id']}.json").is_file()

    second = store.save({"messages": [{"role": "user", "content": "Second"}]})
    assert [chat["id"] for chat in store.list()] == [second["id"], first["id"]]
    store.save({"pinned": True}, first["id"])
    listed = store.list()
    assert [chat["id"] for chat in listed] == [first["id"], second["id"]]  # pinned first
    assert listed[0]["messages"] == 3 and listed[0]["preview"] == "Everest." and listed[0]["pinned"] is True

    renamed = store.save({"title": "Mountains"}, first["id"])
    assert renamed["title"] == "Mountains" and len(renamed["messages"]) == 3  # other fields are kept
    grown = store.save({"messages": [{"role": "user", "content": "A different opening"}]}, first["id"])
    assert grown["title"] == "Mountains"  # a chosen title is never replaced by a derived one
    assert store.save({"messages": [{"role": "user", "content": "Edited"}]}, second["id"])["title"] == "Edited"
    assert store.get(first["id"])["system_prompt"] == "Be brief."
    assert store.save({"messages": []})["title"] == "New chat"

    store.delete(first["id"])
    with pytest.raises(NotFoundError):
        store.get(first["id"])
    with pytest.raises(NotFoundError):
        store.delete(first["id"])
    assert store.clear() == 2 and store.list() == []


def test_deleting_a_conversation_removes_its_tool_workspace(tmp_path):
    store = ChatStore(tmp_path)
    kept = store.save({"messages": [{"role": "user", "content": "Keep this"}]})
    removed = store.save({"messages": [{"role": "user", "content": "Delete this"}]})
    folders = {}
    for chat in (kept, removed):
        folder = tmp_path / "agent-workspace" / "files" / "chats" / chat["id"]
        folder.mkdir(parents=True)
        (folder / "note.txt").write_text("x", encoding="utf-8")
        folders[chat["id"]] = folder

    store.delete(removed["id"])
    assert not folders[removed["id"]].exists()
    assert folders[kept["id"]].exists()

    assert store.clear() == 1
    assert not folders[kept["id"]].exists()


def test_attachments_are_bounded_typed_and_safe(tmp_path):
    store = ChatStore(tmp_path)
    saved = store.save({"messages": [{
        "role": "user", "content": "Look at these",
        "attachments": [
            {"id": "a1", "name": "diagram.png", "kind": "image", "mime": "image/png",
             "size": 120, "data_url": "data:image/png;base64,AAAA", "secret": "dropped"},
            {"id": "a2", "name": "notes.md", "kind": "text", "mime": "text/markdown",
             "size": 40, "text": "# Notes", "language": "markdown"},
            {"name": "archive.bin", "kind": "binary"},
            {"name": "remote.png", "kind": "image", "data_url": "https://example.test/x.png"},
            "junk",
        ],
    }]})
    attachments = saved["messages"][0]["attachments"]
    assert [item["name"] for item in attachments] == ["diagram.png", "notes.md", "archive.bin"]
    assert attachments[0] == {"id": "a1", "name": "diagram.png", "kind": "image",
                              "mime": "image/png", "size": 120,
                              "data_url": "data:image/png;base64,AAAA"}
    assert attachments[1]["language"] == "markdown" and attachments[1]["text"] == "# Notes"
    assert "text" not in attachments[2]
    assert store.get(saved["id"])["messages"][0]["attachments"] == attachments

    limited = store.save({"messages": [{"role": "user", "content": "x", "attachments": [
        {"name": f"f{index}.txt", "kind": "text", "text": "x"} for index in range(20)
    ]}]})
    assert len(limited["messages"][0]["attachments"]) == 8
    huge = store.save({"messages": [{"role": "user", "content": "x", "attachments": [
        {"name": "huge.txt", "kind": "text", "text": "x" * 500_000}
    ]}]})
    assert len(huge["messages"][0]["attachments"][0]["text"]) == 200_000
    assert "attachments" not in store.save({
        "messages": [{"role": "user", "content": "x", "attachments": "not a list"}],
    })["messages"][0]
    invalid_images = store.save({"messages": [{"role": "user", "content": "x", "attachments": [
        {"name": "missing-data.png", "kind": "image", "data_url": "data:image/png;base64"},
        {"name": "empty-data.png", "kind": "image", "data_url": "data:image/png,"},
        {"name": "remote.png", "kind": "image", "data_url": "https://example.test/x.png"},
    ]}]})
    assert "attachments" not in invalid_images["messages"][0]


def test_swarm_cards_are_stored_bounded_and_typed(tmp_path):
    store = ChatStore(tmp_path)
    swarm = {
        "agents": [{
            "agent": "a1", "label": "Reader", "role": "researcher", "task": "read",
            "status": "done", "output": "x" * 9_000, "text": "y" * 5_000,
            "summary": "found it", "score": 4, "version": 2, "steps": 3,
            "elapsed": 1.25, "junk": "dropped",
        }] * 20,
        "meta": {"strategy": "peer_review", "round": 1, "rounds": 1,
                 "finished": True, "messageCount": 3, "unknown": "dropped"},
        "reviews": [{"reviewer": "a1", "target": "a1", "score": 5,
                      "issues": ["missing proof"], "suggestions": ["add a citation"],
                      "summary": "solid", "junk": "dropped"}] * 30,
        "messages": [{"from": "a1", "to": "all", "text": "done",
                       "message_kind": "status"}] * 30,
        "memory": [{"key": "finding.x", "value": {"answer": 42}}] * 30,
        "secret": "dropped",
    }
    saved = store.save({"messages": [
        {"role": "assistant", "content": "Answer", "swarm": swarm},
    ]})["messages"][0]["swarm"]
    assert len(saved["agents"]) == 12
    assert len(saved["agents"][0]["output"]) == 8_000
    assert len(saved["agents"][0]["text"]) == 4_000
    assert saved["agents"][0]["score"] == 4 and saved["agents"][0]["steps"] == 3
    assert "junk" not in saved["agents"][0]
    assert saved["meta"] == {"strategy": "peer_review", "round": 1, "rounds": 1,
                             "finished": True, "messageCount": 3}
    assert len(saved["reviews"]) == 20
    assert len(saved["messages"]) == 20 and saved["messages"][0]["kind"] == "status"
    assert len(saved["memory"]) == 20
    assert saved["memory"][0]["value"] == {"answer": 42}
    assert "secret" not in saved
    assert store.get(store.list()[0]["id"])["messages"][0]["swarm"] == saved

    huge = store.save({"messages": [{"role": "assistant", "content": "x", "swarm": {
        "agents": [{"agent": f"a{index}", "output": "x" * 8_000,
                    "text": "y" * 4_000, "task": "z" * 2_000} for index in range(12)],
        "reviews": [{"issues": ["i" * 300] * 4, "suggestions": ["s" * 300] * 4,
                     "summary": "m" * 400}] * 20,
        "messages": [{"from": "a", "to": "b", "text": "t" * 1_500}] * 20,
        "memory": [{"key": "k", "value": "v" * 1_000}] * 20,
    }}]})["messages"][0]["swarm"]
    assert len(json.dumps(huge)) <= 200_000
    assert "text" not in huge["agents"][0]  # detail is dropped before the card is lost
    assert "swarm" not in store.save({"messages": [
        {"role": "assistant", "content": "x", "swarm": "not a card"},
    ]})["messages"][0]


def test_automatic_titles_keep_updating_until_the_user_names_a_chat(tmp_path):
    store = ChatStore(tmp_path)
    chat = store.save({"messages": [{"role": "user", "content": "First question"}]})
    assert chat["title"] == "First question" and chat["titled"] is False
    chat = store.save({"messages": [{"role": "user", "content": "Edited question"}]}, chat["id"])
    assert chat["title"] == "Edited question" and chat["titled"] is False
    named = store.save({"title": "Mountains"}, chat["id"])
    assert named["title"] == "Mountains" and named["titled"] is True
    named = store.save({"messages": [{"role": "user", "content": "Again"}]}, chat["id"])
    assert named["title"] == "Mountains" and named["titled"] is True

    # A document saved before the flag existed keeps a title that differs from
    # the derived one, so editing the first message cannot silently rename it.
    folder = tmp_path / "agent-workspace" / "chats"
    (folder / "chat-legacy2.json").write_text(json.dumps({
        "id": "chat-legacy2", "title": "Chosen long ago",
        "messages": [{"role": "user", "content": "Derived"}],
    }))
    loaded = store.get("chat-legacy2")
    assert loaded["titled"] is True and loaded["title"] == "Chosen long ago"
    kept = store.save({"messages": [{"role": "user", "content": "Changed"}]}, "chat-legacy2")
    assert kept["title"] == "Chosen long ago"


def test_a_malformed_document_loads_in_the_api_shape(tmp_path):
    store = ChatStore(tmp_path)
    folder = tmp_path / "agent-workspace" / "chats"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "chat-legacy1.json").write_text(json.dumps({
        "id": "chat-somewhereelse",  # stale internal id must not win over the file name
        "title": 7,
        "model": {"bad": True},
        "settings": "not an object",
        "pinned": "yes",
        "messages": [
            {"role": "user", "content": "Hello"},
            "junk",
            {"role": "tool", "content": "dropped"},
            {"role": "assistant", "content": "Hi", "attachments": "not a list"},
            {"role": "assistant", "content": 42},
        ],
    }))
    loaded = store.get("chat-legacy1")
    assert loaded["id"] == "chat-legacy1"
    assert loaded["title"] == "Hello"  # a missing title is derived from the first user turn
    assert loaded["model"] == "{'bad': True}"
    assert loaded["settings"] == {}
    assert loaded["pinned"] is True
    assert [message["role"] for message in loaded["messages"]] == ["user", "assistant"]
    assert loaded["messages"][1]["content"] == "Hi"
    assert "attachments" not in loaded["messages"][1]
    assert store.list()[0]["id"] == "chat-legacy1"

    # A save through the file id must update that file, not the stale id.
    store.save({"title": "Recovered"}, "chat-legacy1")
    assert not (folder / "chat-somewhereelse.json").exists()
    assert store.get("chat-legacy1")["title"] == "Recovered"
    assert store.list()[0]["id"] == "chat-legacy1"


def test_listing_tolerates_malformed_message_shapes(tmp_path):
    store = ChatStore(tmp_path)
    folder = tmp_path / "agent-workspace" / "chats"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "chat-broken1.json").write_text(json.dumps({
        "id": "chat-broken1", "title": "Broken", "messages": {"not": "a list"},
    }))
    (folder / "chat-broken2.json").write_text(json.dumps({
        "id": "chat-broken2", "title": "Odd", "messages": ["junk", {"role": "user", "content": 5}],
    }))
    listed = store.list()
    assert {chat["id"]: chat["preview"] for chat in listed} == {
        "chat-broken1": "", "chat-broken2": "",
    }


def test_non_finite_and_oversized_metadata_is_sanitised(tmp_path):
    store = ChatStore(tmp_path)
    folder = tmp_path / "agent-workspace" / "chats"
    folder.mkdir(parents=True, exist_ok=True)
    deep = "[" * 200_000 + "1" + "]" * 200_000
    (folder / "chat-deep001.json").write_text(
        '{"id":"chat-deep001","messages":[{"role":"user","content":"x"}],"swarm":'
        + deep + "}")
    (folder / "chat-odd001.json").write_text(json.dumps({
        "id": "chat-odd001", "title": "Odd",
        "created": float("nan"), "updated": float("inf"),
        "settings": {"temperature": float("nan"), "stop": ["a", {"x": 1}, 5],
                     "nested": {"x": 1}, "prompt_format": "auto"},
        "messages": [{
            "role": "assistant", "content": "hi",
            "usage": {"completion_tokens": float("nan"), "estimated": True},
            "elapsed": float("inf"), "reasoning": {"deep": [1, 2]},
            "swarm": {
                "agents": [{"agent": "a1", "score": float("nan"), "output": "x"}],
                "memory": [{"key": "k", "value": float("nan")}],
                "meta": {"elapsed": float("inf")},
            },
        }],
    }))

    listed = {chat["id"]: chat for chat in store.list()}
    assert "chat-deep001" not in listed  # unreadable JSON is skipped, not a crash
    assert listed["chat-odd001"]["updated"] == 0 and listed["chat-odd001"]["created"] is None
    with pytest.raises(NotFoundError):
        store.get("chat-deep001")

    loaded = store.get("chat-odd001")
    assert loaded["settings"] == {"stop": ["a", 5], "prompt_format": "auto"}
    message = loaded["messages"][0]
    assert message["usage"] == {"estimated": True}
    assert "elapsed" not in message
    assert message["reasoning"] == "{'deep': [1, 2]}"
    assert "score" not in message["swarm"]["agents"][0]
    assert message["swarm"]["memory"][0]["value"] is None
    assert "elapsed" not in message["swarm"]["meta"]
    json.dumps(loaded, allow_nan=False)  # the API response must reach the browser

    saved = store.save({"messages": [{
        "role": "user", "content": "x",
        "attachments": [{"name": "f.bin", "kind": "file", "size": float("inf")}],
    }]})
    assert saved["messages"][0]["attachments"][0]["size"] == 0


def test_non_finite_settings_are_dropped_before_saving(tmp_path):
    store = ChatStore(tmp_path)
    saved = store.save({"settings": {
        "temperature": float("nan"), "top_p": 0.9, "stop": ["END"],
        "nested": {"x": 1}, "candidates": float("inf"),
    }})
    assert saved["settings"] == {"top_p": 0.9, "stop": ["END"]}


def test_oversized_chat_files_are_skipped_and_refused(tmp_path, monkeypatch):
    import mariana.studio.chats as chats

    monkeypatch.setattr(chats, "MAX_STORED_BYTES", 1_024)
    store = ChatStore(tmp_path)
    folder = tmp_path / "agent-workspace" / "chats"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "chat-huge001.json").write_text(json.dumps({
        "id": "chat-huge001",
        "messages": [{"role": "user", "content": "x" * 2000}],
    }))
    assert store.list() == []
    with pytest.raises(TooLargeError, match="too large to open"):
        store.get("chat-huge001")


def test_bad_input_and_unsafe_ids_are_rejected(tmp_path):
    store = ChatStore(tmp_path)
    for unsafe in ("../../etc/passwd", "chat-../x", "", "chat-UPPER123", "other-abcdef"):
        with pytest.raises(NotFoundError):
            store.get(unsafe)
    with pytest.raises(BadRequestError, match="must be a list"):
        store.save({"messages": "hello"})
    with pytest.raises(BadRequestError, match="at most 2000 messages"):
        store.save({"messages": [{"role": "user", "content": "x"}] * 2001})
    with pytest.raises(BadRequestError, match="JSON object"):
        store.save(["not", "a", "dict"])
    (tmp_path / "agent-workspace" / "chats" / "chat-corrupt1.json").write_text("{broken")
    assert store.list() == []
    long = store.save({"messages": [{"role": "user", "content": "x" * 500_000}]})
    assert len(long["messages"][0]["content"]) == 400_000
