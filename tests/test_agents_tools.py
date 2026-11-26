"""Sandboxed coding abilities: atomic edits, search, tree and diagnostics."""

from __future__ import annotations

import http.client
import os
import shutil
import sqlite3
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from mariana.agents.tools import (
    ToolError,
    _GuardedRedirectHandler,
    _guarded_connection,
    _private_address,
    execute_ability,
    missing_required_arguments,
    normalise_tool_arguments,
    resolve_tool_id,
)
from tests.helpers import mock_provider, serving


def run(tmp_path: Path, ability_type: str, payload=None, **config):
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    return execute_ability(
        {"id": "probe", "type": ability_type, "config": config}, payload or {},
        {"agent": {"id": "test"}}, tmp_path, files_root=project,
    )


def test_file_root_override_edits_atomically_and_refuses_ambiguity(tmp_path):
    run(tmp_path, "file.write", {"path": "notes.txt", "content": "alpha\nbeta\nalpha\n"})
    project = tmp_path / "project"
    assert (project / "notes.txt").read_text() == "alpha\nbeta\nalpha\n"

    with pytest.raises(ToolError, match="occurs 2 times"):
        run(tmp_path, "file.edit", {"path": "notes.txt", "old": "alpha", "new": "gamma"})
    result = run(tmp_path, "file.edit", {
        "path": "notes.txt", "old": "beta", "new": "delta",
    })
    assert result["changed"] is True and result["replacements"] == 1
    assert "delta" in (project / "notes.txt").read_text()

    with pytest.raises(ToolError, match="not found"):
        run(tmp_path, "file.edit", {"path": "notes.txt", "old": "missing", "new": "x"})
    with pytest.raises(ToolError, match="inside"):
        run(tmp_path, "file.read", {"path": "../outside.txt"})


def test_file_edit_refuses_to_corrupt_non_utf8_bytes(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    path = project / "latin1.txt"
    path.write_bytes(b"caf\xe9\n")

    with pytest.raises(ToolError, match="not UTF-8"):
        run(tmp_path, "file.edit", {"path": "latin1.txt", "old": "caf", "new": "tea"})
    assert path.read_bytes() == b"caf\xe9\n"


def test_file_read_reports_the_last_included_line_and_the_continuation(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "long.py").write_text(
        "".join(f"value_{index} = {index}  # " + "x" * 80 + "\n" for index in range(1, 201)),
        encoding="utf-8",
    )

    first = run(tmp_path, "file.read", {"path": "long.py"})
    assert first["start_line"] == 1 and first["total_lines"] == 200
    assert first["end_line"] == first["content"].count("\n")
    assert first["end_line"] < 200 and first["truncated"] is True
    assert first["next_start_line"] == first["end_line"] + 1

    second = run(tmp_path, "file.read", {
        "path": "long.py", "start_line": first["next_start_line"],
    })
    assert second["start_line"] == first["next_start_line"]
    assert second["end_line"] >= second["start_line"]
    assert second["content"].startswith(f"value_{second['start_line']} = ")


def test_file_read_marks_one_line_that_exceeds_the_character_budget(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "minified.js").write_text("x" * 5_000 + "\n", encoding="utf-8")

    result = run(tmp_path, "file.read", {"path": "minified.js"})
    assert result["content_cut_midline"] is True
    assert len(result["content"]) < 5_000
    assert result["end_line"] == 1 and "next_start_line" not in result
    assert "hint" in result


def test_search_tree_and_manifest_are_scoped(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / ".mariana-ide.json").write_text('{"name": "private"}\n', encoding="utf-8")
    (project / "src").mkdir()
    (project / "src" / "app.py").write_text("import os\n\nprint('needle')\n", encoding="utf-8")
    (project / "node_modules").mkdir()
    (project / "node_modules" / "dep.js").write_text("needle\n", encoding="utf-8")

    found = run(tmp_path, "file.search", {"query": "needle", "regex": False})
    assert [match["path"] for match in found["matches"]] == ["src/app.py"]
    tree = run(tmp_path, "file.tree", {})
    paths = {entry["path"] for entry in tree["entries"]}
    assert "src/app.py" in paths and "node_modules" not in paths
    assert ".mariana-ide.json" not in paths


def test_diagnostics_reports_python_and_json_problems(tmp_path):
    run(tmp_path, "file.write", {"path": "ok.py", "content": "def add(a, b):\n    return a + b\n"})
    run(tmp_path, "file.write", {"path": "bad.py", "content": "def broken(:\n    pass\n"})
    run(tmp_path, "file.write", {"path": "data.json", "content": '{"a": 1,}\n'})
    report = run(tmp_path, "code.diagnostics", {})
    assert report["ok"] is False and report["files_checked"] == 3
    rules = {(issue["path"], issue["rule"]) for issue in report["issues"]}
    assert ("bad.py", "syntax") in rules and ("data.json", "json") in rules
    assert all(issue["line"] >= 1 for issue in report["issues"])


def test_symlinks_cannot_expose_files_outside_the_sandbox(tmp_path):
    if os.name == "nt":
        pytest.skip("creating symlinks needs extra privileges on Windows")
    secret = tmp_path / "secret.txt"
    secret.write_text("outside secret\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()
    (project / "inside.txt").write_text("inside\n", encoding="utf-8")
    (project / "link.txt").symlink_to(secret)
    (project / "inside-link.txt").symlink_to(project / "inside.txt")
    found = run(tmp_path, "file.search", {"query": "secret", "regex": False})
    assert found["matches"] == []
    tree = run(tmp_path, "file.tree", {})
    assert "link.txt" not in {entry["path"] for entry in tree["entries"]}
    with pytest.raises(ToolError, match="symlink|inside"):
        run(tmp_path, "file.read", {"path": "link.txt"})
    inside = run(tmp_path, "file.read", {"path": "inside-link.txt"})
    assert inside["content"] == "inside\n"


def test_search_falls_back_to_literal_text_for_an_invalid_regex(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.cpp").write_text("if (strlen(value) > 3) {\n", encoding="utf-8")
    found = run(tmp_path, "file.search", {"query": "strlen(", "regex": True})
    assert found["regex_fallback"] is True
    assert found["matches"] and found["matches"][0]["path"] == "main.cpp"
    assert "literal" in found["hint"]


def test_read_accepts_a_path_prefixed_with_the_project_folder(tmp_path):
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "src" / "main.cpp").write_text("int main() {}\n", encoding="utf-8")
    result = run(tmp_path, "file.read", {"path": "project/src/main.cpp"})
    assert result["content"] == "int main() {}\n"


def test_read_lists_a_directory_and_list_walks_subfolders(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "app.py").write_text("print('x')\n", encoding="utf-8")
    (project / "src" / "widgets").mkdir()

    directory = run(tmp_path, "file.read", {"path": "src"})
    assert directory["directory"] is True
    assert {entry["type"] for entry in directory["entries"]} == {"dir", "file"}
    assert any(entry["path"] == "src/app.py" for entry in directory["entries"])

    listed = run(tmp_path, "file.list", {"path": "src", "pattern": "*.py"})
    assert [item["path"] for item in listed["entries"]] == ["src/app.py"]
    assert listed["files"][0]["name"] == "app.py"


def test_read_streams_a_file_above_the_read_cache_limit(tmp_path, monkeypatch):
    from mariana.agents import tools as tools_module

    monkeypatch.setattr(tools_module, "MAX_READ_STREAM_BYTES", 120)
    project = tmp_path / "project"
    project.mkdir()
    (project / "big.txt").write_text(
        "".join(f"line {index:03d}\n" for index in range(500)), encoding="utf-8")
    first = run(tmp_path, "file.read", {"path": "big.txt"})
    assert first["large_file"] is True and first["truncated"] is True
    assert first["total_lines"] is None
    assert first["next_start_line"] == first["end_line"] + 1
    second = run(tmp_path, "file.read", {
        "path": "big.txt", "start_line": first["next_start_line"],
    })
    assert second["start_line"] == first["next_start_line"]
    assert second["content"].startswith(f"line {second['start_line'] - 1:03d}")


def test_delete_checks_existence_and_dry_run_skips_side_effects(tmp_path):
    run(tmp_path, "file.write", {"path": "remove.txt", "content": "bye"})
    project = tmp_path / "project"
    assert (project / "remove.txt").exists()
    with pytest.raises(ToolError, match="not found"):
        run(tmp_path, "file.delete", {"path": "missing.txt"})
    result = execute_ability(
        {"id": "probe", "type": "file.delete"}, {"path": "remove.txt"},
        {"agent": {"id": "test"}}, tmp_path, dry_run=True, files_root=project,
    )
    assert result["dry_run"] is True and (project / "remove.txt").exists()


def test_sqlite_queries_are_bound_read_only_and_bounded(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    connection = sqlite3.connect(project / "app.db")
    connection.execute("CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT)")
    connection.executemany("INSERT INTO people (name) VALUES (?)", [("Ada",), ("Linus",), ("Grace",)])
    connection.commit()
    connection.close()

    found = run(tmp_path, "sqlite.query", {
        "sql": "SELECT id, name FROM people ORDER BY id",
    }, path="app.db")
    assert found["columns"] == ["id", "name"]
    assert found["rows"] == [[1, "Ada"], [2, "Linus"], [3, "Grace"]]

    bound = run(tmp_path, "sqlite.query", {
        "sql": "SELECT name FROM people WHERE id = ?", "params": [2],
    }, path="app.db")
    assert bound["rows"] == [["Linus"]]

    limited = run(tmp_path, "sqlite.query", {
        "sql": "SELECT name FROM people ORDER BY id", "max_rows": 2,
    }, path="app.db")
    assert limited["row_count"] == 2 and limited["truncated"] is True

    with pytest.raises(ToolError, match="read-only"):
        run(tmp_path, "sqlite.query", {"sql": "DELETE FROM people"}, path="app.db")
    with pytest.raises(ToolError, match="read-only"):
        # A model or tool node cannot widen the ability's write permission.
        run(tmp_path, "sqlite.query", {"sql": "DELETE FROM people", "write": True}, path="app.db")
    changed = run(tmp_path, "sqlite.query", {
        "sql": "DELETE FROM people WHERE name = ?", "params": ["Grace"],
    }, path="app.db", write=True)
    assert changed["changes"] == 1 and changed["write"] is True
    counted = run(tmp_path, "sqlite.query", {"sql": "SELECT COUNT(*) AS n FROM people"},
                  path="app.db")
    assert counted["rows"] == [[2]]

    skipped = execute_ability(
        {"id": "db", "type": "sqlite.query", "config": {"path": "app.db", "write": True}},
        {"sql": "DELETE FROM people"}, {"agent": {"id": "test"}}, tmp_path,
        dry_run=True, files_root=project,
    )
    assert skipped["dry_run"] is True
    assert run(tmp_path, "sqlite.query", {"sql": "SELECT COUNT(*) AS n FROM people"},
               path="app.db")["rows"] == [[2]]
    with pytest.raises(ToolError, match="inside"):
        run(tmp_path, "sqlite.query", {"sql": "SELECT 1"}, path="../app.db")


def test_csv_reads_headers_values_and_tsv_delimiters(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "people.csv").write_text("name,age\nAda,36\nLinus,54\n", encoding="utf-8")
    (project / "notes.tsv").write_text("key\tvalue\nalpha\t1\n", encoding="utf-8")

    found = run(tmp_path, "csv.read", {"path": "people.csv"})
    assert found["columns"] == ["name", "age"]
    assert found["rows"] == [{"name": "Ada", "age": "36"}, {"name": "Linus", "age": "54"}]
    assert found["row_count"] == 2 and found["truncated"] is False

    tsv = run(tmp_path, "csv.read", {"path": "notes.tsv"})
    assert tsv["delimiter"] == "\t" and tsv["rows"][0] == {"key": "alpha", "value": "1"}

    plain = run(tmp_path, "csv.read", {"path": "people.csv", "header": False})
    assert plain["columns"] == ["column_1", "column_2"]
    assert plain["rows"][0] == {"column_1": "name", "column_2": "age"}

    limited = run(tmp_path, "csv.read", {"path": "people.csv", "max_rows": 1})
    assert limited["row_count"] == 1 and limited["truncated"] is True
    with pytest.raises(ToolError, match="one character"):
        run(tmp_path, "csv.read", {"path": "people.csv", "delimiter": "::"})


@pytest.mark.skipif(not shutil.which("git"), reason="git is not installed")
def test_git_search_caps_a_broad_result_without_buffering_it(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    git = shutil.which("git")
    subprocess.run([git, "init", "-q"], cwd=project, check=True)
    (project / "many.txt").write_text("needle\n" * 300, encoding="utf-8")
    found = run(tmp_path, "file.search", {"query": "needle", "regex": False,
                                           "max_results": 5})
    assert found["engine"] == "git" and len(found["matches"]) == 5
    assert found["truncated"] is True


@pytest.mark.skipif(not shutil.which("git"), reason="git is not installed")
def test_git_inspection_reports_status_diff_and_log_read_only(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    git = shutil.which("git")
    subprocess.run([git, "init", "-q"], cwd=project, check=True)
    subprocess.run([git, "config", "user.email", "test@example.com"], cwd=project, check=True)
    subprocess.run([git, "config", "user.name", "Test"], cwd=project, check=True)
    (project / "a.txt").write_text("one\n", encoding="utf-8")
    subprocess.run([git, "add", "a.txt"], cwd=project, check=True)
    subprocess.run([git, "commit", "-qm", "first"], cwd=project, check=True)
    (project / "a.txt").write_text("two\n", encoding="utf-8")

    status = run(tmp_path, "git.inspect", {"action": "status"})
    assert status["clean"] is False
    assert status["files"] == [{"status": "M", "path": "a.txt"}]

    diff = run(tmp_path, "git.inspect", {"action": "diff", "path": "a.txt"})
    assert "-one" in diff["diff"] and "+two" in diff["diff"] and diff["path"] == "a.txt"

    log = run(tmp_path, "git.inspect", {"action": "log"})
    assert log["commits"][0]["subject"] == "first"

    aliased = run(tmp_path, "git.inspect", {"action": "history 5"})
    assert aliased["action"] == "log" and aliased["commits"][0]["subject"] == "first"

    with pytest.raises(ToolError, match="action must be"):
        run(tmp_path, "git.inspect", {"action": "push"})


def test_git_inspection_reports_a_missing_repository_without_failing(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    result = execute_ability({"id": "git", "type": "git.inspect", "config": {}}, {},
                             {"agent": {"id": "test"}}, tmp_path, files_root=project)
    assert result["repository"] is False and result["files"] == []
    assert "No Git repository" in result["hint"]


@pytest.mark.parametrize("address", [
    "127.0.0.1", "10.1.2.3", "169.254.169.254", "192.168.1.1", "::1",
    "::ffff:127.0.0.1", "100.64.0.1", "0.0.0.0",
])
def test_private_addresses_are_recognised(address):
    assert _private_address(address) is True


def test_public_addresses_are_allowed():
    assert _private_address("8.8.8.8") is False
    assert _private_address("1.1.1.1") is False
    assert _private_address("2606:4700:4700::1111") is False


def test_redirect_targets_are_validated_before_they_are_followed():
    seen = []

    def validate(url):
        seen.append(url)
        if "127.0.0.1" in url:
            raise ToolError("Private addresses are blocked")

    handler = _GuardedRedirectHandler(validate)
    request = urllib.request.Request("https://example.com/")
    with pytest.raises(ToolError, match="Private addresses"):
        handler.redirect_request(request, None, 302, "Found", {}, "http://127.0.0.1/admin")
    assert seen == ["http://127.0.0.1/admin"]


def test_the_connected_peer_is_checked_after_dns_resolution(tmp_path):
    provider = mock_provider({})
    with serving(provider) as base:
        host, port = base.rsplit("//", 1)[1].split(":")
        blocked = _guarded_connection(http.client.HTTPConnection, False)(host, int(port), timeout=5)
        try:
            with pytest.raises(ToolError, match="Private addresses"):
                blocked.connect()
        finally:
            blocked.close()
        allowed = _guarded_connection(http.client.HTTPConnection, True)(host, int(port), timeout=5)
        try:
            allowed.connect()
        finally:
            allowed.close()


def test_model_input_cannot_widen_the_private_network_opt_in(tmp_path):
    fetch = {"id": "fetch", "type": "http.fetch", "config": {"allow_private": False}}
    with pytest.raises(ToolError, match="Private addresses"):
        execute_ability(
            fetch, {"url": "http://127.0.0.1:9/", "allow_private": True},
            {"agent": {"id": "test"}}, tmp_path,
        )


RSS_FEED = b'''<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>India Daily</title>
<item><title>Budget passes after debate</title><link>https://example.com/budget</link>
<source>The Hindu</source><pubDate>Thu, 20 Nov 2026 05:00:00 GMT</pubDate>
<description><![CDATA[<p>The budget passed after a long debate.</p>]]></description></item>
<item><title>Rain expected in the north</title><link>https://example.com/weather</link>
<source>BBC News</source><pubDate>Thu, 20 Nov 2026 06:00:00 GMT</pubDate>
<description>Light rain is expected.</description></item>
</channel></rss>'''


def test_news_headlines_parses_rss_and_filters(tmp_path):
    provider = mock_provider({
        ("GET", "/india.xml"): lambda handler, payload: (
            200, RSS_FEED, {"Content-Type": "application/rss+xml"}),
    })
    with serving(provider) as base:
        result = execute_ability(
            {"id": "news", "type": "news.headlines", "config": {"allow_private": True}},
            {"url": base + "/india.xml", "query": "budget", "limit": 10,
             "include_summary": True},
            {"agent": {"id": "test"}}, tmp_path,
        )
    assert result["feed_title"] == "India Daily"
    assert result["count"] == 1 and result["total"] == 1
    item = result["items"][0]
    assert item["title"] == "Budget passes after debate"
    assert item["source"] == "The Hindu"
    assert item["date"] == "2026-11-20"
    assert "long debate" in item["summary"]
    assert item["link"].endswith("/budget")


def test_news_headlines_falls_back_to_the_newest_items_for_a_missing_date(tmp_path):
    provider = mock_provider({
        ("GET", "/india.xml"): lambda handler, payload: (
            200, RSS_FEED, {"Content-Type": "application/rss+xml"}),
    })
    with serving(provider) as base:
        result = execute_ability(
            {"id": "news", "type": "news.headlines", "config": {"allow_private": True}},
            {"url": base + "/india.xml", "date": "2026-11-21", "limit": 10},
            {"agent": {"id": "test"}}, tmp_path,
        )
    assert result["date_matched"] is False
    assert result["requested_date"] == "2026-11-21"
    assert result["latest_date"] == "2026-11-20"
    assert result["items"] and "No items dated 2026-11-21" in result["note"]
    assert "summary" not in result["items"][0]  # compact by default


def test_news_headlines_builds_a_google_news_search_url(tmp_path):
    result = execute_ability(
        {"id": "news", "type": "news.headlines", "config": {}},
        {"query": "India news", "region": "IN", "language": "en", "limit": 5},
        {"agent": {"id": "test"}}, tmp_path, dry_run=True,
    )
    assert "news.google.com/rss/search" in result["url"]
    assert "q=India+news" in result["url"]
    assert "gl=IN" in result["url"] and "ceid=IN:en" in result["url"]
    assert result["items"] == [] and result["dry_run"] is True

    bing = execute_ability(
        {"id": "news", "type": "news.headlines", "config": {"source": "bing"}},
        {"query": "Pakistan", "region": "PK"}, {"agent": {"id": "test"}}, tmp_path,
        dry_run=True,
    )
    assert "bing.com/news/search" in bing["url"] and "setmkt=en-PK" in bing["url"]


def test_news_headlines_falls_back_from_google_to_bing(tmp_path, monkeypatch):
    from mariana.agents import tools as tools_module

    def fetch(self, params, context):
        if "google" in params["url"]:
            raise tools_module.ToolError("blocked")
        return {"ok": True, "status": 200, "body": RSS_FEED.decode("utf-8")}

    monkeypatch.setattr(tools_module._AbilityRunner, "http_fetch", fetch)
    result = execute_ability(
        {"id": "news", "type": "news.headlines", "config": {"source": "auto"}},
        {"query": "Pakistan", "region": "PK", "limit": 5},
        {"agent": {"id": "test"}}, tmp_path,
    )
    assert result["source_used"] == "bing" and result["count"] >= 1
    assert result["fallbacks"] and "google" in result["fallbacks"][0]


def test_browser_fetch_requires_os_access(tmp_path):
    with pytest.raises(ToolError, match="OS access"):
        execute_ability(
            {"id": "browser", "type": "browser.fetch", "config": {}},
            {"url": "http://127.0.0.1:1/"}, {"agent": {"id": "test"}}, tmp_path,
            os_access=False,
        )


def test_browser_fetch_falls_back_to_http_text(tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr("mariana.agents.tools.shutil.which", lambda name: None)
    monkeypatch.setitem(sys.modules, "playwright", None)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    provider = mock_provider({
        ("GET", "/page"): lambda handler, payload: (
            200, b"<html><head><title>Page</title></head><body><p>Fallback text</p></body></html>",
            {"Content-Type": "text/html"}),
    })
    with serving(provider) as base:
        result = execute_ability(
            {"id": "browser", "type": "browser.fetch", "config": {}},
            {"url": base + "/page"}, {"agent": {"id": "test"}}, tmp_path,
            os_access=True,
        )
    assert result["engine"] == "http"
    assert "Fallback text" in result["text"]
    assert "No headless browser" in result["note"]


def test_browser_fetch_uses_a_chromium_binary(tmp_path, monkeypatch):
    fake = tmp_path / "chromium"
    fake.write_text(
        "#!/bin/sh\nprintf '<html><head><title>Example</title></head>"
        "<body><p>Hello rendered</p></body></html>'\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    monkeypatch.setattr("mariana.agents.tools.shutil.which",
                        lambda name: str(fake) if name == "chromium" else None)
    result = execute_ability(
        {"id": "browser", "type": "browser.fetch", "config": {}},
        {"url": "http://127.0.0.1:1/"}, {"agent": {"id": "test"}}, tmp_path,
        os_access=True,
    )
    assert result["engine"] == "chromium"
    assert result["title"] == "Example"
    assert "Hello rendered" in result["text"]


def test_an_invalid_news_feed_reports_an_actionable_error(tmp_path):
    provider = mock_provider({
        ("GET", "/not-a-feed"): lambda handler, payload: (
            200, b"<html><body>not a feed</body></html>",
            {"Content-Type": "text/html"}),
    })
    with serving(provider) as base:
        with pytest.raises(ToolError, match="RSS or Atom"):
            execute_ability(
                {"id": "news", "type": "news.headlines", "config": {"allow_private": True}},
                {"url": base + "/not-a-feed"}, {"agent": {"id": "test"}}, tmp_path,
            )


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell syntax")
def test_os_access_enables_shell_without_the_environment_flag(tmp_path, monkeypatch):
    monkeypatch.delenv("MARIANA_AGENT_ALLOW_SHELL", raising=False)
    monkeypatch.delenv("MARIANA_AGENT_ALLOW_OS", raising=False)
    ability = {"id": "probe", "type": "shell.exec", "config": {}}
    result = execute_ability(ability, {"command": "printf os-access"},
                             {"agent": {"id": "test"}}, tmp_path, os_access=True)
    assert result["ok"] is True and result["stdout"] == "os-access"
    with pytest.raises(ToolError, match="OS access|disabled"):
        execute_ability(ability, {"command": "printf blocked"},
                        {"agent": {"id": "test"}}, tmp_path, os_access=False)


def test_os_access_allows_private_http(tmp_path):
    provider = mock_provider({("GET", "/ok"): lambda handler, payload: (200, {"ok": True})})
    with serving(provider) as base:
        result = execute_ability(
            {"id": "fetch", "type": "http.fetch", "config": {}},
            {"url": base + "/ok"}, {"agent": {"id": "test"}}, tmp_path,
            os_access=True,
        )
    assert result["ok"] is True and result["json"] == {"ok": True}


@pytest.mark.skipif(os.name == "nt", reason="process groups are POSIX-specific")
def test_shell_timeout_stops_the_whole_process_tree(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIANA_AGENT_ALLOW_SHELL", "1")
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(ToolError, match="timed out"):
        execute_ability(
            {"id": "probe", "type": "shell.exec", "config": {}},
            {"command": "sleep 2; touch late.txt", "timeout": 1},
            {"agent": {"id": "test"}}, tmp_path, files_root=project,
        )
    time.sleep(1.5)
    assert not (project / "late.txt").exists()


def test_dry_run_http_fetch_returns_the_real_shape_without_dns(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("test mode must not resolve DNS")

    monkeypatch.setattr("mariana.agents.tools.socket.getaddrinfo", no_network)
    fetch = {"id": "fetch", "type": "http.fetch", "config": {}}
    context = {"agent": {"id": "test"}}
    raw = execute_ability(
        fetch, {"url": "https://example.com/api"}, context, tmp_path, dry_run=True,
    )
    assert raw["ok"] is True and raw["status"] == 200
    assert "example.com" in raw["body"] and raw["json"]["dry_run"] is True
    text = execute_ability(
        fetch, {"url": "https://example.com/page", "extract": "text"},
        context, tmp_path, dry_run=True,
    )
    assert text["text"].startswith("[test mode]") and "body" not in text
    with pytest.raises(ToolError, match="Private addresses"):
        execute_ability(fetch, {"url": "http://127.0.0.1/"}, context, tmp_path, dry_run=True)


@pytest.mark.skipif(os.name == "nt", reason="process groups are POSIX-specific")
def test_shell_cancellation_stops_the_whole_process_tree(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIANA_AGENT_ALLOW_SHELL", "1")
    project = tmp_path / "project"
    project.mkdir()
    cancelled = threading.Event()
    outcome = {}

    def work():
        try:
            execute_ability(
                {"id": "probe", "type": "shell.exec", "config": {}},
                {"command": "sleep 0.8; touch cancelled-late.txt", "timeout": 30},
                {"agent": {"id": "test"}}, tmp_path, files_root=project,
                cancelled=cancelled,
            )
        except ToolError as error:
            outcome["error"] = str(error)

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    time.sleep(0.2)
    cancelled.set()
    thread.join(timeout=5)
    assert not thread.is_alive() and "cancelled" in outcome.get("error", "").lower()
    time.sleep(1.0)
    assert not (project / "cancelled-late.txt").exists()


def test_tool_arguments_accept_bare_strings_and_alias_spellings():
    assert normalise_tool_arguments("file.search", "needle") == {"query": "needle"}
    assert normalise_tool_arguments("file.read", {"filePath": "src/main.py", "offset": "10",
                                                  "limit": "5"}) == {
        "path": "src/main.py", "start_line": 10, "end_line": 14,
    }
    assert normalise_tool_arguments("file.search", {
        "q": "needle", "include_pattern": "*.py", "maxResults": "20",
    }) == {"query": "needle", "include": "*.py", "max_results": 20}
    assert normalise_tool_arguments("file.write", {
        "file_path": "notes.txt", "contents": "body",
    }) == {"path": "notes.txt", "content": "body"}

    assert resolve_tool_id("file_search", ["file.read", "file.search"]) == "file.search"
    assert resolve_tool_id("file.search", ["file.read", "file.search"]) == "file.search"
    assert resolve_tool_id("file.orphan", ["file.read", "file.search"]) == ""


def test_missing_required_tool_arguments_are_reported_by_canonical_key():
    assert missing_required_arguments("file.search", {}) == ["query"]
    assert missing_required_arguments(
        "file.search", normalise_tool_arguments("file.search", {"q": "x"})) == []
    assert missing_required_arguments("file.read", {"path": "   "}) == ["path"]
    assert missing_required_arguments(
        "file.write", normalise_tool_arguments("file.write", {"path": "a.txt"})) == ["content"]
    assert missing_required_arguments("time.now", {}) == []
