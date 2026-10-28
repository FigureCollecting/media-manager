"""LG-U11: the redaction policy, ported from fc-shared 1.8.0.

The cases are fc-shared's own (tests/utils/sanitize.test.ts and the
redaction tables of tests/server/log.test.ts), so the Python port masks and
keeps exactly what the TypeScript library does for the same input.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

import pytest
from app.telemetry import redact as r
from app.telemetry.log import Logger

GHP = "ghp_0123456789abcdefghijABCDEFGHIJ012345"


def _line(**fields: Any) -> dict[str, Any]:
    lines: list[str] = []
    Logger(service="t", version="1", env={}, sink=lines.append).info("", **fields)
    assert len(lines) == 1 and "\n" not in lines[0]
    return json.loads(lines[0])  # type: ignore[no-any-return]


def _raw(msg: str = "", **fields: Any) -> str:
    lines: list[str] = []
    Logger(service="t", version="1", env={}, sink=lines.append).info(msg, **fields)
    return lines[0]


# --- sanitize.test.ts: redactString --------------------------------------------------------


def test_masks_bearer_tokens() -> None:
    assert r.redact_string("Authorization: Bearer abc.def-123_xyz") == "Authorization: [REDACTED]"


def test_masks_jwts() -> None:
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.s5Hdq8KmqZ0"
    assert r.redact_string(f"token={jwt}") == "token=[REDACTED]"


@pytest.mark.parametrize(
    "secret",
    [
        GHP,
        "xoxb-1111111111-abcdefghij",
        "AKIAIOSFODNN7EXAMPLE",
        "AIzaSyD-1234567890abcdefghijklmnopqrstuv",
        "github_pat_11ABCDEFG0123456789_abcdefghij",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----",
    ],
)
def test_masks_github_slack_aws_google_and_pem_credentials(secret: str) -> None:
    assert r.redact_string(secret) == "[REDACTED]"


def test_leaves_innocent_strings_unchanged() -> None:
    text = "a normal log message about figure 42"
    assert r.redact_string(text) == text


def test_truncates_over_long_strings() -> None:
    out = r.redact_string("x" * 9000, max_length=100)
    assert out.endswith("…[truncated]") and len(out) == 100 + len("…[truncated]")


def test_is_stateless_across_calls() -> None:
    assert [r.redact_string("Bearer aaaaaaaaaa") for _ in range(3)] == ["[REDACTED]"] * 3


def test_honours_a_custom_placeholder() -> None:
    assert r.redact_string("Bearer xyzxyzxyzx", placeholder="***") == "***"


# --- sanitize.test.ts: redactValue ---------------------------------------------------------


def test_redacts_values_of_sensitive_keys_wholesale() -> None:
    out = r.redact_value(
        {
            "username": "rob",
            "password": "hunter2",
            "apiKey": "k-123",
            "authorization": "Bearer abc",
            "session_id": "sess-1",
        }
    )
    assert out == {
        "username": "rob",
        "password": "[REDACTED]",
        "apiKey": "[REDACTED]",
        "authorization": "[REDACTED]",
        "session_id": "[REDACTED]",
    }


def test_scrubs_secret_shapes_inside_innocent_keyed_strings() -> None:
    assert r.redact_value({"note": f"use {GHP} to auth"}) == {"note": "use [REDACTED] to auth"}


def test_recurses_into_nested_objects_and_arrays() -> None:
    out = r.redact_value({"outer": {"inner": {"token": "t"}, "items": [{"secret": "s"}, "ok"]}})
    assert out == {
        "outer": {"inner": {"token": "[REDACTED]"}, "items": [{"secret": "[REDACTED]"}, "ok"]}
    }


def test_does_not_mutate_the_original() -> None:
    original = {"password": "hunter2", "nested": {"token": "t"}}
    r.redact_value(original)
    assert original == {"password": "hunter2", "nested": {"token": "t"}}


def test_collapses_circular_references() -> None:
    a: dict[str, Any] = {"name": "a"}
    a["self"] = a
    assert r.redact_value(a) == {"name": "a", "self": "[circular]"}


def test_caps_recursion_depth() -> None:
    out = r.redact_value({"v": {"v": {"v": {"v": {}}}}}, max_depth=2)
    assert out == {"v": {"v": "[truncated:max-depth]"}}


def test_handles_error_date_binary_and_primitive_leaves() -> None:
    out = r.redact_value(
        {
            "err": RuntimeError(f"failed with token {GHP}"),
            "when": dt.datetime(2026, 6, 11, tzinfo=dt.UTC),
            "bytes": b"\x01\x02\x03",
            "n": 7,
            "ok": True,
            "missing": None,
            "fn": len,
            "other": object(),
        }
    )
    assert out["err"]["name"] == "RuntimeError"
    assert out["err"]["message"] == "failed with token [REDACTED]"
    assert out["when"] == "2026-06-11T00:00:00.000Z"
    assert out["bytes"] == "[binary]"
    assert (out["n"], out["ok"], out["missing"]) == (7, True, None)
    assert out["fn"] == "[function]"
    assert out["other"] == "[object]"


# --- log.test.ts: values as the logger prints them -------------------------------------------


def test_snake_cases_keys_deterministically() -> None:
    cases = {
        "itemId": "item_id",
        "HTTP-Status": "http_status",
        "HTTPStatus": "http_status",
        "durationMs": "duration_ms",
        "9lives": "f_9lives",
        "---": "field",
        "already_snake": "already_snake",
        "X-Amz-Signature": "x_amz_signature",
    }
    assert {k: r.to_snake_case(k) for k in cases} == cases


def test_redacts_sensitive_keys_secret_values_newlines_and_caps_length() -> None:
    line = _line(
        authorization="Bearer abcdefghijklmnop",
        cookie="sid=1",
        apiToken="plain-looking",
        note="retry with Bearer abcdefghijklmnop please",
        multi="line one\nline two\r\nline three",
        big="x" * 5000,
        loginForm={"password": "hunter2"},
    )
    assert line["authorization"] == line["cookie"] == line["api_token"] == "[REDACTED]"
    assert line["note"] == "retry with [REDACTED] please"
    assert line["multi"] == "line one line two  line three"
    assert len(line["big"]) <= 1000 + len("...[truncated]")
    assert line["login_form"] == '{"password":"[REDACTED]"}'


def test_flattens_nested_values_to_one_line_json() -> None:
    line = _line(
        nested={"a": {"b": [1, 2]}},
        list=["x", "y"],
        when=dt.datetime(2026, 9, 30, 1, 2, 3, 456000, tzinfo=dt.UTC),
        nothing=None,
        callback=lambda: None,
        ratio=float("nan"),
        ceiling=float("inf"),
        floor=float("-inf"),
        deep={"a": {"b": {"c": {"d": {"e": 1}}}}},
    )
    assert line["nested"] == '{"a":{"b":[1,2]}}'
    assert line["list"] == '["x","y"]'
    assert line["when"] == "2026-09-30T01:02:03.456Z"
    assert line["nothing"] is None
    assert line["callback"] == "[function]"
    assert (line["ratio"], line["ceiling"], line["floor"]) == ("NaN", "Infinity", "-Infinity")
    assert line["deep"] == '{"a":{"b":{"c":{"d":"[truncated:max-depth]"}}}}'


def test_keeps_100_entries_of_a_container_and_counts_the_rest() -> None:
    line = _line(many=list(range(150)), wide={f"k{i}": i for i in range(150)})
    assert json.loads(line["many"]) == [*range(99), "[truncated:51 more]"]
    wide = json.loads(line["wide"])
    assert len(wide) == 100 and wide["[truncated]"] == "51 more" and wide["k98"] == 98


def test_reads_at_most_1000_entries_per_value() -> None:
    shared = {f"k{i}": i for i in range(50)}
    out = r.print_value({f"g{i}": shared for i in range(40)})
    assert isinstance(out, dict)
    assert out["g18"] == shared  # 40 + 19 x 50 = 990 entries read
    assert out["g19"] == out["g39"] == "[truncated:max-entries]"


@pytest.mark.parametrize(
    ("label", "value", "expected"),
    [
        (
            "a signed image URL",
            "https://img.store-cdn.example/a.jpg?X-Amz-Signature=SECRET&exp=1",
            "https://img.store-cdn.example/a.jpg",
        ),
        ("userinfo", "https://user:SECRET@h.example:8443/x", "https://h.example:8443/x"),
        ("userinfo holding an @", "https://u:SEC@RET@h.example/x", "https://h.example/x"),
        (
            "a connection string",
            "postgresql://app:SECRET@pg-spine-rw.data.svc:5432/spine?sslmode=require",
            "postgresql://pg-spine-rw.data.svc:5432/spine",
        ),
        ("an upper-case scheme", "HTTPS://X.EXAMPLE/A?SIG=SECRET", "HTTPS://X.EXAMPLE/A"),
        ("a fragment", "https://x.example/p#SECRET", "https://x.example/p"),
        ("a host with no path", "https://x.example?k=SECRET", "https://x.example"),
        ("an @ in the path", "https://x.example/u/@ross?k=SECRET", "https://x.example/u/@ross"),
        ("a path with a query", "/login?token=SECRET", "/login"),
        ("a path with a fragment", "/p#SECRET", "/p"),
        ("a padded path", "  /login?token=SECRET ", "/login"),
        ("a protocol-relative URL", "//cdn.example/i.jpg?sig=SECRET", "//cdn.example/i.jpg"),
        (
            "a URL inside text",
            "retry https://x.example/a?sig=SECRET in 5s",
            "retry https://x.example/a in 5s",
        ),
        ("a URL glued to a word", "src=https://x.example/a?sig=SECRET", "src=https://x.example/a"),
        (
            "two URLs",
            "from https://a.example/?k=SECRET to http://b.example/c#SECRET",
            "from https://a.example/ to http://b.example/c",
        ),
        (
            "a quoted URL",
            "open 'https://x.example/a?sig=SECRET' now",
            "open 'https://x.example/a' now",
        ),
        (
            "a query holding an apostrophe",
            "https://x.example/a/b?q=it's&sig=SECRET#f",
            "https://x.example/a/b",
        ),
        (
            "a double-quoted URL, keeping the closing quote",
            'open "https://x.example/a?q=say"hi"&sig=SECRET" now',
            'open "https://x.example/a" now',
        ),
        (
            "a URL in brackets, keeping the closing ones",
            "see (https://x.example/a?sig=SECRET).",
            "see (https://x.example/a).",
        ),
        (
            "a URL in JSON inside text",
            'body {"url":"https://x.example/a?sig=SECRET"} end',
            'body {"url":"https://x.example/a"} end',
        ),
        (
            "a host and path with no scheme",
            "img.store.example/a.jpg?X-Amz-Signature=SECRET",
            "img.store.example/a.jpg",
        ),
        ("a relative path", "api/v1/items?sig=SECRET", "api/v1/items"),
        ("a host and port", "localhost:3000/a?sig=SECRET", "localhost:3000/a"),
        (
            "a scheme with no slashes",
            "mailto:ross@x.example?subject=SECRET",
            "mailto:ross@x.example",
        ),
        ("a host with no path and no scheme", "cdn.example?sig=SECRET", "cdn.example"),
        ("an IP address and port", "10.0.0.5:8080?sig=SECRET", "10.0.0.5:8080"),
        ("a query past a line break", "https://h.example/a?x=1\nsig=SECRET", "https://h.example/a"),
        ("a bare query string", "?access_token=SECRET&state=1", ""),
        ("a bare fragment of pairs", "#access_token=SECRET", ""),
        ("a request line", "POST /login?sig=SECRET HTTP/1.1", "POST /login HTTP/1.1"),
        (
            "a request line inside text",
            "got GET /items?sig=SECRET 200 in 5 ms",
            "got GET /items 200 in 5 ms",
        ),
        ("a request target in brackets", "retry (PUT /items/1#SECRET).", "retry (PUT /items/1)."),
        ("a Windows-style path", "dir\\sub\\file.txt?sig=SECRET", "dir\\sub\\file.txt"),
        ("a single-label host with a query", "scraper?q=SECRET", "scraper"),
        ("an IPv6 host and port", "[::1]:8080?sig=SECRET", "[::1]:8080"),
        ("an internationalised host", "bücher.example?sig=SECRET", "bücher.example"),
        (
            "a colon-joined method and target",
            "Route GET:/nothere?sig=SECRET not found",
            "Route GET:/nothere not found",
        ),
        (
            "URLs closed by brackets and punctuation",
            "see <https://a.example/x?s=SECRET>, [https://b.example/y?s=SECRET]; "
            "`https://c.example/z?s=SECRET`: https://d.example/w?s=SECRET!",
            "see <https://a.example/x>, [https://b.example/y]; `https://c.example/z`: https://d.example/w!",
        ),
        *[
            (f"a {m} request line", f"{m} /x?sig=SECRET HTTP/1.1", f"{m} /x HTTP/1.1")
            for m in ("HEAD", "DELETE", "CONNECT", "OPTIONS", "TRACE", "PATCH")
        ],
    ],
)
def test_strips_url_queries_fragments_and_userinfo(label: str, value: str, expected: str) -> None:
    assert _line(note=value)["note"] == expected, label
    assert "SECRET" not in _raw(note=value)


@pytest.mark.parametrize(
    "value",
    [
        "https://X.example",
        "file:///app/dist/server.mjs:10:5",
        "why? because",
        "a?b",
        "/a b?c",
        "see /docs?page=2 for more",
        " 42 ",
        "ready?",
        "item#3",
        "see https://x.example/a.",
        "#3 in the queue",
        "Q: why?",
        "get /items?page=2 later",
        "GET /items HTTP/1.1",
        "the cookie: header was missing",
        "e.g.?",
        "#1 retry with x=2",
        "BUDGET /items?page=2 later",
        "POSTER? no",
        "why? x=1",
        "ok? see a?b=1",
        "scraper?ok",
        "see a // b?c",
        "GET: why?",
    ],
)
def test_leaves_text_that_is_not_a_url_alone(value: str) -> None:
    assert _line(note=value)["note"] == value


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("url", "login?access_token=SECRET", "login"),
        (
            "url",
            "https://store.example/search?q=hello world&sig=SECRET",
            "https://store.example/search",
        ),
        ("url", "https://u:SECRET@x.example/a", "https://x.example/a"),
        ("imageUrl", "img.example/a.jpg?sig=SECRET", "img.example/a.jpg"),
        ("uri", "items?page=2&sig=SECRET", "items"),
        ("href", "next #SECRET", "next"),
        ("redirectUri", "https://x.example/cb?code=SECRET", "https://x.example/cb"),
        ("link", "why? SECRET", "why"),
        ("path", "items?sig=SECRET", "items"),
        ("target", "orders?sig=SECRET", "orders"),
        ("endpoint", "orders#SECRET", "orders"),
        ("location", "next?sig=SECRET", "next"),
        ("referer", "page?sig=SECRET", "page"),
        ("redirect", "home?sig=SECRET", "home"),
        ("nextPageUrl", "page 2?cursor=SECRET", "page 2"),
        ("origin_url", "https://static.myfigurecollection.net/upload/a.jpg?v=SECRET", None),
    ],
)
def test_cuts_a_url_named_field_at_its_first_query_or_fragment(
    key: str, value: str, expected: str | None
) -> None:
    line = _line(**{key: value})
    printed = line[r.to_snake_case(key)]
    assert "SECRET" not in printed
    if expected is not None:
        assert printed == expected


def test_cuts_url_named_keys_at_any_depth_and_in_json_strings() -> None:
    line = _line(
        config={
            "baseURL": "http://h.example/api/",
            "url": "login?access_token=SECRET1",
            "method": "post",
        },
        page={"links": ["next page?cursor=SECRET2", ["prev#SECRET3"]], "title": "why? because"},
        body='{"redirect_uri":"cb?code=SECRET4"}',
        images=[{"src": "x", "href": "a.jpg?q=it's&sig=SECRET5"}],
        urls=["a b?SECRET6"],
        hrefs='["next?cursor=SECRET7"]',
        paging={"nextPageUrl": "page 2?cursor=SECRET8"},
    )
    assert line["config"] == '{"baseURL":"http://h.example/api/","url":"login","method":"post"}'
    assert line["page"] == '{"links":["next page",["prev"]],"title":"why? because"}'
    assert line["body"] == '{"redirect_uri":"cb"}'
    assert line["images"] == '[{"src":"x","href":"a.jpg"}]'
    assert line["urls"] == '["a b"]'
    assert line["hrefs"] == '["next"]'
    assert line["paging"] == '{"nextPageUrl":"page 2"}'
    assert "SECRET" not in json.dumps(line)


def test_masks_sensitive_header_lines_and_header_lists() -> None:
    line = _line(
        head="POST /login?sig=SECRET1 HTTP/1.1\r\nHost: h.example\r\nX-Api-Key: SECRET2\r\n"
        "Cookie: sid=SECRET3\r\nAuthorization: Basic U0VDUkVUNA==",
        note="Note: password rotation is due\nX-Request-Id: 42",
        lone="x\rCookie: sid=SECRET4",
        spaced="Set-Cookie : sid=SECRET5",
        odd="X_Token2: SECRET6",
        raw=[
            "Host",
            "h.example",
            "Set-Cookie",
            "sid=hunter1",
            "Authorization",
            "Basic aHVudGVyMg==",
        ],
        numbered=[7, "seven", "X-Api-Key", "hunter4"],
        notalist=["the token", "kept"],
    )
    assert line["head"] == (
        "POST /login HTTP/1.1  Host: h.example  X-Api-Key: [REDACTED]  "
        "Cookie: [REDACTED]  Authorization: [REDACTED]"
    )
    assert line["note"] == "Note: password rotation is due X-Request-Id: 42"
    assert line["lone"] == "x Cookie: [REDACTED]"
    assert line["spaced"] == "Set-Cookie : [REDACTED]"
    assert line["odd"] == "X_Token2: [REDACTED]"
    assert (
        line["raw"] == '["Host","h.example","Set-Cookie","[REDACTED]","Authorization","[REDACTED]"]'
    )
    assert line["numbered"] == '[7,"seven","X-Api-Key","[REDACTED]"]'
    assert line["notalist"] == '["the token","kept"]'
    assert "SECRET" not in json.dumps(line) and "hunter" not in json.dumps(line)


def test_header_line_in_the_message() -> None:
    assert json.loads(_raw("Proxy-Authorization:Basic U0VDUkVUNQ=="))["msg"] == (
        "Proxy-Authorization:[REDACTED]"
    )


def test_key_redacts_json_strings() -> None:
    inner = json.dumps({"inner": json.dumps({"password": "hunter2"})}, separators=(",", ":"))
    line = _line(
        body='{"password":"hunter2","site":"orzgk"}',
        nested=inner,
        pretty=json.dumps({"a": 1, "b": [1, 2]}, indent=2),
        list='[{"cookie":"sid=1"}]',
        broken='{"password":"hunter2"',
        prose='resp {"password":"hunter2"} done',
        padded='  {"password":"hunter2"}\n',
        bom='﻿{"password":"hunter2"}',
    )
    assert line["body"] == '{"password":"[REDACTED]","site":"orzgk"}'
    assert line["nested"] == json.dumps(
        {"inner": json.dumps({"password": "[REDACTED]"}, separators=(",", ":"))},
        separators=(",", ":"),
    )
    assert line["pretty"] == '{"a":1,"b":[1,2]}'
    assert line["list"] == '[{"cookie":"[REDACTED]"}]'
    assert line["broken"] == '{"password":"hunter2"'  # not a whole document: the documented limit
    assert line["prose"] == 'resp {"password":"hunter2"} done'
    assert line["padded"] == line["bom"] == '{"password":"[REDACTED]"}'


def test_stops_parsing_json_in_strings_four_levels_down() -> None:
    doc: Any = {"password": "hunter2"}
    for _ in range(4):
        doc = {"s": json.dumps(doc)}
    text = _line(body=json.dumps(doc))["body"]
    for _ in range(4):
        text = json.loads(text)["s"]
    assert text == "[truncated:max-depth]"


def test_masks_sensitive_values_of_a_form_encoded_body() -> None:
    form = "grant_type=password&username=u&password=hunter5&client_secret=hunter6"
    line = _line(
        form=form,
        config={"url": "/oauth/token", "data": form},
        body="api%5Fkey=hunter7&session%2Did=hunter8&a=1",
        odd="%E0%A4%A=1&token%=hunter9&=x&flag",
        plus="x=1&api+key=visible",
        list="a=1&b=2",
        prose="retry with password=visible",
        flag="a=1&passwordless",
        padded=" a=1&b=2 ",
    )
    masked = "grant_type=password&username=u&password=[REDACTED]&client_secret=[REDACTED]"
    assert line["form"] == masked
    assert line["config"] == json.dumps(
        {"url": "/oauth/token", "data": masked}, separators=(",", ":")
    )
    assert line["body"] == "api%5Fkey=[REDACTED]&session%2Did=[REDACTED]&a=1"
    assert line["odd"] == "%E0%A4%A=1&token%=[REDACTED]&=x&flag"
    assert line["plus"] == "x=1&api+key=visible"  # 'api key' names no secret by default
    assert (line["list"], line["prose"], line["flag"], line["padded"]) == (
        "a=1&b=2",
        "retry with password=visible",
        "a=1&passwordless",
        " a=1&b=2 ",
    )
    assert "hunter" not in json.dumps(line)


def test_reads_form_keys_as_a_server_does() -> None:
    line = _line(
        escaped="%74oken=hunter13&a=1",
        runs="%61pi%5fkey=hunter17&a=1",
        invalid="%E0=1&token=hunter14",
        gaps="a=1&&token=hunter15&",
    )
    assert line == {
        **line,
        "escaped": "%74oken=[REDACTED]&a=1",
        "runs": "%61pi%5fkey=[REDACTED]&a=1",
        "invalid": "%E0=1&token=[REDACTED]",
        "gaps": "a=1&&token=[REDACTED]&",
    }


def test_sig_signature_and_code_keys_are_sensitive() -> None:
    line = _line(
        oauth="code=hunter50&state=s&code_verifier=hunter51",
        signed="X-Amz-Date=1&X-Amz-Signature=hunter52&sig=hunter53&signature=hunter54",
        words="urlSig=hunter55&device_code=hunter56&codeVerifier=hunter57&%73ig=hunter58",
        unsigned="signal=1&design=2&sigma=3&barcode=4&codec=5&nosig=6",
        q={"sig": "hunter61", "page": "2"},
        params={"X-Amz-Signature": "hunter62", "codeVerifier": "hunter63"},
        signature="hunter64",
        headers="Signature: sig1=:hunter65:",
        query={"code": "oauth-code"},
        repeated={"code": ["hunter71", "hunter72"]},
        grant={"auth_code": "a", "deviceCode": "b", "user_code": "C", "Code": "d", "code[0]": "e"},
        authCode="hunter81",
        prose="retry with code=visible&sig=visible",
    )
    assert line["oauth"] == "code=[REDACTED]&state=s&code_verifier=[REDACTED]"
    assert (
        line["signed"]
        == "X-Amz-Date=1&X-Amz-Signature=[REDACTED]&sig=[REDACTED]&signature=[REDACTED]"
    )
    assert line["words"] == (
        "urlSig=[REDACTED]&device_code=[REDACTED]&codeVerifier=[REDACTED]&%73ig=[REDACTED]"
    )
    assert line["unsigned"] == "signal=1&design=2&sigma=3&barcode=4&codec=5&nosig=6"
    assert line["q"] == '{"sig":"[REDACTED]","page":"2"}'
    assert line["params"] == '{"X-Amz-Signature":"[REDACTED]","codeVerifier":"[REDACTED]"}'
    assert line["signature"] == "[REDACTED]"
    assert line["headers"] == "Signature: [REDACTED]"
    assert line["query"] == '{"code":"[REDACTED]"}'
    assert line["repeated"] == '{"code":"[REDACTED]"}'
    assert set(json.loads(line["grant"]).values()) == {"[REDACTED]"}
    assert line["auth_code"] == "[REDACTED]"
    assert line["prose"] == "retry with code=visible&sig=visible"
    assert "hunter" not in json.dumps(line)


def test_code_as_another_word_or_a_non_string_prints() -> None:
    line = _line(
        grpc={"code": 14, "details": "shown-0"},
        object={"code": {"reason": "shown-1"}},
        words={"jan_code": "4580590123456", "statusCode": 404, "error_code": "card_declined"},
    )
    assert line["grpc"] == '{"code":14,"details":"shown-0"}'
    assert line["object"] == '{"code":{"reason":"shown-1"}}'
    assert json.loads(line["words"])["jan_code"] == "4580590123456"


def test_a_value_that_cannot_be_read_prints_as_unserializable() -> None:
    class Hostile:
        def __iter__(self) -> Any:
            raise RuntimeError("no")

    class Exploding(dict):  # type: ignore[type-arg]
        def items(self) -> Any:
            raise RuntimeError("no")

    assert _line(x=Exploding(a=1))["x"] == "[unserializable]"
    assert _line(x=Hostile())["x"] == "[Hostile]"


def test_stack_keeps_the_raising_frame() -> None:
    try:
        raise ValueError(f"boom {GHP} https://h.example/a?sig=SECRET")
    except ValueError as exc:
        tail = r.stack_tail(exc, limit=200)
    assert tail.startswith("...[truncated]") or "Traceback" in tail
    assert tail.endswith("ValueError: boom [REDACTED] https://h.example/a")
    assert "\n" not in tail and "SECRET" not in tail


# --- the printer's other value kinds (log.ts printValue: errors, binary, functions, cycles) ------


def test_a_printed_structure_holds_errors_bytes_functions_and_cycles_as_log_ts_prints_them() -> (
    None
):
    import enum
    import uuid
    from decimal import Decimal
    from pathlib import PurePosixPath

    class Colour(enum.Enum):
        RED = "red"

    try:
        raise ValueError("boom: Bearer abcdefghijkl")
    except ValueError as exc:
        raised = exc
    loop: dict[str, Any] = {"name": "x"}
    loop["self"] = loop
    printed = r.print_value(
        {
            "plain": ValueError("plain"),
            "raised": raised,
            "blob": b"\x89PNG",
            "fn": len,
            "loop": loop,
            "id": uuid.UUID(int=1),
            "amount": Decimal("1.50"),
            "where": PurePosixPath("/srv/a"),
            "colour": Colour.RED,
        }
    )
    assert printed["plain"] == {"name": "ValueError", "message": "plain"}
    assert printed["raised"]["message"] == "boom: [REDACTED]"
    assert "abcdefghijkl" not in printed["raised"]["stack"]
    assert printed["raised"]["stack"].splitlines()[0] == "Traceback (most recent call last):"
    assert (printed["blob"], printed["fn"], printed["loop"]["self"]) == (
        "[binary]",
        "[function]",
        "[circular]",
    )
    assert (printed["id"], printed["amount"], printed["where"], printed["colour"]) == (
        "00000000-0000-0000-0000-000000000001",
        "1.50",
        "/srv/a",
        "red",
    )


def test_a_scalar_prints_as_itself() -> None:
    assert [r.print_value(v) for v in (None, True, 3, 1.5)] == [None, True, 3, 1.5]


def test_past_the_total_budget_lists_and_json_text_print_as_the_marker() -> None:
    value: dict[str, Any] = {f"l{i}": list(range(100)) for i in range(10)}
    value["z"] = '{"a": 1}'
    printed = r.print_value(value)
    assert printed["l8"] == list(range(100))
    assert printed["l9"] == r.TRUNCATED_ENTRIES  # 11 + 10 x 100 entries > 1000
    assert printed["z"] == r.TRUNCATED_ENTRIES


def test_json_text_with_a_non_json_constant_is_plain_text() -> None:
    assert r.redact_text('{"a": NaN, "token": "t"}') == '{"a": NaN, "token": "t"}'


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "null"),
        (ValueError(""), "Error (no message)"),
        (ValueError("bad\nline"), "bad line"),
        (object, "<class 'object'>"),  # no JSON form: its text (Python only; JS has no such value)
    ],
)
def test_sanitize_log_value_prints_one_line(value: Any, expected: str) -> None:
    assert r.sanitize_log_value(value) == expected
