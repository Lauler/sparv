"""Unit tests comparing the streaming pretty XML export with the in-memory export."""

from __future__ import annotations

import gzip
import pickle
import random
from pathlib import Path

import pytest

from sparv.api.classes import Annotation
from sparv.api.util import export
from sparv.modules.xml_export import xml_utils

NS_URI = "http://example.org/ns"
N_CASES = 300


def _put(workdir: Path, source_file: str, annotation: str, values: list) -> None:
    """Write an annotation file in Sparv's workdir format, without using Sparv's own writer."""
    elem, _, attr = annotation.partition(":")
    path = workdir / source_file / elem / (attr or "@span")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        for i in range(0, len(values), 1000):
            pickle.dump(values[i : i + 1000], f, protocol=pickle.HIGHEST_PROTOCOL)


def _make_case(rnd: random.Random, workdir: Path, source_file: str) -> tuple[list[str], list[str], dict]:
    """Create random annotations with tokens, nested and overlapping elements, empty elements, sub-token elements and
    headers.

    Returns:
        Annotation names, header annotation names and export options.
    """  # noqa: D205
    length = rnd.randint(5, 60)
    text = "".join(rnd.choice("ab cd.\n") for _ in range(length))
    spans = {"doc": [((0,), (length,))]}
    tokens, pos = [], 0
    while pos < length - 1:
        start = pos + rnd.randint(0, 2)
        end = min(length, start + rnd.randint(1, 4))
        if start >= end:
            break
        tokens.append(((start,), (end,)))
        pos = end
    spans["tok"] = tokens
    sentences, i = [], 0
    while i < len(tokens):
        j = min(len(tokens), i + rnd.randint(1, 4))
        sentences.append((tokens[i][0], tokens[j - 1][1]))
        i = j
    if rnd.random() < 0.8:  # noqa: PLR2004
        spans["sent"] = sentences
    for name in rnd.sample(["ne", "pb", "x"], rnd.randint(0, 3)):
        element_spans = []
        for _ in range(rnd.randint(1, 6)):
            r = rnd.random()
            if r < 0.25:  # noqa: PLR2004
                p = rnd.randint(0, length)
                element_spans.append(((p,), (p,)))
            elif r < 0.45 and sentences:  # noqa: PLR2004
                element_spans.append(rnd.choice(sentences))
            else:
                s = rnd.randint(0, length - 1)
                element_spans.append(((s,), (rnd.randint(s + 1, length),)))
        spans[name] = sorted(set(element_spans))
    if rnd.random() < 0.4 and tokens:  # noqa: PLR2004
        subs = []
        for t in rnd.sample(tokens, min(len(tokens), 3)):
            s, e = t[0][0], t[1][0]
            if e - s >= 2:  # noqa: PLR2004
                a = rnd.randint(s, e - 1)
                subs.append(((a,), (rnd.randint(a + 1, e),)))
        if subs:
            spans["sub"] = sorted(set(subs))
    for name, element_spans in spans.items():
        _put(workdir, source_file, name, element_spans)

    attributes = {"doc": ["title"], "tok": ["w", "pos"], "sent": ["id"], "ne": ["type"], "x": ["v"], "sub": ["s"]}
    values_choice = ["", "", "x", "a&b", 'q"<>', "ö”\n\t", " ", "lång text", "1|2|"]
    names = []
    for name in spans:
        names.append(name)
        for attr in attributes.get(name, []):
            if attr == "w":  # Token strings, sometimes shorter than the span (like when formatting chars are removed)
                values = [text[s[0] : e[0]] if rnd.random() < 0.8 else text[s[0] : e[0]][:-1] for s, e in spans[name]]  # noqa: PLR2004
            else:
                values = [rnd.choice(values_choice) for _ in spans[name]]
            _put(workdir, source_file, f"{name}:{attr}", values)
            names.append(f"{name}:{attr}")
    headers = []
    if rnd.random() < 0.3:  # noqa: PLR2004
        p = rnd.randint(0, length)
        _put(workdir, source_file, "hdr", [((p,), (p,))])
        _put(workdir, source_file, "hdr:contents", ['<h a="1">\n\t<t>Rubrik &amp; mer</t>\n\t<u/>\n</h>'])
        headers.append("hdr")
    options = {
        "include_empty": rnd.random() < 0.3,  # noqa: PLR2004
        "sparv_namespace": rnd.choice([None, "sparv"]),
        "namespaced": rnd.random() < 0.3,  # noqa: PLR2004
    }
    return names, headers, options


def _export(tmp_path: Path, source_file: str, names: list, headers: list, options: dict, streaming: bool) -> bytes:
    """Export pretty XML with the streaming or the in-memory implementation.

    Returns:
        The XML file contents.
    """
    out = tmp_path / f"{source_file}.{'streaming' if streaming else 'in_memory'}.xml"
    annotations = [Annotation(n, source_file=source_file) for n in names]
    header_annotations = [Annotation(h, source_file=source_file) for h in headers]
    export_names = {n: n.split(":")[-1] for n in names + headers}
    xml_namespaces = {}
    if options["namespaced"]:
        export_names["ne"] = f"{{{NS_URI}}}ne"
        export_names["tok:pos"] = f"{{{NS_URI}}}pos"
        xml_namespaces = {"ns": NS_URI}
    word = Annotation("tok:w", source_file=source_file)
    args = (options["include_empty"], options["sparv_namespace"], xml_namespaces)
    if streaming:
        xml_utils.write_pretty_xml(
            out, annotations, export_names, header_annotations, source_file, "tok", word, "f1", *args
        )
    else:
        span_positions, annotation_dict = export.gather_annotations(
            annotations, export_names, header_annotations, source_file=source_file, split_overlaps=True
        )
        xmlstr = xml_utils.make_pretty_xml(
            span_positions, annotation_dict, export_names, "tok", list(word.read()), "f1", *args
        )
        with out.open(mode="w", encoding="utf-8") as f:
            print(xmlstr, file=f)
    return out.read_bytes()


@pytest.mark.unit
@pytest.mark.noexternal
def test_etree_internals_work() -> None:
    """Test that the private ElementTree functions used for streaming work with this Python version."""
    assert xml_utils._etree_internals_work()  # noqa: SLF001


@pytest.mark.unit
@pytest.mark.noexternal
def test_streaming_pretty_xml_is_identical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that streaming pretty XML gives the same output as the in-memory export for random documents."""
    monkeypatch.chdir(tmp_path)
    workdir = tmp_path / "sparv-workdir"
    streamed = 0
    for case in range(N_CASES):
        source_file = f"case{case}"
        names, headers, options = _make_case(random.Random(case), workdir, source_file)
        expected = _export(tmp_path, source_file, names, headers, options, streaming=False)
        try:
            result = _export(tmp_path, source_file, names, headers, options, streaming=True)
        except export.StreamingNotPossibleError:
            continue  # The in-memory export is used instead
        assert result == expected, f"Output differs for case {case}"
        streamed += 1
    # Streaming is only expected to be impossible in rare cases (inconsistent element ordering)
    assert streamed >= 0.9 * N_CASES
