"""Util functions for XML export."""

from __future__ import annotations

import bz2
import re
import typing as t
import xml.etree.ElementTree as etree  # noqa: N813
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from shutil import copyfileobj
from typing import Any

from sparv.api import SparvErrorMessage, get_logger, util
from sparv.api.classes import Annotation, Corpus, ExportInput, OutputMarker

logger = get_logger(__name__)

INDENTATION = "  "


def make_pretty_xml(
    span_positions: list[tuple],
    annotation_dict: dict[str, dict[str, list[str]]],
    export_names: dict[str, str],
    token_name: str,
    word_annotation: list[str],
    fileid: str,
    include_empty_attributes: bool,
    *,
    stream: t.TextIO,
    sparv_namespace: str | None = None,
    xml_namespaces: dict | None = None,
) -> None:
    """Create a pretty XML tree from span_positions and write it to the given stream.

    Used by pretty and sentence_scrambled.

    Args:
        span_positions: List of tuples with span positions.
        annotation_dict: Dictionary with annotations.
        export_names: Dictionary with export names.
        token_name: Name of the token element.
        word_annotation: Word annotation.
        fileid: File ID.
        include_empty_attributes: Whether to include empty attributes.
        stream: Text stream to write the XML to.
        sparv_namespace: Optional namespace for Sparv attributes.
        xml_namespaces: Optional dictionary with XML namespaces.

    Raises:
        SparvErrorMessage: If the root tag is missing.
    """
    # Root tag sanity check
    if not valid_root(span_positions[0], span_positions[-1]):
        raise SparvErrorMessage(
            "Root tag is missing! If you have manually specified which elements to include, "
            "make sure to include an element that encloses all other included elements and "
            "text content."
        )

    # Create root node
    root_span = span_positions[0][2]
    root_span.set_node()
    add_attrs(root_span.node, root_span.name, annotation_dict, export_names, 0, include_empty_attributes)
    node_stack = [root_span]

    last_start_pos = -1
    last_end_pos = -1
    current_token_text = None
    last_node = None
    inside_token = False

    if xml_namespaces:
        register_namespaces(xml_namespaces)

    def handle_subtoken_text(
        position: int, last_start_position: int, last_end_position: int, node: etree.Element, token_text: str
    ) -> str:
        """Handle text for subtoken elements.

        Args:
            position: Current position.
            last_start_position: Last start position.
            last_end_position: Last end position.
            node: Current node.
            token_text: Current token text.

        Returns:
            Updated token text.
        """
        if last_start_position < last_end_position < position:
            node.tail = token_text[: position - last_end_position]
            token_text = token_text[position - last_end_position :]
        elif position > last_start_position:
            node.text = token_text[: position - last_start_position]
            token_text = token_text[position - last_start_position :]
        return token_text

    # Go through span_positions and build xml tree
    for _pos, instruction, span in span_positions[1:]:
        # Handle headers
        if span.is_header:
            if instruction == "open":
                header = annotation_dict[span.name][util.constants.HEADER_CONTENTS][span.index]
                # Replace any leading tabs with spaces
                header = re.sub(r"^\t+", lambda m: INDENTATION * len(m.group()), header, flags=re.MULTILINE)
                header_xml = etree.fromstring(header)
                header_xml.tag = span.export  # Rename element if needed
                node_stack[-1].node.append(header_xml)
            continue

        # Create child node under the top stack node
        if instruction == "open":
            span.set_node(parent_node=node_stack[-1].node)
            node_stack.append(span)
            add_attrs(span.node, span.name, annotation_dict, export_names, span.index, include_empty_attributes)
            if span.overlap_id:
                if sparv_namespace:
                    span.node.set(f"{sparv_namespace}.{util.constants.OVERLAP_ATTR}", f"{fileid}-{span.overlap_id}")
                else:
                    span.node.set(
                        f"{util.constants.SPARV_DEFAULT_NAMESPACE}.{util.constants.OVERLAP_ATTR}",
                        f"{fileid}-{span.overlap_id}",
                    )

            # Add text if this node is a token
            if span.name == token_name:
                inside_token = True
                # Save text until later
                last_start_pos = span.start
                current_token_text = word_annotation[span.index]

            if inside_token and current_token_text:
                if last_node is not None:
                    current_token_text = handle_subtoken_text(
                        span.start, last_start_pos, last_end_pos, last_node, current_token_text
                    )
                last_start_pos = span.start
                last_node = span.node

        # Close node
        else:
            if inside_token and current_token_text:
                if last_node is not None:
                    current_token_text = handle_subtoken_text(
                        span.end, last_start_pos, last_end_pos, last_node, current_token_text
                    )
                last_end_pos = span.end
                last_node = span.node
            if span.name == token_name:
                inside_token = False

            # Make sure closing node == top stack node
            assert span == node_stack[-1], f"Overlapping elements found. Expected {span} but found {node_stack[-1]}"
            # Pop stack and move on to next span
            node_stack.pop()

    # Pretty formatting of XML tree
    etree.indent(root_span.node, space=INDENTATION)

    # We use write() instead of tostring() here to be able to get an XML declaration
    etree.ElementTree(root_span.node).write(stream, encoding="unicode", method="xml", xml_declaration=True)


def write_pretty_xml(
    out_path: Path,
    annotation_list: list,
    export_names: dict[str, str],
    header_annotations: list,
    source_file: str,
    token_name: str,
    word: Annotation,
    fileid: str,
    include_empty_attributes: bool,
    sparv_namespace: str | None = None,
    xml_namespaces: dict | None = None,
) -> None:
    """Write pretty formatted XML to `out_path`, streaming the annotations instead of building the tree in memory.

    The output is identical to writing the result of `gather_annotations()` + `make_pretty_xml()` to a file followed
    by a newline. Whenever identical output can't be guaranteed while streaming, `StreamingNotPossibleError` is raised
    before anything is written to `out_path`, and the caller should use the non-streaming functions instead.

    Args:
        out_path: Path of the XML file to write.
        annotation_list: Annotations to include (as for `gather_annotations()`).
        export_names: Dictionary with export names.
        header_annotations: Header annotations.
        source_file: Source filename.
        token_name: Name of the token element.
        word: Annotation with the token strings.
        fileid: File ID.
        include_empty_attributes: Whether to include empty attributes.
        sparv_namespace: Optional namespace for Sparv attributes.
        xml_namespaces: Optional dictionary with XML namespaces.

    Raises:
        StreamingNotPossibleError: If the XML can't be streamed with a guaranteed identical result.
    """
    stream = util.export.SpanStream(
        annotation_list, export_names, header_annotations, source_file=source_file, split_overlaps=True
    )
    if xml_namespaces:
        register_namespaces(xml_namespaces)
    qnames, namespaces = _streaming_qnames(stream, export_names, include_empty_attributes, sparv_namespace)

    tmp_path = out_path.with_name(f"{out_path.name}.tmp")
    try:
        try:
            events = stream.events()
            with tmp_path.open(mode="w", encoding="utf-8") as outfile:
                _write_pretty_xml_events(
                    outfile.write, events, stream.annotation_dict(), export_names, token_name,
                    util.export.SequentialValues(word.read()), fileid, include_empty_attributes, sparv_namespace,
                    qnames, namespaces,
                )  # fmt: skip
        except util.export.LateOverlapError:
            # Overlap IDs were found after the affected elements were written; get all IDs first and start over
            overlap_ids = stream.record_overlap_ids()
            events = stream.events(overlap_ids=overlap_ids)
            with tmp_path.open(mode="w", encoding="utf-8") as outfile:
                _write_pretty_xml_events(
                    outfile.write, events, stream.annotation_dict(), export_names, token_name,
                    util.export.SequentialValues(word.read()), fileid, include_empty_attributes, sparv_namespace,
                    qnames, namespaces,
                )  # fmt: skip
        tmp_path.replace(out_path)
    finally:
        tmp_path.unlink(missing_ok=True)


class _QNames(dict):  # noqa: FURB189 (only item access is needed, which dict supports directly)
    """Qualified names for serialization; names without namespace are added on first use."""

    def __missing__(self, key: Any) -> str:
        if isinstance(key, str) and key[:1] != "{":
            self[key] = key
            return key
        raise util.export.StreamingNotPossibleError(f"Unexpected namespaced name: {key!r}")


def _streaming_qnames(
    stream: util.export.SpanStream,
    export_names: dict[str, str],
    include_empty_attributes: bool,
    sparv_namespace: str | None,
) -> tuple[dict, dict]:
    """Calculate the qualified names and namespace declarations that ElementTree would use for the whole document.

    ElementTree declares all namespaces used anywhere in the document on the root element, so they need to be known
    before anything is written. Only names that may contain a namespace (i.e. start with '{') need to be checked.

    Args:
        stream: The span stream.
        export_names: Dictionary with export names.
        include_empty_attributes: Whether to include empty attributes.
        sparv_namespace: Optional namespace for Sparv attributes.

    Returns:
        A tuple of qnames and namespaces, as returned by ElementTree's `_namespaces()`.

    Raises:
        StreamingNotPossibleError: If the namespaces can't be determined in advance.
    """
    overlap_attr = (
        f"{sparv_namespace}.{util.constants.OVERLAP_ATTR}"
        if sparv_namespace
        else f"{util.constants.SPARV_DEFAULT_NAMESPACE}.{util.constants.OVERLAP_ATTR}"
    )
    if overlap_attr[:1] == "{":
        raise util.export.StreamingNotPossibleError("Namespaced overlap attribute")

    used_tags = set()
    used_attrs = set()
    sources = stream.attribute_annotations()
    for name, is_header, count in stream.non_empty_elements():
        used_tags.add(export_names.get(name, name))
        if is_header:
            for header_contents in sources[name][util.constants.HEADER_CONTENTS][:count]:
                header = re.sub(r"^\t+", lambda m: INDENTATION * len(m.group()), header_contents, flags=re.MULTILINE)
                header_xml = etree.fromstring(header)
                for i, elem in enumerate(header_xml.iter()):
                    if i:  # The root element is renamed
                        used_tags.add(elem.tag)
                    used_attrs.update(elem.keys())
        else:
            for attrib_name, source in sources[name].items():
                export_name = export_names.get(f"{name}:{attrib_name}", attrib_name)
                if export_name[:1] != "{":
                    continue  # Names without namespace don't affect the namespace declarations
                if include_empty_attributes or any(source.read()):
                    used_attrs.add(export_name)

    dummy = etree.Element("_")
    for tag in used_tags:
        etree.SubElement(dummy, tag)
    for attr in used_attrs:
        dummy.set(attr, "")
    qnames, namespaces = etree._namespaces(dummy)
    if any(uri not in etree._namespace_map for uri in namespaces):
        # Unregistered namespaces get prefixes based on the order they're found in the document
        raise util.export.StreamingNotPossibleError("Unregistered XML namespace")
    return _QNames(qnames), namespaces


class _Frame:
    """An element outside of tokens that has been opened but not closed."""

    __slots__ = ("level", "node", "prev_tail", "started")

    def __init__(self, node: etree.Element, level: int) -> None:
        self.node = node
        self.level = level
        self.started = False  # Whether the start tag has been written (happens when the first child is added)
        self.prev_tail = None  # Tail of the previous child, written when we know if it's the last child


_SENTINEL = "\x00"


def _write_pretty_xml_events(
    write: Callable[[str], Any],
    span_positions: Iterator[tuple],
    annotation_dict: dict,
    export_names: dict[str, str],
    token_name: str,
    word_annotation: Any,
    fileid: str,
    include_empty_attributes: bool,
    sparv_namespace: str | None,
    qnames: dict,
    namespaces: dict,
) -> None:
    """Write XML from streamed span positions, with the same result as `make_pretty_xml()` + newline.

    This follows `make_pretty_xml()` step by step. Elements outside tokens are written as soon as possible, while
    tokens (with any sub-token elements) are built as small trees, indented with `etree.indent()` and serialized with
    ElementTree's own serializer when the token is closed. Tokens are the only place where `make_pretty_xml()`
    modifies text after an element is created.

    Raises:
        StreamingNotPossibleError: If identical output can't be guaranteed.
    """
    serialize = etree._serialize_xml
    escape_cdata = etree._escape_cdata
    indentations = ["\n"]

    def indentation(level: int) -> str:
        while len(indentations) <= level:
            indentations.append(indentations[-1] + INDENTATION)
        return indentations[level]

    def begin_child(frame: _Frame) -> None:
        """Prepare for a new child of `frame`: write the start tag of `frame` or the tail of the previous child."""
        if not frame.started:
            text = frame.node.text
            if not text or not text.strip():
                text = indentation(frame.level + 1)
            # Serialize a childless copy with a sentinel text and keep everything before the sentinel, to get the start
            # tag (with namespace declarations for the root) exactly as ElementTree writes it
            template = etree.Element(frame.node.tag, frame.node.attrib)
            template.text = text + _SENTINEL
            parts = []
            serialize(
                parts.append, template, qnames, namespaces if frame.level == 0 else None, short_empty_elements=True
            )
            write("".join(parts).rpartition(_SENTINEL)[0])
            frame.started = True
        else:
            tail = frame.prev_tail
            write(escape_cdata(tail if tail and tail.strip() else indentation(frame.level + 1)))

    def write_subtree(node: etree.Element, parent: _Frame) -> None:
        """Indent and write a complete subtree (token or header) whose parent is `parent`."""
        etree.indent(node, space=INDENTATION, level=parent.level + 1)
        tail = node.tail
        node.tail = None
        serialize(write, node, qnames, None, short_empty_elements=True)
        parent.prev_tail = tail

    def end_frame(frame: _Frame) -> None:
        """Write the end of an element outside tokens."""
        if frame.started:
            tail = frame.prev_tail
            write(escape_cdata(tail if tail and tail.strip() else indentation(frame.level)))
            write("</" + qnames[frame.node.tag] + ">")
        else:
            tail = frame.node.tail
            frame.node.tail = None
            serialize(write, frame.node, qnames, namespaces if frame.level == 0 else None, short_empty_elements=True)
            frame.node.tail = tail

    def handle_subtoken_text(
        position: int, last_start_position: int, last_end_position: int, node: etree.Element, token_text: str
    ) -> str:
        """Handle text for subtoken elements (identical to the function in `make_pretty_xml()`).

        Returns:
            The remaining token text.
        """
        if last_start_position < last_end_position < position:
            node.tail = token_text[: position - last_end_position]
            token_text = token_text[position - last_end_position :]
        elif position > last_start_position:
            node.text = token_text[: position - last_start_position]
            token_text = token_text[position - last_start_position :]
        return token_text

    first_item = next(span_positions, None)
    if first_item is None:
        raise util.export.StreamingNotPossibleError("No spans")
    root_span = first_item[2]
    if first_item[1] != "open" or root_span.is_header or root_span.name == token_name:
        raise util.export.StreamingNotPossibleError("Unexpected root element")

    # Create root node
    root_span.set_node()
    add_attrs(root_span.node, root_span.name, annotation_dict, export_names, 0, include_empty_attributes)
    write("<?xml version='1.0' encoding='utf-8'?>\n")
    frames = [_Frame(root_span.node, 0)]
    node_stack = [root_span]
    token_root = None  # The open token whose subtree is being built in memory

    last_start_pos = -1
    last_end_pos = -1
    current_token_text = None
    last_node = None
    inside_token = False
    last_item = first_item

    for item in span_positions:
        last_item = item
        _pos, instruction, span = item
        # Handle headers
        if span.is_header:
            if instruction == "open":
                header = annotation_dict[span.name][util.constants.HEADER_CONTENTS][span.index]
                # Replace any leading tabs with spaces
                header = re.sub(r"^\t+", lambda m: INDENTATION * len(m.group()), header, flags=re.MULTILINE)
                header_xml = etree.fromstring(header)
                header_xml.tag = span.export  # Rename element if needed
                if token_root is not None:
                    node_stack[-1].node.append(header_xml)
                else:
                    begin_child(frames[-1])
                    write_subtree(header_xml, frames[-1])
            continue

        if instruction == "open":
            if token_root is not None:
                span.set_node(parent_node=node_stack[-1].node)
            else:
                begin_child(frames[-1])
                span.set_node()
            node_stack.append(span)
            add_attrs(span.node, span.name, annotation_dict, export_names, span.index, include_empty_attributes)
            if span.overlap_id:
                if sparv_namespace:
                    span.node.set(f"{sparv_namespace}.{util.constants.OVERLAP_ATTR}", f"{fileid}-{span.overlap_id}")
                else:
                    span.node.set(
                        f"{util.constants.SPARV_DEFAULT_NAMESPACE}.{util.constants.OVERLAP_ATTR}",
                        f"{fileid}-{span.overlap_id}",
                    )

            # Add text if this node is a token
            if span.name == token_name:
                inside_token = True
                # Save text until later
                last_start_pos = span.start
                current_token_text = word_annotation[span.index]

            if inside_token and current_token_text:
                if last_node is not None:
                    current_token_text = handle_subtoken_text(
                        span.start, last_start_pos, last_end_pos, last_node, current_token_text
                    )
                last_start_pos = span.start
                last_node = span.node

            if token_root is None:
                if span.name == token_name:
                    token_root = span
                else:
                    frames.append(_Frame(span.node, len(frames)))

        # Close node
        else:
            if inside_token and current_token_text:
                if last_node is not None:
                    current_token_text = handle_subtoken_text(
                        span.end, last_start_pos, last_end_pos, last_node, current_token_text
                    )
                last_end_pos = span.end
                last_node = span.node
            if span.name == token_name:
                inside_token = False

            # Make sure closing node == top stack node
            if span is not node_stack[-1]:
                raise util.export.StreamingNotPossibleError("Overlapping elements")
            node_stack.pop()

            if token_root is not None:
                if span is token_root:
                    token_root = None
                    write_subtree(span.node, frames[-1])
            else:
                frame = frames.pop()
                end_frame(frame)
                if frames:
                    frames[-1].prev_tail = frame.node.tail

    # Root tag sanity check (done before writing in make_pretty_xml())
    if not valid_root(first_item, last_item) or frames or token_root is not None:
        raise util.export.StreamingNotPossibleError("Invalid root")

    # Newline added by print() in the non-streaming export
    write("\n")


def valid_root(first_item: tuple, last_item: tuple, true_root: bool = False) -> bool:
    """Check the validity of the root tag.

    Checks that the first item is an opening tag and the last item is a closing tag with the same name and index. If
    `true_root` is `True`, it also checks that the first item's index is 0, meaning that it is the real root tag.

    Args:
        first_item: First item from the list of spans.
        last_item: Last item in the list of spans.
        true_root: Whether to check for a true root tag.

    Returns:
        `True` if the root tag is valid, `False` otherwise.
    """
    return (
        first_item[1] == "open"
        and last_item[1] == "close"
        and first_item[2].name == last_item[2].name
        and first_item[2].index == last_item[2].index
        and (not true_root or (first_item[0] == 0))
    )


def register_namespaces(xml_namespaces: dict) -> None:
    """Register all namespace prefixes."""
    for prefix, uri in xml_namespaces.items():
        etree.register_namespace(prefix, uri)


def add_attrs(
    node: etree.Element,
    annotation: str,
    annotation_dict: dict[str, dict[str, list[str]]],
    export_names: dict[str, str],
    index: int,
    include_empty_attributes: bool,
) -> None:
    """Add attributes from annotation_dict to node.

    Args:
        node: XML node to add attributes to.
        annotation: Annotation name.
        annotation_dict: Dictionary with attribute annotations and values.
        export_names: Dictionary with export names.
        index: Index of the annotation.
        include_empty_attributes: Whether to include empty attributes.
    """
    for attrib_name, attrib_values in annotation_dict[annotation].items():
        export_name = export_names.get(f"{annotation}:{attrib_name}", attrib_name)
        if attrib_values[index] or include_empty_attributes:
            node.set(export_name, attrib_values[index])


def replace_invalid_chars_in_names(export_names: dict[str, str]) -> None:
    """Replace invalid characters with underscore in export names.

    Args:
        export_names: Dictionary with export names.
    """
    # https://www.w3.org/TR/REC-xml/#NT-NameStartChar
    start_chars = [
        ":",
        "A-Z",
        "_",
        "a-z",
        "\xc0-\xd6",
        "\xd8-\xf6",
        "\xf8-\u02ff",
        "\u0370-\u037d",
        "\u037f-\u1fff",
        "\u200c-\u200d",
        "\u2070-\u218f",
        "\u2c00-\u2fef",
        "\u3001-\ud7ff",
        "\uf900-\ufdcf",
        "\ufdf0-\ufffd",
        "\u10000-\uefffF",
    ]
    chars = ["-", ".", "0-9", "\xb7", "\u0300-\u036f", "\u203f-\u2040"]

    name_start_char = re.compile(r"[^{}]".format("".join(start_chars)))
    name_char = re.compile(r"[^{}{}]".format("".join(chars), "".join(start_chars)))
    namespace_split = re.compile(r"^({[^}]+})?(.+)")

    for n, n2 in export_names.items():
        assert n and n2, f"Annotation and export name must be non-empty strings. Found '{n}' and '{n2}'."  # noqa: PT018
        namespace, name = namespace_split.match(n2).groups()  # type: ignore
        original_name = name

        if name_start_char.match(name[0]):
            name = name_start_char.sub("_", name[0]) + name[1:]

        if name_char.search(name[1:]):
            name = name[0] + name_char.sub("_", name[1:])

        if name != original_name:
            logger.warning(
                "The name '%s' contains invalid characters and will be renamed to '%s'.", original_name, name
            )
            export_names[n] = (namespace or "") + name


def combine(
    corpus: str,
    out: str,
    source_files: Sequence[str],
    xml_input: ExportInput,
    version_info_file: str | None = None,
    compress: bool = False,
) -> None:
    """Combine XML files into one single XML file, optionally compressing it."""
    xml_files = [xml_input.file(file) for file in source_files]
    xml_files.sort()
    logger.progress(total=len(xml_files))
    opener = bz2.open if compress else open
    with opener(out, "wt", encoding="utf-8") as outf:
        print("<?xml version='1.0' encoding='UTF-8'?>", file=outf)
        if version_info_file:
            print("<!--", file=outf)
            with Path(version_info_file).open(encoding="utf-8") as vi:
                for line in vi:
                    print(line.strip(), file=outf)
            print("-->", file=outf)
        print('<corpus id="{}">'.format(corpus.replace("&", "&amp;").replace('"', "&quot;")), file=outf)
        for infile in xml_files:
            logger.info("Read: %s", infile)
            with Path(infile).open(encoding="utf-8") as inf:
                for n, line in enumerate(inf):
                    # Skip xml declaration
                    if n == 0 and line.startswith("<?xml"):
                        continue
                    # Indent line
                    outf.write(f"{INDENTATION}{line}")
                logger.progress()
        print("</corpus>", file=outf)
        logger.info("Exported: %s", out)


def compress(xmlfile: str, out: str) -> None:
    """Compress XML file using bzip2.

    Args:
        xmlfile: Path to source file.
        out: Path to target bz2 file.
    """
    with Path(xmlfile).open("rb") as infile, bz2.BZ2File(out, "wb") as outfile:
        copyfileobj(infile, outfile)


def install_compressed_xml(
    corpus: Corpus, bz2file: ExportInput, marker: OutputMarker, export_path: str, host: str | None
) -> None:
    """Install XML file.

    Args:
        corpus: Corpus name.
        bz2file: Path to the bz2 file.
        marker: Installation marker to write.
        export_path: Path to the export directory to install to.
        host: Optional host name to install to.
    """
    filename = corpus + ".xml.bz2"
    remote_file_path = Path(export_path) / filename
    util.install.install_path(bz2file, host, remote_file_path)
    marker.write()


def uninstall_compressed_xml(corpus: Corpus, marker: OutputMarker, export_path: str, host: str | None) -> None:
    """Uninstall XML file.

    Args:
        corpus: Corpus name.
        marker: Uninstallation marker to write.
        export_path: Path to the export directory to uninstall from.
        host: Optional host name to uninstall from.
    """
    remote_file_path = Path(export_path) / (corpus + ".xml.bz2")
    util.install.uninstall_path(remote_file_path, host)
    marker.write()
