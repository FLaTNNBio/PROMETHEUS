"""Small read-only OOXML reader for aggregate EMS workbooks.

The source files are simple single-sheet ``.xlsx`` exports.  Reading their cell
values directly keeps the application case dependency-light and, importantly,
never rewrites the original healthcare-administration files.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile
from xml.etree import ElementTree


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CELL_REFERENCE = re.compile(r"^([A-Z]+)([1-9][0-9]*)$")


def _column_index(reference: str) -> tuple[int, int]:
    match = CELL_REFERENCE.fullmatch(reference)
    if match is None:
        raise ValueError(f"Invalid OOXML cell reference: {reference!r}")
    letters, row_text = match.groups()
    column = 0
    for letter in letters:
        column = column * 26 + ord(letter) - ord("A") + 1
    return int(row_text) - 1, column - 1


def _shared_strings(archive: ZipFile) -> list[str]:
    name = "xl/sharedStrings.xml"
    if name not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read(name))
    return [
        "".join(node.text or "" for node in item.findall(f".//{{{MAIN_NS}}}t"))
        for item in root.findall(f"{{{MAIN_NS}}}si")
    ]


def _sheet_targets(archive: ZipFile) -> list[tuple[str, str]]:
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    relationships = ElementTree.fromstring(
        archive.read("xl/_rels/workbook.xml.rels")
    )
    targets = {
        item.attrib["Id"]: item.attrib["Target"]
        for item in relationships.findall(f"{{{PACKAGE_REL_NS}}}Relationship")
    }
    sheets = []
    for item in workbook.findall(f".//{{{MAIN_NS}}}sheet"):
        relationship_id = item.attrib[f"{{{REL_NS}}}id"]
        target = PurePosixPath("xl") / targets[relationship_id]
        normalized = str(PurePosixPath(*(
            part for part in target.parts if part not in ("", ".")
        )))
        if ".." in PurePosixPath(normalized).parts:
            raise ValueError(f"Unsafe OOXML worksheet target: {normalized!r}")
        sheets.append((item.attrib["name"], normalized))
    if not sheets:
        raise ValueError("Workbook contains no worksheets")
    return sheets


def _cell_value(
    cell: ElementTree.Element,
    shared_strings: list[str],
) -> Any:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(
            node.text or "" for node in cell.findall(f".//{{{MAIN_NS}}}t")
        )
    value_node = cell.find(f"{{{MAIN_NS}}}v")
    if value_node is None or value_node.text is None:
        return None
    raw = value_node.text
    if cell_type == "s":
        index = int(raw)
        if not 0 <= index < len(shared_strings):
            raise ValueError(f"Invalid shared-string index: {index}")
        return shared_strings[index]
    if cell_type == "b":
        return raw == "1"
    if cell_type in ("str", "e"):
        return raw
    try:
        numeric = float(raw)
    except ValueError:
        return raw
    return int(numeric) if numeric.is_integer() else numeric


def _read_sheet(
    archive: ZipFile,
    target: str,
    shared_strings: list[str],
) -> list[list[Any]]:
    root = ElementTree.fromstring(archive.read(target))
    values: dict[tuple[int, int], Any] = {}
    maximum_row = -1
    maximum_column = -1
    for cell in root.findall(f".//{{{MAIN_NS}}}c"):
        reference = cell.attrib.get("r")
        if reference is None:
            raise ValueError("OOXML cell is missing its reference")
        row, column = _column_index(reference)
        values[(row, column)] = _cell_value(cell, shared_strings)
        maximum_row = max(maximum_row, row)
        maximum_column = max(maximum_column, column)
    if maximum_row < 0:
        return []
    return [
        [
            values.get((row, column))
            for column in range(maximum_column + 1)
        ]
        for row in range(maximum_row + 1)
    ]


def read_xlsx_sheets(path: str | Path) -> dict[str, list[list[Any]]]:
    """Return worksheet cell matrices without changing the source workbook."""

    source = Path(path)
    if source.suffix.lower() != ".xlsx":
        raise ValueError(f"Expected an .xlsx workbook, received {source.name!r}")
    try:
        with ZipFile(source) as archive:
            names = set(archive.namelist())
            required = {
                "xl/workbook.xml",
                "xl/_rels/workbook.xml.rels",
            }
            missing = required - names
            if missing:
                raise ValueError(
                    f"Workbook is missing required OOXML members: {sorted(missing)}"
                )
            shared_strings = _shared_strings(archive)
            return {
                sheet_name: _read_sheet(archive, target, shared_strings)
                for sheet_name, target in _sheet_targets(archive)
            }
    except BadZipFile as error:
        raise ValueError(f"Invalid .xlsx archive: {source}") from error
