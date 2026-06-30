import os
import re
import zipfile
from pathlib import Path
from typing import TypeAlias
from xml.sax.saxutils import escape


XlsxCell: TypeAlias = int | str


FILE_LIKE_NAMES = {
    "CMakeLists.txt",
    "LICENSE",
    "NOTICE",
    "README",
    "README.md",
    "README.en.md",
    "package.xml",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
}


def get_url_from_text(text, base_url, repo_root=None):
    clean_text = text.split('(')[0].strip().strip('[]`')
    match = re.match(r'^(?P<path>.+?):(?P<start>\d+)(?:-(?P<end>\d+))?$', clean_text)
    if match:
        path = match.group('path')
        start = match.group('start')
        end = match.group('end')
        lines = f"#L{start}-L{end}" if end else f"#L{start}"
        return f"{repo_url(path, base_url, repo_root, force_route='blob')}{lines}"
    return repo_url(clean_text, base_url, repo_root)


def looks_like_source_ref(text):
    clean_text = text.strip().strip('[]`')
    return bool(re.match(r'^.+?:\d+(?:-\d+)?$', clean_text))


def looks_like_repo_path(text):
    clean_text = text.strip().strip('[]`')
    if not clean_text or any(ch.isspace() for ch in clean_text):
        return False
    if clean_text.startswith(('http://', 'https://', '#', './', '../')):
        return False
    if re.fullmatch(r'\d+(?:\.\d+)*', clean_text):
        return False
    if clean_text.startswith('.'):
        return True
    if '/' in clean_text:
        return True
    return bool(re.search(r'\.[A-Za-z0-9][A-Za-z0-9_-]{0,15}$', clean_text))


def is_line_noise_link(text):
    return bool(re.fullmatch(r'line\s+\d+', text.strip(), flags=re.IGNORECASE))


def strip_page_marker(text):
    clean_text = text.strip()
    clean_text = re.sub(r'^#?\d+(?:\.\d+)*\s+', '', clean_text).strip()
    clean_text = re.sub(r'\s*\(\d+(?:\.\d+)*\)\s*$', '', clean_text).strip()
    return clean_text


def repo_url(path, base_url, repo_root=None, force_route=None):
    clean_path = path.strip().strip('[]`').lstrip('/')
    route = force_route or atomgit_route_for_path(clean_path, repo_root)
    repo_base, branch = split_atomgit_base_url(base_url)
    return f"{repo_base}/{route}/{branch}/{clean_path}"


def split_atomgit_base_url(base_url):
    clean_base = base_url.rstrip('/')
    repo_base, branch = clean_base.rsplit('/', 1)
    return repo_base, branch


def atomgit_route_for_path(path, repo_root=None):
    clean_path = path.strip().strip('[]`').lstrip('/')
    if repo_root:
        target = Path(repo_root) / clean_path
        if target.is_dir():
            return "tree"
        if target.is_file():
            return "blob"
    return "blob" if looks_like_file_path(clean_path) else "tree"


def looks_like_file_path(path):
    clean_path = path.strip().rstrip('/')
    if not clean_path:
        return False
    name = clean_path.rsplit('/', 1)[-1]
    if name in FILE_LIKE_NAMES:
        return True
    return bool(re.search(r'\.[A-Za-z0-9][A-Za-z0-9_-]{0,15}$', name))


def extract_page_id(text, url):
    clean_url = url.strip()
    if re.fullmatch(r'#?\d+(?:\.\d+)*', clean_url):
        return clean_url.lstrip('#')

    text_match = re.match(r'^#?(\d+(?:\.\d+)*)(?:\s+|$)', text.strip())
    if text_match:
        return text_match.group(1)

    parenthetical_match = re.search(r'\((\d+(?:\.\d+)*)\)\s*$', text.strip())
    if parenthetical_match:
        return parenthetical_match.group(1)
    return None


def resolve_page_link(text, url, current_filepath, id_to_label, title_to_label, label_to_filepath, warn, label_to_title=None):
    clean_url = url.strip()
    if clean_url and not re.fullmatch(r'#?\d+(?:\.\d+)*', clean_url):
        return None

    page_id = extract_page_id(text, url)
    if not page_id:
        if not clean_url:
            return resolve_title_link(text, current_filepath, title_to_label, label_to_filepath, warn, label_to_title)
        return None

    label = id_to_label.get(page_id)
    if not label:
        title_link = resolve_title_link(text, current_filepath, title_to_label, label_to_filepath, warn, label_to_title)
        if title_link:
            return title_link
        warn(f"Unresolved page id '{page_id}' in {current_filepath}: [{text}]({url})")
        return None

    target_path = label_to_filepath.get(label)
    if not target_path:
        warn(f"Page id '{page_id}' maps to missing label '{label}' in {current_filepath}: [{text}]({url})")
        return None

    link_text = strip_page_marker(text) or text.strip()
    relative_path = get_relative_path(current_filepath, target_path)
    return f"[{link_text}]({relative_path})"


def resolve_title_link(text, current_filepath, title_to_label, label_to_filepath, warn, label_to_title=None):
    original_text = text.strip()
    clean_text = original_text
    label = title_to_label.get(clean_text)
    if not label:
        clean_text = strip_page_marker(clean_text)
        label = title_to_label.get(clean_text)
    if not label:
        return None

    target_path = label_to_filepath.get(label)
    if not target_path:
        warn(f"Title '{clean_text}' maps to missing label '{label}' in {current_filepath}")
        return None

    relative_path = get_relative_path(current_filepath, target_path)
    link_text = label_to_title.get(label, original_text) if label_to_title else original_text
    return f"[{link_text}]({relative_path})"


def protect_blocks(content):
    placeholders = []

    def store_block(match):
        placeholder = f"@@DEEPWIKI_BLOCK_{len(placeholders)}@@"
        placeholders.append(match.group(0))
        return placeholder

    content = re.sub(r'```.*?```', store_block, content, flags=re.DOTALL)
    return content, placeholders


def restore_blocks(content, placeholders):
    for i, block in enumerate(placeholders):
        content = content.replace(f"@@DEEPWIKI_BLOCK_{i}@@", block)
    return content


def get_relative_path(from_file, to_file):
    from_dir = str(Path(from_file).parent)
    if from_dir == '.':
        return f"./{to_file}"
    try:
        rel = os.path.relpath(to_file, from_dir).replace('\\', '/')
        if not rel.startswith('.'):
            rel = f"./{rel}"
        return rel
    except ValueError:
        return to_file


def record_conversion(link_conversions, filepath, original, converted):
    link_conversions.append({
        "file": filepath,
        "original": original,
        "converted": converted,
    })


class MarkdownLinkMatch:
    def __init__(self, original, prefix, text, url):
        self.groups = (original, prefix, text, url)

    def group(self, index):
        return self.groups[index]


def replace_markdown_links(content, replace_link):
    output = []
    index = 0
    while index < len(content):
        prefix = "!" if content[index] == "!" and index + 1 < len(content) and content[index + 1] == "[" else ""
        link_start = index + 1 if prefix else index
        if content[link_start] != "[":
            output.append(content[index])
            index += 1
            continue

        text_end = content.find("]", link_start + 1)
        if text_end == -1 or text_end + 1 >= len(content) or content[text_end + 1] != "(":
            output.append(content[index])
            index += 1
            continue

        url_start = text_end + 2
        depth = 1
        url_end = url_start
        while url_end < len(content):
            char = content[url_end]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    break
            url_end += 1

        if depth != 0:
            output.append(content[index])
            index += 1
            continue

        original = content[index:url_end + 1]
        text = content[link_start + 1:text_end]
        url = content[url_start:url_end]
        output.append(replace_link(MarkdownLinkMatch(original, prefix, text, url)))
        index = url_end + 1
    return "".join(output)


def report_link_conversions(link_conversions, output_dir):
    if not link_conversions:
        print("\nNo link conversions recorded.")
        return

    entries = link_conversions
    output_dir = Path(output_dir)
    report_dir = output_dir.parent / "reports"
    report_path = report_dir / "link_conversions.xlsx"
    report_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd

        df = pd.DataFrame(entries)
        df.index = df.index + 1
        df.index.name = "#"
        df.columns = ["File", "Original", "Converted"]

        with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Link Conversions")
            ws = writer.sheets["Link Conversions"]
            ws.column_dimensions["B"].width = 30
            ws.column_dimensions["C"].width = 50
            ws.column_dimensions["D"].width = 50
    except ModuleNotFoundError:
        write_link_conversions_xlsx(entries, report_path)

    print(f"\nLink Conversions ({len(entries)} total) written to {report_path}")


def write_link_conversions_xlsx(entries, report_path):
    rows: list[list[XlsxCell]] = [["#", "File", "Original", "Converted"]]
    rows.extend([[index, entry["file"], entry["original"], entry["converted"]] for index, entry in enumerate(entries, start=1)])

    def cell_ref(row_index, col_index):
        col_name = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[col_index]
        return f"{col_name}{row_index}"

    row_xml = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        for col_index, value in enumerate(row):
            ref = cell_ref(row_index, col_index)
            if isinstance(value, int):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>')
        row_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    sheet_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<cols><col min="2" max="2" width="30" customWidth="1"/><col min="3" max="4" width="50" customWidth="1"/></cols>
<sheetData>{"".join(row_xml)}</sheetData>
</worksheet>'''

    with zipfile.ZipFile(report_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>''')
        archive.writestr("_rels/.rels", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>''')
        archive.writestr("xl/workbook.xml", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="Link Conversions" sheetId="1" r:id="rId1"/></sheets>
</workbook>''')
        archive.writestr("xl/_rels/workbook.xml.rels", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>''')
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def fix_links(content, current_filepath, base_url, id_to_label, title_to_label, label_to_filepath, link_conversions, warn, repo_root=None, label_to_title=None):
    content, placeholders = protect_blocks(content)

    def replace_backtick_wrapped_link(m):
        text = m.group(1).strip()
        url = m.group(2).strip()
        original = m.group(0)
        converted = f"[{text}]({url})"
        if original != converted:
            record_conversion(link_conversions, current_filepath, original, converted)
        return converted

    content = re.sub(
        r'`\[([^\]\n]+?)\]\((https?://[^)\n]+)\)`',
        replace_backtick_wrapped_link,
        content,
    )

    def replace_bare_code_source(m):
        text = m.group(1).strip()
        if not looks_like_source_ref(text):
            return m.group(0)
        original = m.group(0)
        converted = f"[{text}]({get_url_from_text(text, base_url, repo_root)})"
        record_conversion(link_conversions, current_filepath, original, converted)
        return converted

    content = re.sub(r'`([^`\n]+?)`\(\)', replace_bare_code_source, content)

    def replace_double_source(m):
        text = m.group(1).strip().replace('`', '')
        if not looks_like_source_ref(text):
            warn(f"Unresolved double-bracket link in {current_filepath}: [[{text}]]()")
            return m.group(0)
        original = m.group(0)
        converted = f"[{text}]({get_url_from_text(text, base_url, repo_root)})"
        record_conversion(link_conversions, current_filepath, original, converted)
        return converted

    content = re.sub(r'\[\[(.+?)\]\]\(\)', replace_double_source, content)

    def replace_link(m):
        prefix = m.group(1)
        text = m.group(2).strip().replace('`', '')
        url = m.group(3).strip()

        if prefix:
            return m.group(0)

        if url.startswith(('http://', 'https://', './', '../')):
            return m.group(0)

        if url.startswith('#') and not re.fullmatch(r'#\d+(?:\.\d+)*', url):
            return m.group(0)

        page_link = resolve_page_link(text, url, current_filepath, id_to_label, title_to_label, label_to_filepath, warn, label_to_title)
        if page_link:
            original = m.group(0)
            if original != page_link:
                record_conversion(link_conversions, current_filepath, original, page_link)
            return page_link

        if not url:
            if is_line_noise_link(text):
                original = m.group(0)
                record_conversion(link_conversions, current_filepath, original, "(removed)")
                return ""
            if looks_like_source_ref(text):
                original = m.group(0)
                converted = f"[{text}]({get_url_from_text(text, base_url, repo_root)})"
                record_conversion(link_conversions, current_filepath, original, converted)
                return converted
            if looks_like_repo_path(text):
                original = m.group(0)
                converted = f"[{text}]({repo_url(text, base_url, repo_root)})"
                record_conversion(link_conversions, current_filepath, original, converted)
                return converted
            warn(f"Unresolved empty link in {current_filepath}: [{text}]()")
            return m.group(0)

        if looks_like_repo_path(url):
            original = m.group(0)
            converted = f"[{text}]({repo_url(url, base_url, repo_root)})"
            if original != converted:
                record_conversion(link_conversions, current_filepath, original, converted)
            return converted

        warn(f"Unresolved link in {current_filepath}: [{text}]({url})")
        return m.group(0)

    content = replace_markdown_links(content, replace_link)
    content = re.sub(
        r'`\[([^\]\n]+?)\]\((https?://[^)\n]+)\)`',
        replace_backtick_wrapped_link,
        content,
    )
    return restore_blocks(content, placeholders)
