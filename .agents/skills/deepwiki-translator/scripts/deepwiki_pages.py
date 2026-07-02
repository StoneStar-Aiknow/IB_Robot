from pathlib import Path


def clean_content(md_content, page_title):
    lines = md_content.split('\n')

    while lines and not lines[0].strip():
        lines = lines[1:]

    title_heading = f"# {page_title}".strip()
    if not lines or lines[0].strip() != title_heading:
        lines.insert(0, title_heading)

    normalized = [lines[0]]
    index = 1
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        if lines[index].strip() == title_heading:
            index += 1
            continue
        break

    normalized.append('')
    normalized.extend(lines[index:])
    remove_trailing_transitions(normalized)
    return '\n'.join(normalized).rstrip() + '\n'


def remove_trailing_transitions(lines):
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and lines[-1].strip() in {"---", "***", "___"}:
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()


def load_pages(input_dir, warn):
    md_files = list(Path(input_dir).glob("*.md"))
    title_to_content = {}
    used_files = set()

    for file_path in md_files:
        with open(file_path, 'r', encoding='utf-8-sig') as f_in:
            text = f_in.read()
        title = extract_title(text, file_path, warn)
        if not title:
            continue
        if title in title_to_content:
            warn(f"Duplicate page title '{title}' in {file_path}")
        title_to_content[title] = text
        used_files.add(file_path)

    return title_to_content, used_files


def extract_title(text, file_path, warn):
    for line in text.split('\n'):
        stripped = line.strip()
        if stripped.startswith('# '):
            return stripped[2:].strip()
        if stripped:
            warn(f"Missing leading H1 in {file_path}; first content line is: {stripped[:80]}")
            return None
    warn(f"Empty markdown file: {file_path}")
    return None
