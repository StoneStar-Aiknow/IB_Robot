import json
import sys


def load_config(config_file):
    with open(config_file, 'r', encoding='utf-8-sig') as f:
        config = json.load(f)
    return config["id_to_label"], config["title_to_label"], config["hierarchy"]


def merged_title_to_label(title_to_label, source_config_file=None):
    merged = dict(title_to_label)
    if not source_config_file:
        return merged
    try:
        _, source_title_to_label, _ = load_config(source_config_file)
    except FileNotFoundError:
        return merged
    except (json.JSONDecodeError, KeyError) as exc:
        print(f"Warning: failed to load source config {source_config_file}: {exc}", file=sys.stderr)
        return merged
    merged.update(source_title_to_label)
    return merged


def label_to_title(title_to_label):
    return {label: title for title, label in title_to_label.items()}


def build_label_to_filepath(hierarchy, title_to_label):
    label_to_filepath = {}
    for key, cfg in hierarchy.items():
        if "subs" not in cfg:
            title = cfg["title"]
            label = title_to_label.get(title, "")
            if label:
                label_to_filepath[label] = key
        else:
            dir_name = key
            main_title = cfg["title"]
            main_label = title_to_label.get(main_title, "")
            if main_label:
                label_to_filepath[main_label] = f"{dir_name}/overview.md"
            for sub_file, sub_title in cfg["subs"].items():
                sub_label = title_to_label.get(sub_title, "")
                if sub_label:
                    label_to_filepath[sub_label] = f"{dir_name}/{sub_file}"
    return label_to_filepath


def configured_titles(hierarchy):
    titles = []
    for cfg in hierarchy.values():
        titles.append(cfg["title"])
        if "subs" in cfg:
            titles.extend(cfg["subs"].values())
    return titles


def validate_config(hierarchy, id_to_label, label_to_filepath, title_to_content, warn):
    configured = set(configured_titles(hierarchy))
    input_titles = set(title_to_content.keys())

    for title in sorted(configured - input_titles):
        warn(f"Configured title missing from input: {title}")

    for title in sorted(input_titles - configured):
        warn(f"Input page not used by hierarchy: {title}")

    for page_id, label in id_to_label.items():
        if label not in label_to_filepath:
            warn(f"id_to_label maps page id '{page_id}' to label without output path: {label}")
