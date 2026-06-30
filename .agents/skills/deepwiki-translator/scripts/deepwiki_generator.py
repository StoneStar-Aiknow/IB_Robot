import shutil
from pathlib import Path

from deepwiki_config import validate_config
from deepwiki_links import fix_links, report_link_conversions
from deepwiki_pages import clean_content, load_pages


def reset_output_dir(output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def build_directory_index(display_title, sub_toctree):
    index_content = f"{display_title}\n{'#' * (len(display_title) * 2)}\n\n"
    index_content += f"本章节包含关于{display_title}的详细指南和参考资料。\n\n"
    index_content += ".. toctree::\n   :maxdepth: 1\n\n"
    for item in sub_toctree:
        index_content += f"   {item}\n"
    return index_content


def build_main_index(index_toctree):
    main_index = ".. _ib_robot_intro:\n\nIB-Robot 具身智能套件\n################################\n\n"
    main_index += "IB-Robot（Intelligence Boom Robot）是一个将 Hugging Face LeRobot 机器学习生态系统与 ROS 2 机器人中间件连接起来的集成开发框架，旨在实现端到端的具身智能（Embodied AI）工作流。\n\n"
    main_index += ".. toctree::\n   :maxdepth: 2\n   :caption: 内容\n\n"
    for item in index_toctree:
        main_index += f"   {item}\n"
    return main_index


def print_warnings(warnings):
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            try:
                print(f"  - {warning}")
            except UnicodeEncodeError:
                print(f"  - {warning.encode('ascii', 'replace').decode('ascii')}")


def generate_leaf_page(output_dir, name, cfg, title_to_content, base_url, id_to_label, title_to_label, label_to_filepath, link_conversions, warn, repo_root, label_to_title=None):
    print(f"Generating {name}...")
    title = cfg["title"]
    if title in title_to_content:
        content = clean_content(title_to_content[title], title)
        converted = fix_links(content, name, base_url, id_to_label, title_to_label, label_to_filepath, link_conversions, warn, repo_root, label_to_title)
        with open(Path(output_dir) / name, 'w', encoding='utf-8') as f_out:
            f_out.write(converted)
        return Path(name).stem
    warn(f"Skipped leaf page because input title was not found: {title} -> {name}")
    return None


def generate_directory_section(output_dir, name, cfg, title_to_content, base_url, id_to_label, title_to_label, label_to_filepath, link_conversions, warn, repo_root, label_to_title=None):
    print(f"Creating directory {name}...")
    output_dir = Path(output_dir)
    dir_path = output_dir / name
    if not dir_path.exists():
        dir_path.mkdir(parents=True)

    sub_toctree = []
    main_title = cfg["title"]

    if main_title in title_to_content:
        print(f"  Generating overview for {name}...")
        overview_content = clean_content(title_to_content[main_title], main_title)
        overview_content = fix_links(overview_content, f"{name}/overview.md", base_url, id_to_label, title_to_label, label_to_filepath, link_conversions, warn, repo_root, label_to_title)
        with open(dir_path / "overview.md", 'w', encoding='utf-8') as f_out:
            f_out.write(overview_content)
        sub_toctree.append("overview")
    else:
        warn(f"Skipped overview because input title was not found: {main_title} -> {name}/overview.md")

    for sub_file, sub_title in cfg["subs"].items():
        print(f"  Generating sub-page {name}/{sub_file}...")
        if sub_title in title_to_content:
            sub_content = clean_content(title_to_content[sub_title], sub_title)
            sub_content = fix_links(sub_content, f"{name}/{sub_file}", base_url, id_to_label, title_to_label, label_to_filepath, link_conversions, warn, repo_root, label_to_title)
            with open(dir_path / sub_file, 'w', encoding='utf-8') as f_out:
                f_out.write(sub_content)
            sub_toctree.append(Path(sub_file).stem)
        else:
            warn(f"Skipped sub-page because input title was not found: {sub_title} -> {name}/{sub_file}")

    if not sub_toctree:
        warn(f"Directory has no generated pages: {name}")
        return None

    display_title = main_title.replace("概述", "").strip()
    with open(dir_path / "index.rst", 'w', encoding='utf-8') as f_out:
        f_out.write(build_directory_index(display_title, sub_toctree))
    return f"{name}/index"


def run_generation(input_dir, output_dir, branch, id_to_label, title_to_label, hierarchy, label_to_filepath, warnings, link_conversions, label_to_title=None):
    output_dir = Path(output_dir)
    base_url = f"https://atomgit.com/openeuler/IB_Robot/{branch}/"
    repo_root = Path.cwd()

    def warn(message):
        warnings.append(message)

    reset_output_dir(output_dir)

    title_to_content, _ = load_pages(input_dir, warn)
    validate_config(hierarchy, id_to_label, label_to_filepath, title_to_content, warn)

    index_toctree = []

    for name, cfg in hierarchy.items():
        if "subs" not in cfg:
            item = generate_leaf_page(output_dir, name, cfg, title_to_content, base_url, id_to_label, title_to_label, label_to_filepath, link_conversions, warn, repo_root, label_to_title)
            if item:
                index_toctree.append(item)
        else:
            item = generate_directory_section(output_dir, name, cfg, title_to_content, base_url, id_to_label, title_to_label, label_to_filepath, link_conversions, warn, repo_root, label_to_title)
            if item:
                index_toctree.append(item)

    print("Generating main index.rst...")
    with open(output_dir / "index.rst", 'w', encoding='utf-8') as f_out:
        f_out.write(build_main_index(index_toctree))

    print_warnings(warnings)
    report_link_conversions(link_conversions, output_dir)
