#!/usr/bin/env python3
"""
AtomGit Issue Management Tool

Platform access goes through the official AtomGit CLI via the shared boundary
in ``.agents/skills/_shared/atomgit_cli.py`` (OpenSpec change
``migrate-atomgit-skills-to-cli``). Authentication is owned by the CLI:
``gitcode auth login --hostname atomgit.com --with-token``.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "_shared"))

import atomgit_cli


def parse_args():
    parser = argparse.ArgumentParser(description="Manage AtomGit issues across create, fetch, update, and close flows")
    parser.add_argument("--title", help="Issue title")
    parser.add_argument("--body", help="Issue body")
    parser.add_argument("--issue", type=int, help="Issue number for update or fetch")
    parser.add_argument("--state", choices=["open", "closed", "reopen"], help="Issue state for update")
    parser.add_argument("--fetch-info", action="store_true", help="Fetch issue info to JSON")
    parser.add_argument(
        "--comment",
        help="Post a comment (or a community slash command like '/kind bug') on an existing issue",
    )
    parser.add_argument(
        "--no-comments",
        action="store_true",
        help="Skip fetching issue comments in --fetch-info mode",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.json (optional atomgit.owner/repo defaults; no longer used for authentication)",
    )
    parser.add_argument("--owner", help="Target repository owner override")
    parser.add_argument("--repo", help="Target repository name override")
    parser.add_argument("--url", help="Issue or repository URL for auto-resolving owner/repo/issue")
    parser.add_argument("--output-dir", default="tmp", help="Output directory for JSON")
    parser.add_argument("--dry-run", action="store_true", help="Preview action without executing")

    return parser.parse_args()


def issue_url(target: atomgit_cli.RepoTarget, issue_number: int) -> str:
    """Construct the web URL locally (no platform call; registered web host only)."""
    return f"https://{atomgit_cli.DEFAULT_WEB_HOST}/{target.owner}/{target.repo}/issues/{issue_number}"


def run_or_raise(argv: list[str], cli: str, *, stdin_text: str | None = None) -> atomgit_cli.CLIResult:
    """Run one CLI invocation; a nonzero exit raises its classified failure."""
    result = atomgit_cli.run(argv, cli=cli, stdin_text=stdin_text)
    failure = result.failure()
    if failure is not None:
        raise failure
    return result


def main():
    args = parse_args()

    # Load configuration
    try:
        cli = atomgit_cli.resolve_cli()
        atomgit_cli.check_cli_identity(cli)
        target = atomgit_cli.resolve_target(
            owner=args.owner,
            repo=args.repo,
            url=args.url,
            config_path=args.config or "config.json",
            explicit_config=args.config is not None,
            remotes=atomgit_cli.read_git_remotes(),
        )
    except (atomgit_cli.CLIError, atomgit_cli.TargetResolutionError) as e:
        print(f"Error initializing AtomGit CLI access: {e}")
        sys.exit(1)

    if args.issue is None and target.url_info:
        args.issue = target.url_info.get("issue_number")

    print(f"Target repo: {target.slug}")
    if args.url:
        print(f"Resolved from URL: {args.url}")

    # 1. Fetch info mode
    if args.fetch_info:
        if not args.issue:
            print("Error: --issue <number> is required for --fetch-info")
            sys.exit(1)

        print(f"Fetching info for issue #{args.issue}...")
        try:
            view = atomgit_cli.run_json(
                ["issue", "view", str(args.issue), "-R", target.slug, "--json"],
                cli=cli,
            )
            issue_data = view.require_data()
            if not isinstance(issue_data, dict):
                raise atomgit_cli.CLIJSONError(
                    f"issue view returned {type(issue_data).__name__} instead of a JSON object",
                    argv=view.argv,
                    returncode=view.returncode,
                    stdout=view.stdout,
                    stderr=view.stderr,
                )
            if not args.no_comments:
                comments = atomgit_cli.run_json(
                    [
                        "issue",
                        "comments",
                        str(args.issue),
                        "-R",
                        target.slug,
                        "--json",
                        "--limit",
                        "0",
                        "--order",
                        "asc",
                    ],
                    cli=cli,
                )
                comment_list = comments.require_data()
                if not isinstance(comment_list, list):
                    raise atomgit_cli.CLIJSONError(
                        f"issue comments returned {type(comment_list).__name__} instead of a JSON array",
                        argv=comments.argv,
                        returncode=comments.returncode,
                        stdout=comments.stdout,
                        stderr=comments.stderr,
                    )
                non_object_entries = [comment for comment in comment_list if not isinstance(comment, dict)]
                if non_object_entries:
                    raise atomgit_cli.CLIJSONError(
                        f"issue comments array contains {len(non_object_entries)} non-object entries; "
                        "refusing to write a partial issue context",
                        argv=comments.argv,
                        returncode=comments.returncode,
                        stdout=comments.stdout,
                        stderr=comments.stderr,
                    )
                issue_data["comments_detail"] = comment_list

            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            repo_name = target.repo.lower().replace("-", "_")
            output_file = output_dir / f"{repo_name}_issue_{args.issue}_context.json"

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(issue_data, f, indent=2, ensure_ascii=False)

            print(f"Successfully saved issue info to {output_file}")
            if not args.no_comments:
                print(f"Included {len(issue_data.get('comments_detail', []))} issue comments")
            return
        except atomgit_cli.CLIError as e:
            print(f"Error fetching issue info: {e}")
            sys.exit(1)

    # 2. Comment mode (post a comment or community slash command on an existing issue)
    if args.comment is not None:
        if not args.issue:
            print("Error: --issue <number> is required for --comment")
            sys.exit(1)
        print(f"Commenting on issue #{args.issue}...")
        if args.dry_run:
            print(f"[DRY RUN] Comment: {args.comment}")
            return
        try:
            run_or_raise(
                [
                    "issue",
                    "comment",
                    str(args.issue),
                    "-R",
                    target.slug,
                    "--body-file",
                    "-",
                    "--json",
                ],
                cli=cli,
                stdin_text=args.comment,
            )
            print(f"Successfully commented on issue #{args.issue}")
            print(f"URL: {issue_url(target, args.issue)}")
            return
        except atomgit_cli.CLIError as e:
            print(f"Error commenting on issue: {e}")
            sys.exit(1)

    # 3. Update issue mode
    if args.issue and args.comment is None:
        print(f"Updating issue #{args.issue}...")

        if args.dry_run:
            print("[DRY RUN] Plan to update issue:")
            if args.title:
                print(f"  Title: {args.title}")
            if args.state:
                print(f"  State: {args.state}")
            return

        try:
            argv = ["issue", "edit", str(args.issue), "-R", target.slug]
            stdin_text = None
            if args.title:
                argv += ["--title", args.title]
            if args.body is not None:
                argv += ["--body-file", "-"]
                stdin_text = args.body
            if args.state:
                argv += ["--state", args.state, "--yes"]
            argv += ["--json"]
            run_or_raise(argv, cli=cli, stdin_text=stdin_text)
            print(f"Successfully updated issue #{args.issue}")
            print(f"URL: {issue_url(target, args.issue)}")
            return
        except atomgit_cli.CLIError as e:
            print(f"Error updating issue: {e}")
            sys.exit(1)

    # 4. Create issue mode
    if not args.title:
        print("Error: --title is required to create a new issue")
        sys.exit(1)

    print(f"Creating new issue: {args.title}...")

    if args.dry_run:
        print("[DRY RUN] Plan to create issue:")
        print(f"  Title: {args.title}")
        return

    try:
        result = atomgit_cli.run_json(
            [
                "issue",
                "create",
                "-R",
                target.slug,
                "--title",
                args.title,
                "--body-file",
                "-",
                "--json",
            ],
            cli=cli,
            stdin_text=args.body or "",
        )
        data = result.require_data()
        if not isinstance(data, dict) or data.get("number") is None:
            # rc=0 means the mutation was very likely applied; never auto-retry.
            print(
                "Error: issue create succeeded but returned no issue number in its JSON output; "
                "the issue may already exist — check the repository issue list before retrying"
            )
            sys.exit(1)
        issue_number = data["number"]
        print(f"Successfully created issue #{issue_number}")
        print(f"URL: {issue_url(target, issue_number)}")
    except atomgit_cli.CLIError as e:
        print(f"Error creating issue: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
