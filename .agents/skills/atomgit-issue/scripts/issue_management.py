#!/usr/bin/env python3
"""
AtomGit Issue Management Tool
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from atomgit_sdk import AtomGitClient, IssueService, resolve_atomgit_context
except ImportError:
    print(
        "Error: AtomGit SDK not found. Please install project requirements by running setup.sh, "
        "or install atomgit-sdk in the active environment."
    )
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Manage AtomGit issues across create, fetch, update, and close flows")
    parser.add_argument("--title", help="Issue title")
    parser.add_argument("--body", help="Issue body")
    parser.add_argument("--issue", type=int, help="Issue number for update or fetch")
    parser.add_argument("--state", choices=["open", "closed"], help="Issue state for update")
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
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--owner", help="Target repository owner override")
    parser.add_argument("--repo", help="Target repository name override")
    parser.add_argument("--url", help="Issue or repository URL for auto-resolving owner/repo/issue")
    parser.add_argument("--output-dir", default="tmp", help="Output directory for JSON")
    parser.add_argument("--dry-run", action="store_true", help="Preview action without executing")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    try:
        config, parsed_url = resolve_atomgit_context(args.config, owner=args.owner, repo=args.repo, url=args.url)
        client = AtomGitClient(config)
        service = IssueService(client)
    except Exception as e:
        print(f"Error initializing AtomGit SDK: {e}")
        sys.exit(1)

    if args.issue is None:
        args.issue = parsed_url.get("issue_number")

    print(f"Target repo: {config.owner}/{config.repo}")
    if args.url:
        print(f"Resolved from URL: {args.url}")

    # 1. Fetch info mode
    if args.fetch_info:
        if not args.issue:
            print("Error: --issue <number> is required for --fetch-info")
            sys.exit(1)

        print(f"Fetching info for issue #{args.issue}...")
        try:
            issue_data = service.get_issue(args.issue)
            if not args.no_comments:
                issue_data["comments_detail"] = service.get_issue_comments(args.issue)

            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            repo_name = config.repo.lower().replace("-", "_")
            output_file = output_dir / f"{repo_name}_issue_{args.issue}_context.json"

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(issue_data, f, indent=2, ensure_ascii=False)

            print(f"Successfully saved issue info to {output_file}")
            if not args.no_comments:
                print(f"Included {len(issue_data.get('comments_detail', []))} issue comments")
            return
        except Exception as e:
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
            service.submit_issue_comment(args.issue, args.comment)
            print(f"Successfully commented on issue #{args.issue}")
            print(f"URL: {service.get_issue_url(args.issue)}")
            return
        except Exception as e:
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
            result = service.update_issue(args.issue, title=args.title, body=args.body, state=args.state)
            print(f"Successfully updated issue #{args.issue}")
            print(f"URL: {service.get_issue_url(args.issue)}")
            return
        except Exception as e:
            print(f"Error updating issue: {e}")
            sys.exit(1)

    # 3. Create issue mode
    if not args.title:
        print("Error: --title is required to create a new issue")
        sys.exit(1)

    print(f"Creating new issue: {args.title}...")

    if args.dry_run:
        print("[DRY RUN] Plan to create issue:")
        print(f"  Title: {args.title}")
        return

    try:
        result = service.create_issue(title=args.title, body=args.body or "")
        issue_number = result.get("number")
        print(f"Successfully created issue #{issue_number}")
        print(f"URL: {service.get_issue_url(issue_number)}")
    except Exception as e:
        print(f"Error creating issue: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
