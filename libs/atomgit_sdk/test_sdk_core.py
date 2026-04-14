#!/usr/bin/env python3
"""
AtomGit SDK 冒烟测试
重点验证 calculate_position 逻辑和 PRService 批量提交功能
"""

import sys
from pathlib import Path

print("=" * 80)
print("AtomGit SDK Smoke Test")
print("=" * 80)

# 测试 1: 导入检查
print("\n[TEST 1] SDK Import Check")
print("-" * 80)
try:
    from atomgit_sdk import (
        AtomGitClient,
        AtomGitConfig,
        CodeIssue,
        ArchitectureIssue,
        FixResult,
        AtomGitAPIError,
        ConfigurationError,
        DiffParseError,
    )
    from atomgit_sdk.utils import calculate_diff_position, parse_atomgit_url
    from atomgit_sdk.services import PRService, RepairService

    print("✓ All SDK modules imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# 测试 2: calculate_diff_position 核心算法
print("\n[TEST 2] Diff Position Calculation (Core Algorithm)")
print("-" * 80)

test_cases = [
    {
        "name": "Simple modification (line 12)",
        "patch": """@@ -10,5 +10,6 @@
 context line 1
 context line 2
-old line
+new line
 context line 3""",
        "line_number": 12,
        "is_new_file": False,
        "expected": "position in diff (not None)",
    },
    {
        "name": "New file",
        "patch": "",
        "line_number": 5,
        "is_new_file": True,
        "expected": 5,
    },
    {
        "name": "Empty patch (not new file)",
        "patch": "",
        "line_number": 10,
        "is_new_file": False,
        "expected": None,
    },
    {
        "name": "Invalid line number (0)",
        "patch": "@@ -1,3 +1,3 @@\n line1\n line2\n line3",
        "line_number": 0,
        "is_new_file": False,
        "expected": None,
    },
    {
        "name": "Multiple hunks (hunk starting at line 20)",
        "patch": """@@ -10,3 +10,4 @@
 first hunk
+addition
 end first
 
@@ -20,3 +21,4 @@
 second hunk
+another addition
 end second""",
        "line_number": 22,
        "is_new_file": False,
        "expected": "position in diff (not None)",
    },
]

passed = 0
failed = 0

for i, test in enumerate(test_cases, 1):
    print(f"\n  Test Case {i}: {test['name']}")
    result = calculate_diff_position(
        test["patch"], test["line_number"], test["is_new_file"]
    )

    if test["expected"] == "position in diff (not None)":
        success = result is not None
    else:
        success = result == test["expected"]

    if success:
        print(f"    ✓ PASS - Got: {result}")
        passed += 1
    else:
        print(f"    ✗ FAIL - Expected: {test['expected']}, Got: {result}")
        failed += 1

print(f"\n  Results: {passed}/{len(test_cases)} passed")

# 测试 3: 真实的 Diff 场景（从 atomgit-pr-review 提取）
print("\n[TEST 3] Real-world Diff Scenario")
print("-" * 80)

real_patch = """@@ -92,32 +92,34 @@ def calculate_position(self, patch: str, line_number: int, is_new_file: bool = F
     lines = patch.split("\\n")
     position = 0
     current_new_line = 0
     in_hunk = False
 
-        for i, line in enumerate(lines):
-            hunk_match = re.match(r"^@@\\s+-\\d+,?\\d*\\)\\s+\\(\\d+),?\\\\d*\\\\s+@@\\s+(\\d+),?", line)
-            if hunk_match:
-                in_hunk = True
-                position = i + 1
-                continue
+    for i, line in enumerate(lines):
+        hunk_match = re.match(r"^@@\\s+-\\d+,?\\d*\\s+\\+(\\d+),?\\d*\\s+@@", line)
+        if hunk_match:
+            in_hunk = True
+            position = i + 1
+            current_new_line = int(hunk_match.group(1)) - 1
+            continue
         
-            if not in_hunk:
-                continue
+        if not in_hunk:
+            continue"""

# 测试能否找到第 100 行（修改后的代码）
position = calculate_diff_position(real_patch, 100, False)
if position is not None and position > 0:
    print(f"  ✓ Can find line 100 in real diff at position {position}")
    passed += 1
else:
    print(f"  ✗ Failed to find line 100 in real diff (position: {position})")
    failed += 1

# 测试 4: URL 解析
print("\n[TEST 4] URL Parsing")
print("-" * 80)

url_tests = [
    {
        "url": "https://atomgit.com/wuxiaoqiang12/IB_Robot/pulls/123",
        "expected": {"owner": "wuxiaoqiang12", "repo": "IB_Robot", "pr_number": 123},
    },
    {
        "url": "https://atomgit.com/wuxiaoqiang12/IB_Robot",
        "expected": {"owner": "wuxiaoqiang12", "repo": "IB_Robot", "branch": "master"},
    },
    {
        "url": "https://atomgit.com/wuxiaoqiang12/IB_Robot/tree/feat/new-feature",
        "expected": {
            "owner": "wuxiaoqiang12",
            "repo": "IB_Robot",
            "branch": "feat/new-feature",
        },
    },
]

for test in url_tests:
    result = parse_atomgit_url(test["url"])
    if result == test["expected"]:
        print(f"  ✓ {test['url']}")
        passed += 1
    else:
        print(f"  ✗ {test['url']}")
        print(f"    Expected: {test['expected']}")
        print(f"    Got: {result}")
        failed += 1

# 测试 5: 数据模型
print("\n[TEST 5] Data Models")
print("-" * 80)

try:
    code_issue = CodeIssue(
        file="test.py",
        line=10,
        title="Test Issue",
        description="Test description",
        type="bug",
        severity="error",
        confidence=90,
    )
    print(f"  ✓ CodeIssue created: {code_issue.title}")
    passed += 1

    arch_issue = ArchitectureIssue(
        file="config.yaml",
        line=5,
        title="Architecture Violation",
        description="SSOT violation",
        pillar="ssot",
        severity="warning",
    )
    print(f"  ✓ ArchitectureIssue created: {arch_issue.title}")
    passed += 1

    fix_result = FixResult(
        has_fix=True,
        file_path="test.py",
        fix_description="Remove duplicate code",
        original_code="duplicate",
        fixed_code="unique",
    )
    print(f"  ✓ FixResult created: {fix_result.fix_description}")
    passed += 1

except Exception as e:
    print(f"  ✗ Model creation failed: {e}")
    failed += 1

# 测试 6: 配置加载（模拟）
print("\n[TEST 6] Configuration Loading (Mock)")
print("-" * 80)

try:
    # 测试 Pydantic 模型
    config = AtomGitConfig(
        token="test-token",
        owner="test-owner",
        repo="test-repo",
        base_url="https://api.atomgit.com",
    )
    print(f"  ✓ AtomGitConfig created: {config.owner}/{config.repo}")
    print(f"    Base URL: {config.base_url}")
    passed += 1
except Exception as e:
    print(f"  ✗ Config creation failed: {e}")
    failed += 1

# 测试 7: Client 初始化
print("\n[TEST 7] Client Initialization")
print("-" * 80)

try:
    config = AtomGitConfig(
        token="test-token",
        owner="test-owner",
        repo="test-repo",
    )
    client = AtomGitClient(config)
    print(f"  ✓ AtomGitClient initialized")
    print(
        f"    Headers: Authorization present = {bool('Authorization' in client.session.headers)}"
    )
    passed += 1
except Exception as e:
    print(f"  ✗ Client initialization failed: {e}")
    failed += 1

# 测试 8: PRService 初始化
print("\n[TEST 8] PRService Initialization")
print("-" * 80)

try:
    config = AtomGitConfig(
        token="test-token",
        owner="test-owner",
        repo="test-repo",
    )
    client = AtomGitClient(config)
    pr_service = PRService(client)
    print(f"  ✓ PRService initialized")
    print(f"    Methods: get_pr, get_pr_files, submit_batch_comments, etc.")
    passed += 1
except Exception as e:
    print(f"  ✗ PRService initialization failed: {e}")
    failed += 1

# 总结
print("\n" + "=" * 80)
print("SMOKE TEST SUMMARY")
print("=" * 80)
print(f"Total Tests: {passed + failed}")
print(f"✓ Passed: {passed}")
print(f"✗ Failed: {failed}")

if failed == 0:
    print("\n🎉 All smoke tests passed! SDK is ready for migration.")
    sys.exit(0)
else:
    print(f"\n⚠️  {failed} test(s) failed. Please fix before migration.")
    sys.exit(1)
