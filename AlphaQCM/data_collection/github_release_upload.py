#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通过 GitHub API 创建 Release 并上传大文件资产（Release assets）。

为什么需要这个脚本？
- GitHub 普通 git push 单文件限制 100MB；但 Release 附件允许更大的单文件（以 GitHub 实际限制为准，常见上限约 2GB）。
- 本脚本避免依赖 `gh` CLI（环境里可能没有、也未必有 sudo 安装权限）。

安全要求：
- 不要在命令行参数里直接传 token（会进入 shell history）。
- 请使用环境变量提供 token：`GH_TOKEN` 或 `GITHUB_TOKEN`。

用法示例：
  export GH_TOKEN="***"   # 建议在本机/安全终端设置，不要粘贴到聊天里
  python3 AlphaQCM/data_collection/github_release_upload.py \
    --tag data-2025-02-15 \
    --name "Dataset data-2025-02-15" \
    --files AlphaQCM/AlphaQCM_data/final_dataset_vision_metrics85_filtered.parquet \
            AlphaQCM/data_collection/final_dataset_vision_metrics85_filtered.parquet.sha256

如果整文件上传失败（网络不稳/超过限制），建议改为上传分片：
  python3 ... --files AlphaQCM/AlphaQCM_data/final_dataset_vision_metrics85_filtered.parquet.part_*
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import requests


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def get_token() -> str:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
    if not token:
        raise RuntimeError("未检测到 GH_TOKEN/GITHUB_TOKEN，请先设置环境变量再运行。")
    return token.strip()


def parse_repo_from_remote_url(url: str) -> str | None:
    """
    支持：
    - https://github.com/OWNER/REPO.git
    - git@github.com:OWNER/REPO.git
    """
    url = url.strip()
    m = re.search(r"github\\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/.]+)(?:\\.git)?$", url)
    if not m:
        return None
    return f"{m.group('owner')}/{m.group('repo')}"


def detect_git_root() -> Path:
    """
    尝试定位 git 仓库根目录。

    兼容场景：
    - 用户在仓库根目录执行脚本
    - 用户在任意目录执行脚本（但脚本文件本身位于仓库内）
    """
    import subprocess

    candidates = [Path.cwd(), Path(__file__).resolve().parent]
    for c in candidates:
        try:
            root = (
                subprocess.check_output(
                    ["git", "-C", str(c), "rev-parse", "--show-toplevel"],
                    stderr=subprocess.DEVNULL,
                )
                .decode("utf-8", "replace")
                .strip()
            )
            if root:
                return Path(root)
        except Exception:
            continue
    raise RuntimeError("未能定位 git 仓库根目录：请在仓库目录内执行，或使用 --repo 显式指定。")


def detect_repo() -> str:
    import subprocess

    root = detect_git_root()
    try:
        url = (
            subprocess.check_output(
                ["git", "-C", str(root), "remote", "get-url", "origin"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8", "replace")
            .strip()
        )
    except Exception:
        url = ""
    repo = parse_repo_from_remote_url(url) if url else None
    if not repo:
        raise RuntimeError(
            "无法从 git remote 自动解析 repo。"
            "请在仓库根目录执行，或用 --repo 显式指定（例如 --repo OWNER/REPO）。"
        )
    return repo


def detect_head_sha() -> str:
    import subprocess

    try:
        root = detect_git_root()
    except Exception:
        root = None

    try:
        cmd = ["git"]
        if root is not None:
            cmd += ["-C", str(root)]
        cmd += ["rev-parse", "HEAD"]
        sha = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8", "replace").strip()
        if re.fullmatch(r"[0-9a-f]{40}", sha):
            return sha
    except Exception:
        pass
    return ""


def gh_api_headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "CryptoQuant-ReleaseUploader",
    }


def create_or_get_release(
    session: requests.Session,
    repo: str,
    tag: str,
    name: str,
    body: str,
    draft: bool,
    target_commitish: str | None,
) -> dict:
    api = f"https://api.github.com/repos/{repo}/releases"
    payload = {
        "tag_name": tag,
        "name": name,
        "body": body,
        "draft": bool(draft),
        "prerelease": False,
        "generate_release_notes": False,
    }
    if target_commitish:
        payload["target_commitish"] = target_commitish

    r = session.post(api, json=payload, timeout=60)
    if r.status_code in (200, 201):
        return r.json()

    # tag 已存在时（422），尝试查找该 tag 对应 release
    if r.status_code == 422:
        get_api = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
        r2 = session.get(get_api, timeout=60)
        if r2.status_code == 200:
            return r2.json()

    raise RuntimeError(f"创建/获取 release 失败：{r.status_code} {r.text[:500]}")


def list_assets(session: requests.Session, release: dict) -> list[dict]:
    assets_url = release.get("assets_url")
    if not assets_url:
        return []
    r = session.get(assets_url, timeout=60)
    if r.status_code != 200:
        return []
    return r.json() if isinstance(r.json(), list) else []


def delete_asset_if_exists(session: requests.Session, release: dict, asset_name: str) -> None:
    for a in list_assets(session, release):
        if a.get("name") == asset_name and a.get("url"):
            r = session.delete(a["url"], timeout=60)
            if r.status_code not in (204, 404):
                raise RuntimeError(f"删除旧资产失败：{asset_name} {r.status_code} {r.text[:200]}")


class ProgressFile:
    def __init__(self, path: Path, chunk_size: int = 4 * 1024 * 1024) -> None:
        self.path = path
        self.f = path.open("rb")
        self.total = path.stat().st_size
        self.read_bytes = 0
        self.chunk_size = int(chunk_size)
        self._last_print = 0.0

    def __len__(self) -> int:
        return self.total

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.chunk_size
        b = self.f.read(n)
        self.read_bytes += len(b)
        now = time.time()
        if now - self._last_print > 1.0:
            self._last_print = now
            pct = (self.read_bytes / self.total) * 100 if self.total else 0.0
            eprint(f"  -> {self.path.name}: {self.read_bytes/1024/1024:.1f}MiB / {self.total/1024/1024:.1f}MiB ({pct:.1f}%)")
        return b

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass


def upload_asset(
    session: requests.Session,
    release: dict,
    file_path: Path,
    overwrite: bool,
    content_type: str = "application/octet-stream",
) -> None:
    upload_url = release.get("upload_url")
    if not upload_url:
        raise RuntimeError("release 返回中没有 upload_url")

    base = upload_url.split("{", 1)[0]  # strip {?name,label}
    name = file_path.name
    if overwrite:
        delete_asset_if_exists(session, release, name)

    url = f"{base}?name={requests.utils.quote(name)}"
    eprint(f"开始上传资产：{name} ({file_path.stat().st_size/1024/1024:.1f}MiB)")
    pf = ProgressFile(file_path)
    try:
        r = session.post(
            url,
            data=pf,
            headers={
                "Content-Type": content_type,
                "Content-Length": str(file_path.stat().st_size),
            },
            timeout=3600,
        )
    finally:
        pf.close()

    if r.status_code not in (200, 201):
        raise RuntimeError(f"上传失败：{name} {r.status_code} {r.text[:500]}")
    eprint(f"上传完成：{name}")


def expand_files(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in patterns:
        # 支持 shell 已展开 * 的情况；这里也兼容显式传入 glob
        if any(ch in p for ch in ["*", "?", "["]):
            out.extend(sorted(Path().glob(p)))
        else:
            out.append(Path(p))
    # 去重、保持顺序
    seen: set[str] = set()
    uniq: list[Path] = []
    for x in out:
        k = str(x.resolve())
        if k not in seen:
            seen.add(k)
            uniq.append(x)
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser(description="创建 GitHub Release 并上传大文件资产")
    ap.add_argument("--repo", default="", help="GitHub 仓库（OWNER/REPO），默认从 git remote origin 推断")
    ap.add_argument("--tag", required=True, help="Release tag，例如 data-2025-02-15")
    ap.add_argument("--name", default="", help="Release 名称，默认与 tag 相同")
    ap.add_argument("--body", default="", help="Release 描述（可选）")
    ap.add_argument("--draft", action="store_true", help="创建为 Draft（草稿）")
    ap.add_argument("--target", default="", help="target_commitish（默认 HEAD）")
    ap.add_argument("--overwrite", action="store_true", help="同名 asset 存在时先删除再上传")
    ap.add_argument("--files", nargs="+", required=True, help="要上传的文件路径（支持 glob）")
    args = ap.parse_args()

    token = get_token()
    repo = args.repo.strip() or detect_repo()
    tag = args.tag.strip()
    name = (args.name.strip() or tag)
    target = args.target.strip() or detect_head_sha() or None

    files = expand_files(args.files)
    if not files:
        raise RuntimeError("未找到任何 --files 文件")
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise RuntimeError(f"以下文件不存在：{missing}")

    session = requests.Session()
    session.headers.update(gh_api_headers(token))

    release = create_or_get_release(
        session=session,
        repo=repo,
        tag=tag,
        name=name,
        body=args.body,
        draft=args.draft,
        target_commitish=target,
    )
    eprint(f"release ok: repo={repo} tag={tag} draft={bool(release.get('draft'))}")

    for f in files:
        upload_asset(session, release, f, overwrite=bool(args.overwrite))

    html_url = release.get("html_url", "")
    if html_url:
        print(html_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
