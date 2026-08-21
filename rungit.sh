#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# AutoDL 默认同步实验分支；也可以传入其他远端分支名：
#   bash rungit.sh rtsplat-baseline
TARGET_BRANCH="${1:-rtsplat-test}"

git fetch origin "$TARGET_BRANCH"
git reset --hard "origin/$TARGET_BRANCH"

# ``git pull`` 不会自动初始化子模块。先同步 URL，再强制检出主仓库
# 记录的 3dgrut commit；随后只初始化 3DGRT 编译必需的 OptiX headers。
git submodule sync -- submodules/3dgrut
git submodule update --init --force submodules/3dgrut
git -C submodules/3dgrut submodule sync -- threedgrt_tracer/dependencies/optix-dev
git -C submodules/3dgrut submodule update --init --force threedgrt_tracer/dependencies/optix-dev

echo "Updated origin/$TARGET_BRANCH with pinned 3DGRT and OptiX headers."
