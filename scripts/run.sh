#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/.."
echo "Project directory: $PROJECT_DIR"

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <network_interface> <ros_distro> <command...>" >&2
  exit 2
fi

NET_IF="$1"
ROS_DISTRO="$2"
shift 2

if [[ "$1" == "colcon" && "${2:-}" == "build" ]]; then
  # Building does not require runtime networking, sudo, or an existing install.
  source "/opt/ros/$ROS_DISTRO/setup.bash"
  export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
  # Humble's colcon supplies its own make -j value unless MAKEFLAGS sets one.
  export MAKEFLAGS="${MAKEFLAGS:--j${CMAKE_BUILD_PARALLEL_LEVEL}}"
else
  source "$PROJECT_DIR/../unitree_lowlevel/scripts/setup.sh" "$NET_IF" "$ROS_DISTRO"
fi

cd "$PROJECT_DIR"
if [[ "$(uname -m)" == "x86_64" ]]; then

  if [[ -d "$PROJECT_DIR/thirdparty/libtorch" ]]; then
    export CMAKE_PREFIX_PATH="$PROJECT_DIR/thirdparty/libtorch:${CMAKE_PREFIX_PATH:-}"
    export LD_LIBRARY_PATH="$PROJECT_DIR/thirdparty/libtorch/lib:${LD_LIBRARY_PATH:-}"
  fi

else

  export CMAKE_PREFIX_PATH="$(uv run python3 -c 'import torch; print(torch.utils.cmake_prefix_path)'):${CMAKE_PREFIX_PATH}"
  export LD_LIBRARY_PATH="$(uv run python3 -c 'import torch, pathlib; p=pathlib.Path(torch.__file__).resolve().parent; print(p / "lib")'):${LD_LIBRARY_PATH}"

fi

echo "CMAKE_PREFIX_PATH: $CMAKE_PREFIX_PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

cd "$PROJECT_DIR/../../"
"$@"
