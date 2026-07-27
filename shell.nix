# Nix shell for NixOS users. Prebuilt manylinux wheels (torch, pyarrow, ...)
# don't run on NixOS directly because there is no /lib64 dynamic linker, so
# this provides an FHS environment where uv and the wheels work unmodified.
#
# Usage:
#   nix-shell          # drops you into the env, then follow the README quickstart
#
# If you already run nix-ld (programs.nix-ld.enable = true), you likely don't
# need this file at all — the plain quickstart should work.
#
# Non-NixOS Linux with the nix package manager also doesn't need this.
{ pkgs ? import <nixpkgs> {} }:
(pkgs.buildFHSEnv {
  name = "autoresearch-env";
  targetPkgs = ps: with ps; [
    uv
    git
    curl
    zlib
    openssl
    stdenv.cc.cc.lib  # libstdc++ for torch/pyarrow wheels
  ];
  profile = ''
    # NVIDIA on NixOS: userspace driver (libcuda.so) lives outside the FHS root
    if [ -d /run/opengl-driver/lib ]; then
      export LD_LIBRARY_PATH=/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    fi
  '';
  runScript = "bash";
}).env
