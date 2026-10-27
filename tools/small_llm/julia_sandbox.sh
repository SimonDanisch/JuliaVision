#!/usr/bin/env bash
# Execute a copied candidate/contract in a private, CPU-only namespace. No
# generated program runs in the persistent agent Julia session.
set -euo pipefail
task_dir=$(realpath "$1")
category=${2:-base}
compiled_modules=${JULIA_TASK_COMPILED_MODULES:-existing}
[[ "$compiled_modules" == existing || "$compiled_modules" == yes ]] || exit 2
# Keep Julia's bundled stdlib/pkgimage depots. Dropping them makes PoCL's
# generated LLVM intrinsics infer as dynamic calls, even for a correct kernel.
# Bundled files remain under the already read-only Julia distribution mount.
depot_order=${JULIA_TASK_DEPOT_ORDER:-withsystem}
case "$depot_order" in
    privatefirst) task_depots=/work/depot:/home/sim/.julia ;;
    cachefirst) task_depots=/home/sim/.julia:/work/depot ;;
    withsystem) task_depots=/work/depot:/home/sim/.julia: ;;
    *) exit 2 ;;
esac
root=/home/sim/Programmieren/VulkanDev
julia_tree=/home/sim/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu
args=(--clearenv --die-with-parent --new-session --unshare-all --cap-drop ALL
      --ro-bind /usr /usr --ro-bind /lib /lib --ro-bind /lib64 /lib64
      --proc /proc --dev /dev --tmpfs /tmp --dir /work
      --ro-bind "$julia_tree" "$julia_tree"
      --ro-bind "$root/Project.toml" "$root/Project.toml"
      --ro-bind "$root/Manifest.toml" "$root/Manifest.toml")
[[ ! -f "$root/LocalPreferences.toml" ]] || args+=(--ro-bind "$root/LocalPreferences.toml" "$root/LocalPreferences.toml")
for directory in packages artifacts compiled; do
    args+=(--ro-bind "/home/sim/.julia/$directory" "/home/sim/.julia/$directory")
done
# Makie's already-downloaded immutable font atlas; do not give generated code
# access to general user scratch data or permit network downloads.
args+=(--ro-bind /home/sim/.julia/scratchspaces/ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a/makie /work/depot/scratchspaces/ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a/makie)
for directory in Makie KernelAbstractions Adapt CompilerCaching; do
    args+=(--ro-bind "$root/dev/$directory" "$root/dev/$directory")
done
# Opt-in software Vulkan compiler/runtime qualification. No /dev/dri mount is
# added. Expose source components only, not these repositories' .git metadata.
if [[ "$category" == vulkan_kernel ]]; then
    for package in Mantle Lava Vulkan VulkanCore Raycore Metal; do
        package_root="$root/dev/$package"
        args+=(--dir "$package_root" --ro-bind "$package_root/Project.toml" "$package_root/Project.toml")
        for component in src ext lib deps generated; do
            [[ ! -d "$package_root/$component" ]] || args+=(--ro-bind "$package_root/$component" "$package_root/$component")
        done
    done
    args+=(--setenv VK_DRIVER_FILES /usr/share/vulkan/icd.d/lvp_icd.json
           --setenv VK_ICD_FILENAMES /usr/share/vulkan/icd.d/lvp_icd.json)
fi
[[ ! -d /etc/fonts ]] || args+=(--ro-bind /etc/fonts /etc/fonts)
args+=(--ro-bind "$task_dir" /input --tmpfs /home/sim/.cache
       --setenv HOME /work --setenv JULIA_DEPOT_PATH "$task_depots"
       --setenv PATH /usr/bin:/bin --setenv LANG C.UTF-8
       --setenv JULIA_NUM_THREADS 2 --setenv OPENBLAS_NUM_THREADS 1
       --setenv JULIA_GLFW_PLATFORM null --chdir /work)
task_file_limit=16777216
if [[ "$category" == vulkan_kernel ]]; then
    # Software Vulkan can use file-backed device memory larger than16MiB.
    # Keep a bounded per-file limit plus explicit private-tmpfs quotas; never
    # grant a writable host output directory to this category.
    [[ $# -lt 3 ]] || exit 2
    task_file_limit=536870912
    args+=(--size 1073741824 --tmpfs /work
           --size 536870912 --tmpfs /tmp
           --size 268435456 --tmpfs /home/sim/.cache)
fi
# Optional, explicitly scoped writable artifact directory. The candidate still
# cannot write its source, the project, caches, or arbitrary host paths.
if [[ $# -ge 3 ]]; then
    output_dir=$(realpath "$3")
    [[ -d "$output_dir" && "$output_dir" != "$task_dir" ]] || exit 2
    args+=(--bind "$output_dir" /output)
fi
compile_args=()
# Figure semantics, not CPU JIT throughput, are scored. Interpreting new host
# methods avoids spending minutes compiling a cold Makie layout per candidate.
[[ "$category" != makie ]] || compile_args+=(--compile=min -O0)
[[ "$category" != kernel ]] || compile_args+=(-g0)
[[ "$category" != vulkan_kernel ]] || compile_args+=(-g0)
# Limit CPU time, address space, subprocess count and writable file growth.
exec timeout --kill-after=5s 180s prlimit --cpu=150 --as=12884901888 --nproc=4096 --core=0 --fsize="$task_file_limit" \
    bwrap "${args[@]}" "$julia_tree/bin/julia" -C native --startup-file=no --compiled-modules="$compiled_modules" \
    --project="$root" "${compile_args[@]}" /input/runner.jl
