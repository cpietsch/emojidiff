# Source before e2e/browser.mjs on gpubox-4080 (the operator's container): Chrome for Testing's
# missing system libraries, the Playwright browsers and NVIDIA Vulkan through libEGL_nvidia
# (the libGLX_nvidia ICD fails in this container). Milestone 0 installed all three under the
# probe directory; elsewhere, `npx playwright install chromium` and a working Vulkan driver.
probe=${MOJIDIFF_BROWSER_PROBE:-/home/dev/.cache/mojidiff/browser-probe-2026-10-04}
export LD_LIBRARY_PATH=$probe/sysroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PLAYWRIGHT_BROWSERS_PATH=$probe/pw-browsers
export VK_DRIVER_FILES=$probe/nvidia_egl_icd.json
export CHROME_ARGS="--enable-unsafe-webgpu --enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan --use-angle=vulkan --use-vulkan=native --ignore-gpu-blocklist --disable-vulkan-surface"
