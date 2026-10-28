# Splitting the Intel B60 (SR-IOV) after a reboot

The VFs do not survive a reboot. Boot kernel `6.17.0-42-generic` and use the kernel command line `intel_iommu=off xe.max_vfs=7` (already set in GRUB).

## 1. Check the kernel and that the PF offers VFs

```
uname -r                                                # 6.17.0-42-generic
cat /proc/cmdline                                       # must contain xe.max_vfs=7
09:00.0 VGA compatible controller [0300]: Intel Corporation Device [8086:e211]
09:00.1 VGA compatible controller [0300]: Intel Corporation Device [8086:e211]
09:00.2 VGA compatible controller [0300]: Intel Corporation Device [8086:e211]
09:00.3 VGA compatible controller [0300]: Intel Corporation Device [8086:e211]
09:00.4 VGA compatible controller [0300]: Intel Corporation Device [8086:e211]
09:00.5 VGA compatible controller [0300]: Intel Corporation Device [8086:e211]
09:00.6 VGA compatible controller [0300]: Intel Corporation Device [8086:e211]
09:00.7 VGA compatible controller [0300]: Intel Corporation Device [8086:e211]
[ 2889.278276] xe 0000:09:00.0: [drm] GT0: PF: VF1..VF7 provisioned with 3632267264 (3.38 GiB) LMEM
[ 2889.280504] xe 0000:09:00.0: [drm] GT0: PF: VF1..VF7 provisioned with 9324 GuC context IDs
[ 2889.281822] xe 0000:09:00.0: [drm] GT0: PF: VF1..VF7 provisioned with 36 GuC doorbell IDs
[ 2889.285426] xe 0000:09:00.0: [drm] GT1: PF: VF1..VF7 provisioned with 9325 GuC context IDs
[ 2889.289830] xe 0000:09:00.0: [drm] GT1: PF: VF1..VF7 provisioned with 36 GuC doorbell IDs
[ 2889.399155] xe 0000:09:00.1: [drm] Running in SR-IOV VF mode
[ 2889.400811] xe 0000:09:00.1: [drm] VF: migration not supported by this module version
[ 2889.491014] xe 0000:09:00.2: [drm] Running in SR-IOV VF mode
[ 2889.492027] xe 0000:09:00.2: [drm] VF: migration not supported by this module version
[ 2889.568781] xe 0000:09:00.3: [drm] Running in SR-IOV VF mode
[ 2889.569043] xe 0000:09:00.3: [drm] VF: migration not supported by this module version
[ 2889.646632] xe 0000:09:00.4: [drm] Running in SR-IOV VF mode
[ 2889.647545] xe 0000:09:00.4: [drm] VF: migration not supported by this module version
[ 2889.728498] xe 0000:09:00.5: [drm] Running in SR-IOV VF mode
[ 2889.729294] xe 0000:09:00.5: [drm] VF: migration not supported by this module version
[ 2889.794891] xe 0000:09:00.6: [drm] Running in SR-IOV VF mode
[ 2889.795687] xe 0000:09:00.6: [drm] VF: migration not supported by this module version
[ 2889.865483] xe 0000:09:00.7: [drm] Running in SR-IOV VF mode
[ 2889.865748] xe 0000:09:00.7: [drm] VF: migration not supported by this module version
[ 2889.927530] xe 0000:09:00.0: [drm] PF: Enabled 7 of 7 VFs
benny@MS-02-1:~/Desktop/video_demo$ 
cat /sys/bus/pci/devices/0000:09:00.0/sriov_totalvfs    # expect 7
```

## 2. Create the 7 VFs

```
echo 7 | sudo tee /sys/bus/pci/devices/0000:09:00.0/sriov_numvfs
lspci -nn | grep e211                                   # expect 09:00.0 through 09:00.7
sudo dmesg | grep -iE "xe.*(sriov|vf)" | tail -20       # expect "PF: Enabled 7 of 7 VFs"
```

To remove the VFs again: `echo 0 | sudo tee /sys/bus/pci/devices/0000:09:00.0/sriov_numvfs`

## 3. Check the Dolphin services

```
lsmod | grep ^dis_
/opt/DIS/sbin/dis_status
```

## 4. Lend a VF with Dolphin SmartIO

The PF (`09:00.0`) must stay bound to `xe` on this machine.

```
sudo /opt/DIS/sbin/smartio_tool add 09:00.1
sudo /opt/DIS/sbin/smartio_tool available 09:00.1 --unbind
sudo /opt/DIS/sbin/smartio_tool connect <nodeid>        # borrower nodes seen earlier: 28, 48
```

Repeat for `09:00.2` to `09:00.7` as needed.

## 5. Borrow on the other computer

```
sudo /opt/DIS/sbin/smartio_tool list
sudo /opt/DIS/sbin/smartio_tool borrow <deviceid>       # 100901 was 09:00.1
lspci | grep e211
```

To give a VF back:

```
sudo /opt/DIS/sbin/smartio_tool return <deviceid>       # on the borrower
sudo /opt/DIS/sbin/smartio_tool unavailable 09:00.1     # on this machine
```

## 6. Run the program on a borrowed GPU

On the borrower, once a borrowed VF shows up with `Kernel driver in use: xe` and has its own `/dev/dri/card*` and `renderD*` nodes, pick it with `DRI_PRIME`:

```
DRI_PRIME=pci-0000_03_02_1 ./video_broadcast -speedClient
```

`DRI_PRIME` is an environment variable that Mesa (the OpenGL and Vulkan driver) reads when the program starts. It tells Mesa which GPU to render on, instead of the system default. Setting it on the same line as the command applies it to that one run only, and nothing else on the system changes.

The value is the GPU's PCI address, with every `:` and `.` replaced by `_`, and a `pci-` prefix. PCI address `0000:03:02.1` becomes `pci-0000_03_02_1`. To use another slice, change the address, for example `pci-0000_03_02_3` for `0000:03:02.3`.

To find the addresses and confirm each slice has a render node:

```
lspci -D -d 8086:e211              # PCI addresses of the borrowed slices
ls /sys/bus/pci/devices/0000:03:02.1/drm    # should list cardN and renderDN
intel_gpu_top -L                   # GPUs the monitoring tool can see
```

Limits to know about:

- It only affects what Mesa draws. `video_broadcast` currently renders through SDL's default renderer, and the palm detection and OCR run on the CPU. So this selects the GPU that draws the window, not one that runs the inference.
- If the address is wrong or the slice has no working `xe` driver, Mesa falls back to the default GPU without an error. Check the address with the commands above first.

## Resolved: borrowing VFs now works

Two problems blocked borrowing. Both are fixed, and both borrowed VFs (`03:02.1` and `03:02.3` on the borrower) now run on `xe` with their own `/dev/dri/card*` and `renderD*` nodes.

### Fix 1: NTB prefetch window raised from 256 MB to 128 GB

`borrow` from node 28 failed with `line 1187: echo: write error: Input/output error`. The lender showed the device as borrowed and returned a second later. The cause was the Dolphin NTB function's prefetchable window, which was fixed at 256 MB in the switch image. The VF's 8 GB VRAM BAR could not be mapped through it (see the finding below).

The fix was a switch image with a 128 GB prefetch window. That is large enough for the unpatched 8 GB VF BAR, so the `xe` patch that shrinks the BAR to 256 MB is not needed.

### Fix 2: IOMMU turned off on the lending computer

After the larger window, the borrow succeeded, but the borrowed VFs did not work. The `xe` driver bound to them in SR-IOV VF mode and then failed:

```
xe 0000:03:02.1: [drm] *ERROR* GT0: VF: Failed to reset GuC state (-EPROTO)
xe 0000:03:02.1: [drm] *ERROR* Hardware reports unknown graphics version 0.00
```

No `/dev/dri` nodes appeared, and `intel_gpu_top -L` listed only the local GPU. Reading the VF's registers directly (`resource0`, offset `0xd8c`) returned `0xffffffff`, so reads never reached the GPU. The `dis_smartio` log also showed `IOMMU enabled`.

Turning the IOMMU off on the lending computer fixed it. Afterwards the borrower logs showed `Running in SR-IOV VF mode` followed by the VF's VRAM and engines, with no `-EPROTO` errors. The two VFs appear as `card0`/`renderD129` and `card2`/`renderD130`, and `intel_gpu_top -L` lists them as extra `8086:e211` devices.

To verify after a borrow:

```
lspci -nnk -d 8086:e211            # Kernel driver in use: xe
ls /dev/dri
intel_gpu_top -L
sudo dmesg | grep -E "xe 0000:03" | grep -iE "error|fail"    # should print nothing
```

Still untested: running a real workload on a VF. `intel_gpu_top -d drm:/dev/dri/card0` fails with `Failed to detect engines` because that tool only supports the i915 PMU, and `vainfo` failed with `vaInitialize` errors on both render nodes. Those may be tool and runtime limits, not VF faults, but they have not been ruled out.

