# Splitting the Intel B60 (SR-IOV) after a reboot

The VFs do not survive a reboot. Boot kernel `6.17.0-42-generic` and use the kernel command line `intel_iommu=on iommu=pt xe.max_vfs=7` (already set in GRUB).

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

## Open issue

`borrow` from node 28 failed with `line 1187: echo: write error: Input/output error`. The lender showed the device as borrowed and returned a second later. Check the borrower's `dmesg` for BAR or DMA errors and confirm Above 4G Decoding is on.

## Optional: recreate the VFs automatically at boot

Find the real sysfs path, then add it to `/etc/sysfs.conf` (needs the `sysfsutils` package):

```
readlink -f /sys/bus/pci/devices/0000:09:00.0
echo 'devices/pci0000:00/0000:00:06.0/0000:01:00.0/0000:02:00.0/0000:03:00.0/0000:04:00.0/0000:05:00.0/0000:06:00.0/0000:07:00.0/0000:08:01.0/0000:09:00.0/sriov_numvfs = 7' | sudo tee -a /etc/sysfs.conf
```

The VFs may be created before the Dolphin modules load, so check the order in step 3 after a boot.

## In progress: shrink the VF BAR 2 from 8 GB to 256 MB

Dolphin cannot lend `09:00.1` because its VRAM BAR (VF BAR 2) is 8 GB. The card's VF Resizable BAR supports 256MB to 32GB, so 256 MB is the minimum (128 MB is not possible). Stock `xe` always picks the largest size, so it is patched.

- Patch: in `drivers/gpu/drm/xe/xe_pci_sriov.c`, `resize_vf_vram_bar()` uses `__ffs(sizes)` instead of `__fls(sizes)`.
- Source: Ubuntu `linux-hwe-6.17` 6.17.0-42 at `/tmp/xe-build/src` (in `/tmp`, so it may be gone after a reboot).
- Built module: `/tmp/xe-stripped.ko`, srcversion `F91CF99A13FE9EC146A220E`, built for `6.17.0-42-generic`.
- Status: built, **not installed yet**. `rmmod xe` fails because Cinnamon, Xorg and VS Code hold `/dev/dri`, so the module has to be installed and loaded at boot.

Install it (as root), then reboot:

```
cp /lib/modules/6.17.0-42-generic/kernel/drivers/gpu/drm/xe/xe.ko.zst /root/xe.ko.zst.orig
zstd -f /tmp/xe-stripped.ko -o /lib/modules/6.17.0-42-generic/kernel/drivers/gpu/drm/xe/xe.ko.zst
depmod -a 6.17.0-42-generic
update-initramfs -u -k 6.17.0-42-generic
reboot
```

After the reboot, check and create the VFs:

```
modinfo -F srcversion xe                                    # F91CF99A13FE9EC146A220E
echo 7 | sudo tee /sys/bus/pci/devices/0000:09:00.0/sriov_numvfs
lspci -vs 09:00.1 | grep "Memory at"                        # second line should say size=256M
sudo lspci -vvv -s 09:00.0 | grep -A1 "Virtual Resizable"   # current size: 256MB
```

Then retry the lend and borrow in steps 4 and 5.

Rollback if the B60 does not come up: boot kernel 6.14 from GRUB "Advanced options", then:

```
sudo cp /root/xe.ko.zst.orig /lib/modules/6.17.0-42-generic/kernel/drivers/gpu/drm/xe/xe.ko.zst
sudo depmod -a 6.17.0-42-generic
sudo update-initramfs -u -k 6.17.0-42-generic
```

A 6.17.0-42 kernel update or modules reinstall replaces the patched module. To prevent that: `sudo apt-mark hold linux-modules-6.17.0-42-generic`.

If `/tmp/xe-stripped.ko` is gone, rebuild it: download `linux-hwe-6.17_6.17.0.orig.tar.gz` and `linux-hwe-6.17_6.17.0-42.42.diff.gz` from `https://archive.ubuntu.com/ubuntu/pool/main/l/linux-hwe-6.17/`, unpack and apply the diff, make the patch above, link the headers into a combined tree in `/tmp/ktree` (everything from `/usr/src/linux-headers-6.17.0-42-generic` except `drivers`, which points at the unpacked source), then run `PATH=<xe dir>:$PATH make -C /tmp/ktree M=drivers/gpu/drm/xe KCFLAGS="-I<src>/include/trace" -j$(nproc) modules` and `strip --strip-debug` the result.

## Finding: the switch image's NTB prefetch window is 256 MB

Checked 2026-10-26 on this host:

- The Dolphin NTB function `03:00.1` has one 64-bit prefetchable BAR, `size=256M`. It is the only prefetchable BAR among the Dolphin and PLX functions.
- `sudo lspci -vvv -s 03:00.1 | grep -A3 -i Resizable` printed nothing, so there is no Resizable BAR capability. The 256 MB is fixed in the custom switch image. The host kernel, BIOS and `setpci` cannot enlarge it.
- The host is not the limit: the root bus has a 64-bit MMIO window of about 3.4 TB (`0xa000000000-0x3ffbfffffff`).
- The path to the B60 forwards about 90 GB of prefetchable space. The PF (32 GB BAR) plus 7 VFs (8 GB each) use about 88 GB of that.

Likely cause of the borrow failure (an inference; the borrower's `dmesg` has not been checked): the VF's 8 GB VRAM BAR cannot be mapped through a 256 MB window.

Plan:

- Get a switch image with a larger NTB prefetch window. At least 1 GB is safer. 8 GB or more would allow the unpatched VF.
- A 256 MB VF BAR (the patch above) exactly fills a 256 MB window, with nothing left for the 16 MB BAR 0, so it may still fail. It is worth a test.
- To confirm the cause, borrow a device with small BARs (for example an NVMe) using the same `add`, `available` and `borrow` steps. If that works, the window size is the problem.
- On the borrower, run this right after a failed borrow and read the result:

```
sudo dmesg | grep -iE "smartio|dis_|BAR|no space|can't assign"
```

- Compare the borrower's `dmesg | grep "root bus resource"` with this host's, since the machines are similar builds.

## Borrow error still open

The borrower (node 28) failed with `line 1187: echo: write error: Input/output error`. Once the VF BAR is 256 MB, retry and check the borrower's `dmesg` for BAR or DMA errors.
