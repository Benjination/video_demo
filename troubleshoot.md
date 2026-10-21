# PCIe Expansion Box (eBox) on MS-02-1: Troubleshooting and Bring-up Notes

## Summary

MS-02-1 would not reach GRUB with the eBox attached. It now boots reliably enough to enumerate the whole chain after setting **Primary Display = IGFX** in the BIOS (previously Auto). The likely cause: with Auto, the BIOS tried to use the Intel B60 behind the PCIe switch as the boot display and stalled in its option ROM / BAR setup.

Status: first successful boot confirmed. Repeatability not yet verified (see checklist).

## Setup

- 3 servers connected through a Dolphin MX switch. The three communicate with no issues.
- MS-02-1 has the eBox attached in transparent mode, so the eBox devices should appear on MS-02-1's PCIe bus.
- eBox contents (working config): Intel B60 GPU (8086:e211) and audio function, three NVMe drives (SanDisk 15b7:5063, Samsung 144d:a80c, SanDisk 15b7:5045), Renesas uPD720201 USB 3.0 adapter.
- Host: MS-02-1, Ubuntu 24.04, kernel 6.14.0-37-generic, BIOS 1.03 (10/22/2025), board `ARWSA`.

## Timeline of findings

1. **Initial failure:** with the eBox attached (2-slot backplane with B60, 4/5-slot backplane with USB and NVMe), the host never reached GRUB.
2. **Resource exhaustion seen when it did boot:** the kernel logged `bridge window ... can't assign; no space` for the switch bridges. The switch was given only about 256 MB prefetchable and about 70 MB of 32-bit space, too small for a B60 BAR.
3. **Dolphin software:** the first installer run failed to build `DIS_MX` (the installer targets kernel 6.17, host has 6.14). A later build loaded `dis_*` modules. Observed: `dis_smartio` error `0x20000e07`, `Unable to connect to 4: error 28`, and a call trace in `dis_msq` (`scilib_module_init`).
4. **Removing the 2-slot backplane and an extra PCI card** let the 4/5-slot backplane enumerate fully (NVMe x3, USB). The resource errors went away.
5. **Fix:** BIOS **Primary Display = IGFX**. Full chain then enumerated including the B60. Firmware time was about 16 s, so this was not just a slow POST.

## Working BIOS settings

| Setting | Value |
|---|---|
| Primary Display | **IGFX**, set to always enabled (not Auto). One of the first two reboots failed to boot until it was changed from "IGFX" to "always enable IGFX" |
| Above 4G Decoding | TODO: record |
| Resizable BAR | TODO: record |
| CSM | TODO: record (expected off) |
| PCIe hot-plug | TODO: record |
| ASPM | TODO: record |
| Boot mode | TODO: record |

A BIOS update or CMOS reset reverts these. Save a BIOS profile or take screenshots.

## Working hardware config

- Backplanes attached: TODO: record which backplane(s) and slot of each card.
- B60 power: 6-pin and 2-pin joined into one 8-pin connector, from the eBox PSU on a dedicated lead (TODO: confirm what was done).
- Host cable and port used: TODO: record.
- The PCI card that came with the MS-02 was removed (it may compete for resources).
- Power-on order: eBox first, let it settle, then MS-02-1. TODO: test with the other two servers powered on (they were off during the first success).

## Expected result after boot

`lspci -tv` should show, behind root port `00:06.0`:

- Dolphin/PMC-Sierra switch chain (buses 01 to 10)
- `09:00.0` Intel `8086:e211` (B60) and `0a:00.0` audio (`8086:e2f7`)
- `0d:00.0` SanDisk NVMe, `0e:00.0` Samsung NVMe, `0f:00.0` Renesas USB 3.0, `10:00.0` SanDisk NVMe

Quick check:

```
lspci -nn | grep -E "e211|5063|a80c|0014|5045"
```

Bus numbers may shift if hardware changes.

## Known issues and warnings

- **Resizable BAR: resolved (2026-10-21 15:34 boot).** Earlier boots logged `Failed to resize BAR2 to 32768M (-ENOENT)` and `Small BAR device` (256 MB visible VRAM). After enabling Above 4G Decoding / Resizable BAR in the BIOS, `xe` reports a 32 GB BAR window and about 23.9 GB CPU-visible VRAM. Check with `journalctl -k -b | grep "xe 0000:09"`; the `VISIBLE VRAM` size should be `0x800000000`.
- **GPU stack (checked 2026-10-21):** `xe` driver on kernel 7.0.0-38; GuC 70.44.1, HuC 8.2.10, DMC v2.6 loaded OK; `linux-firmware` 20240318 (newest in Ubuntu 24.04 here); `mesa-vulkan-drivers` 25.2.8. Compute runtime (`intel-opencl-icd`, `libze-intel-gpu1`) not installed. The B60 enumerates with full VRAM accessible after Resizable BAR was enabled.

- Many correctable AER errors from the SanDisk NVMe `15b7:5063` (1379 this boot). Non-fatal, but suggests a marginal link. Check `sudo lspci -vv -s <dev> | grep -E "LnkSta|CESta"`, reseat the card, try another slot.
- **Dolphin eXpressWare 5.25.3 does not build on installed HWE kernel 7.0.0-38.** Installer `Dolphin_eXpressWare-Linux-x86_64-MX-DIS_RELEASE_5_25_3_JUN_05_2026.ubuntu24.04-hwe.sh` targets a 6.17 build kernel, but the attempt on 7.0.0-38 failed linking `dis_mx_dma.ko`: many duplicate `__pfx_*` symbol definitions from `dis_mx_dma.o` ("multiple definition", same object named on both sides). `objtool` also says the object already has `.retpoline_sites`, `.return_sites`, and `.call_sites`. The installer log is [DIS_install.log_bknh8l](./DIS_install.log_bknh8l); detailed build log was `/tmp/dis_build_deb.log`. This points to a Dolphin driver build/link compatibility or packaging issue; the log alone does not prove the kernel itself is incompatible. Ask Dolphin for a DIS 5.25.3 fix for Ubuntu 24.04 kernel 7.0.0-38 before relying on cluster services under this kernel. On the current 7.0.0-38 boot, `dis_*` modules are not loaded and `dis_mx_dma` is not present for this kernel.
- Earlier module load on 6.14 showed a call trace in `dis_msq` and SmartIO errors. Verify Dolphin service and inter-server communication after booting either kernel. The previously collected 6.14 build failed as well; its details are not retained in the current installer log.
- `linux-firmware` is dated March 2024. Battlemage blobs (`bmg_guc_70`, `bmg_huc`, `bmg_dmc`) are present and `xe` is loaded. After the B60 is stable, consider `sudo apt install --only-upgrade linux-firmware`.
- A missing OS driver does not explain the pre-GRUB hang: GRUB uses firmware only.

## If the hang returns

1. Wait up to 5 minutes (option ROM and BAR sizing through the switch can be slow).
2. Confirm Primary Display = IGFX and that BIOS settings were not reset.
3. Boot with only the 4/5-slot backplane, then add the B60 backplane.
4. Check B60 power (full 8-pin) and the host cable.
5. Try Above 4G Decoding and Resizable BAR on/off.
6. Check for BIOS, B60 firmware, and eBox/adapter firmware updates (`dis_firmwaretool` in `/opt/DIS/sbin`).
7. Collect: `lspci -tv`, `dmesg | grep -iE "pcieport|AER|no space|can't assign|BAR"`, BIOS version.

## Repeatability checklist

- [ ] 5 cold boots, each reaching GRUB and enumerating the full chain
- [ ] 5 warm reboots
- [ ] 1 boot with the other two servers powered on
- [ ] 1 boot after a power loss to the eBox (power-on order test)
- [ ] After each boot: `lspci` shows `e211`, three NVMe drives, and the USB controller

Record each result with the date and any anomalies below.

| Date | Type | Reached GRUB | Full chain | Notes |
|---|---|---|---|---|
| 2026-10-21 | first success | yes | yes | Primary Display = IGFX |
| 2026-10-21 | reboot attempt | no (1 of first 2) | n/a | Changed BIOS primary display from IGFX to **always enable IGFX**, then booted successfully |
| 2026-10-21 | HWE kernel test | yes | yes | Kernel 7.0.0-38 boots and B60 works; Dolphin eXpressWare 5.25.3 module build fails at `dis_mx_dma.ko` link |
