# Draft message to Dolphin support

Subject: eXpressWare 5.25.3: borrowed Intel Battlemage SR-IOV VF reads all-ones on the borrower

Hello,

We lend SR-IOV virtual functions of an Intel Arc Pro B60 (Battlemage, 8086:e211) over SmartIO and cannot use them on the borrower.

Setup
- Dolphin eXpressWare 5.25.3 (DIS_RELEASE_5_25_3_JUN_05_2026), Ubuntu 24.04, kernel 6.17.0-42-generic on all nodes, `intel_iommu=on iommu=pt`.
- Lender: node 16. GPU PF 09:00.0 (xe driver). VFs 09:00.1 to 09:00.7 created with sriov_numvfs=7. PF reports `IOVCtl: Enable+ MSE+`, "Enabled 7 of 7 VFs", and the VFs work locally under xe.
- Borrowers: nodes 28 and 48. NTB function BAR 2 is 128 GB with the current switch image.
- Lent with `smartio_tool add / available --unbind / connect / borrow`.

Symptoms on the borrower
- Borrow succeeds: the device appears as 03:02.N with BAR 0 16M and BAR 2 8G assigned, `Mem+`, `BusMaster-`. SmartIO logs "borrowed device", "MSI window setup" and "Computed DMA window size" (6 to 37 GiB, varies per borrow).
- xe binds and logs `Running in SR-IOV VF mode`, then `VF: Failed to reset GuC state (-EPROTO)` and `Hardware reports unknown graphics version 0.00`. No /dev/dri node is created.
- Reading BAR 0 of the borrowed VF (resource0 mmap, as root) returns 0xffffffff at offsets 0x0, 0x4 and 0x1901f8.

Questions
1. Does 5.25.3 support lending Intel Battlemage SR-IOV VFs (8086:e211)? Are there known issues with VFs or with GPUs?
2. Why would BAR 0 reads of a borrowed device return all-ones even though the borrow reports success? Is any additional step needed (a mode flag, a DMA or MMIO window setting, an IOMMU setting)?
3. How is the per-device DMA window ("Computed DMA window size") chosen, and why is it so much larger than the VF BARs? Could the window allocation fail partway through?
4. Could the 8 GB VF BAR 2 be the cause, and is there a supported way to limit it?

Logs available on request: lender and borrower dmesg, `lspci -vvv`, smartio_tool output.

Thanks,
Benny
