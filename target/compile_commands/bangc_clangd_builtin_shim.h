#ifndef BANGC_MCP_CLANGD_BUILTIN_SHIM_H
#define BANGC_MCP_CLANGD_BUILTIN_SHIM_H

#if defined(__BANGC_CLANGD__) || defined(__clang__)
#ifndef __BANG_ARCH__
#define __BANG_ARCH__ 372
#endif

#define __mlvm_read_mlu_sreg_taskidx() 0
#define __mlvm_read_mlu_sreg_taskidy() 0
#define __mlvm_read_mlu_sreg_taskidz() 0
#define __mlvm_read_mlu_sreg_taskid() 0
#define __mlvm_read_mlu_sreg_taskdimx() 1
#define __mlvm_read_mlu_sreg_taskdimy() 1
#define __mlvm_read_mlu_sreg_taskdimz() 1
#define __mlvm_read_mlu_sreg_taskdim() 1
#define __mlvm_read_mlu_sreg_clusterdim() 1
#define __mlvm_read_mlu_sreg_clusterid() 0
#define __mlvm_read_mlu_sreg_coreid() 0
#define __mlvm_read_mlu_sreg_coredim() 1
#define __mlvm_read_mlu_sreg_nramsize() 0
#define __mlvm_read_mlu_sreg_wramsize() 0
#define __mlvm_read_mlu_sreg_sramsize() 0
#define __mlvm_read_mlu_sreg_carrysize() 0
#define __mlvm_read_mlu_sreg_roleid() 0
#define __mlvm_read_mlu_sreg_resendflag() 0
#define __mlvm_mlu_arch() __BANG_ARCH__
#define __mlvm_sync(...) ((void)0)
#endif

#endif
