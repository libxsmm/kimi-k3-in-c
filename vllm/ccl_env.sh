# oneCCL over libfabric mlx (UCX) provider; source before python with backend "ccl"
source /swtools/intel/compiler/latest/env/vars.sh >/dev/null 2>&1
source /swtools/intel/ccl/2022.0/env/vars.sh
source /swtools/intel/mpi/latest/env/vars.sh
export CCL_ATL_TRANSPORT=ofi FI_PROVIDER=${FI_PROVIDER:-mlx} CCL_LOG_LEVEL=${CCL_LOG_LEVEL:-warn}
export CCL_KVS_IFACE=${CCL_KVS_IFACE:-enp50s0f0}
