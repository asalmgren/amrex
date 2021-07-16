#ifndef CNS_F_H_
#define CNS_F_H_

#include <AMReX_BLFort.H>

#ifdef __cplusplus
extern "C" {
#endif

    void cns_init_fort (const int* phys_bc_lo, const int* phys_bc_hi,
                        int, int, int, int, int, int, int,
                        const amrex_real* problo, const amrex_real* probhi);

    void cns_hypfill (BL_FORT_FAB_ARG_3D(state),
                      const int* dlo, const int* dhi,
                      const amrex_real* dx, const amrex_real* glo,
                      const amrex_real* time, const int* bc);

    void cns_denfill (BL_FORT_FAB_ARG_3D(state),
                      const int* dlo, const int* dhi,
                      const amrex_real* dx, const amrex_real* glo,
                      const amrex_real* time, const int* bc);

    void cns_nullfill (BL_FORT_FAB_ARG_3D(state),
                       const int* dlo, const int* dhi,
                       const amrex_real* dx, const amrex_real* glo,
                       const amrex_real* time, const int* bc);

#ifdef __cplusplus
}
#endif

#endif
