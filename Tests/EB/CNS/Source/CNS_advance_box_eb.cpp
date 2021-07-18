
#include <CNS.H>
#include <CNS_hydro_K.H>
#include <CNS_hydro_eb_K.H>

#include <AMReX_EBFArrayBox.H>
#include <AMReX_MultiCutFab.H>

#if (AMREX_SPACEDIM == 2)
#include <AMReX_EBMultiFabUtil_2D_C.H>
#elif (AMREX_SPACEDIM == 3)
#include <AMReX_EBMultiFabUtil_3D_C.H>
#endif

using namespace amrex;

void
CNS::compute_dSdt_box_eb (const Box& bx,
                          Array4<Real const>& sfab,
                          Array4<Real      >& dsdtfab,
                          const std::array<FArrayBox*, AMREX_SPACEDIM>& flux,
                          Array4<Real const>& vfrac,
                          AMREX_D_DECL(Array4<Real const> const& apx,
                                       Array4<Real const> const& apy,
                                       Array4<Real const> const& apz),
                          AMREX_D_DECL(Array4<Real const> const& fcx,
                                       Array4<Real const> const& fcy,
                                       Array4<Real const> const& fcz),
                          Array4<EBCellFlag const> flag,
                          Array4<int        const> ccm)
{
    BL_PROFILE("CNS::compute_dSdt_box_eb()");

    const auto dxinv = geom.InvCellSizeArray();
    const int ncomp = NUM_STATE;
    const int neqns = 5;
    const int ncons = 7;
    const int nprim = 8;

    FArrayBox qtmp, slopetmp;

    Parm const* lparm = d_parm;

    AMREX_D_TERM(auto const& fxfab = flux[0]->array();,
                 auto const& fyfab = flux[1]->array();,
                 auto const& fzfab = flux[2]->array(););

    AMREX_D_TERM(Array4<Real const> const& fx_arr = flux[0]->const_array();,
                 Array4<Real const> const& fy_arr = flux[1]->const_array();,
                 Array4<Real const> const& fz_arr = flux[2]->const_array());

        const Box& bxg2 = amrex::grow(bx,2);
        qtmp.resize(bxg2, nprim);
        Elixir qeli = qtmp.elixir();
        auto const& q = qtmp.array();

        amrex::ParallelFor(bxg2,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            cns_ctoprim(i, j, k, sfab, q, *lparm);
        });

        const Box& bxg1 = amrex::grow(bx,1);
        slopetmp.resize(bxg1,neqns);
        Elixir slopeeli = slopetmp.elixir();
        auto const& slope = slopetmp.array();

        // x-direction
        int cdir = 0;
        const Box& xslpbx = amrex::grow(bx, cdir, 1);
        amrex::ParallelFor(xslpbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            cns_slope_eb_x(i, j, k, slope, q, flag, plm_iorder, plm_theta);
        });
        const Box& xflxbx = amrex::surroundingNodes(bx,cdir);
        amrex::ParallelFor(xflxbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            cns_riemann_x(i, j, k, fxfab, slope, q, *lparm);
            for (int n = neqns; n < ncons; ++n) fxfab(i,j,k,n) = Real(0.0);
        });

        // y-direction
        cdir = 1;
        const Box& yslpbx = amrex::grow(bx, cdir, 1);
        amrex::ParallelFor(yslpbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        { cns_slope_eb_y(i, j, k, slope, q, flag, plm_iorder, plm_theta);
        });
        const Box& yflxbx = amrex::surroundingNodes(bx,cdir);
        amrex::ParallelFor(yflxbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            cns_riemann_y(i, j, k, fyfab, slope, q, *lparm);
            for (int n = neqns; n < ncons; ++n) fyfab(i,j,k,n) = Real(0.0);
        });

        // z-direction
        cdir = 2;
        const Box& zslpbx = amrex::grow(bx, cdir, 1);
        amrex::ParallelFor(zslpbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            cns_slope_eb_z(i, j, k, slope, q, flag, plm_iorder, plm_theta);
        });
        const Box& zflxbx = amrex::surroundingNodes(bx,cdir);
        amrex::ParallelFor(zflxbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            cns_riemann_z(i, j, k, fzfab, slope, q, *lparm);
            for (int n = neqns; n < ncons; ++n) fzfab(i,j,k,n) = Real(0.0);
        });

        // don't have to do this, but we could
        qeli.clear(); // don't need them anymore
        slopeeli.clear();

        // "false" in the argument list means the data is not already on centroids
        amrex::ParallelFor(bx, ncons,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            // This does the divergence but not the redistribution -- we will do that
            //      separately
            eb_compute_divergence(i,j,k,n,dsdtfab,AMREX_D_DECL(fx_arr,fy_arr,fz_arr),
                                  ccm, flag, vfrac, AMREX_D_DECL(apx,apy,apz),
                                  AMREX_D_DECL(fcx,fcy,fcz), dxinv, false);
        });

        if (gravity != Real(0.0)) {
            const Real g = gravity;
            const int irho = Density;
            const int imz = Zmom;
            const int irhoE = Eden;
            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                dsdtfab(i,j,k,imz) += g * sfab(i,j,k,irho);
                dsdtfab(i,j,k,irhoE) += g * sfab(i,j,k,imz);
            });
        }
}
