
#include <CNS.H>
#include <CNS_hydro_K.H>

#include <AMReX_EBFArrayBox.H>
#include <AMReX_MultiCutFab.H>

using namespace amrex;

void
CNS::compute_dSdt_box (const Box& bx, 
                       Array4<Real const>& sfab, 
                       Array4<Real      >& dsdtfab,
                       const std::array<FArrayBox*, AMREX_SPACEDIM>& flux)
{
    BL_PROFILE("CNS::compute_dSdt_regular_box()");

    const auto dx = geom.CellSizeArray();
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
            cns_slope_x(i, j, k, slope, q);
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
        {
            cns_slope_y(i, j, k, slope, q);
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
            cns_slope_z(i, j, k, slope, q);
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

        amrex::ParallelFor(bx, ncons,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            cns_flux_to_dudt(i, j, k, n, dsdtfab, AMREX_D_DECL(fxfab,fyfab,fzfab), dxinv);
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
