#include <AMReX.H>
#include <AMReX_MultiFab.H>

using namespace amrex;

void
InitTorus (amrex::MultiFab& a_xyz_loc, amrex::Geometry& geom)
{
    const Real tpi = Real(2.0) * amrex::Math::pi<Real>();

    auto domain = geom.Domain();
    auto probhi = geom.ProbHiArray();
    auto problo = geom.ProbLoArray();

    Real ilen = static_cast<Real>(domain.length(0));
    Real jlen = static_cast<Real>(domain.length(1));
    Real klen = static_cast<Real>(domain.length(2));

    // Center of the annulus
    Real cx = 0.5 * (problo[0]+probhi[0]);
    Real cy = 0.5 * (problo[1]+probhi[1]);
    Real cz = 0.5 * (problo[2]+probhi[2]);

    // This is "delta xi" which is in the i direction
    Real d_xi = tpi / ilen;

    // This is "delta eta" which is in the j direction
    Real d_eta = 1. / jlen;

    // This is "delta zeta" which is in the k direction
    Real d_zeta = 1. / klen;

    // loc_arr is nodal so no offset
    for(MFIter mfi(a_xyz_loc); mfi.isValid(); ++mfi)
    {
        const Box& gtbx = mfi.growntilebox();
        amrex::Print()  << " grown tile box : " << gtbx << "\n";
        auto loc_arr = a_xyz_loc.array(mfi);

        ParallelFor(gtbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real xi     = (static_cast<int>(i)) * d_xi;
            Real eta    = (static_cast<int>(j)) * d_eta;
            Real zeta   = (static_cast<int>(k)) * d_zeta;

            loc_arr(i,j,k,0) = (0.1 * eta + 0.1) * cos(xi);
            loc_arr(i,j,k,0) = (0.1 * eta + 0.1) * sin(xi);
            loc_arr(i,j,k,2) = -0.1 * ( 1 + eta ) + zeta * 0.1 * (2 + 2 * eta);
        });

    }
    a_xyz_loc.FillBoundary(geom.periodicity());

}
