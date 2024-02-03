#include <AMReX.H>
#include <AMReX_MultiFab.H>

using namespace amrex;

void
InitTorus (amrex::MultiFab& a_xyz_loc, amrex::Geometry& geom)
{
    const Real tpi = 2.0* amrex::Math::pi<Real>();

    auto domain = geom.Domain();
    auto probhi = geom.ProbHiArray();
    auto problo = geom.ProbLoArray();

    Real ilen = static_cast<Real>(domain.length(0));
    Real jlen = static_cast<Real>(domain.length(1));
    Real klen = static_cast<Real>(domain.length(2));

    // Center of the toroidal quadrilateral
    Real cx = 0.5 * (problo[0]+probhi[0]);
    Real cy = 0.5 * (problo[1]+probhi[1]);
    Real cz = 0.5 * (problo[2]+probhi[2]);

    // This is "delta eta" which is in the j direction
    Real d_eta = 1. / jlen;

    // This is "delta xi" which is in the i direction
    Real d_xi = 1. / ilen;

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
           
            loc_arr(i,j,k,0) = cx + (0.1 * xi + 0.1) * cos(tpi * eta);
            loc_arr(i,j,k,1) = cy + (0.1 * xi + 0.1) * sin(tpi * eta);
            //loc_arr(i,j,k,2) = cz + ( xi * 0.1 - 0.1 ) + zeta * 0.1 * (2. + 1. * xi); 3D torus
            loc_arr(i,j,k,2) = cz + zeta*0.2 - 0.1 + 0.3;
            amrex::Real rad = std::sqrt((loc_arr(i,j,k,0)-0.5)*(loc_arr(i,j,k,0)-0.5) + (loc_arr(i,j,k,1)-0.5)*(loc_arr(i,j,k,1)-0.5)) ;
        });

    }
    a_xyz_loc.FillBoundary(geom.periodicity());

}
