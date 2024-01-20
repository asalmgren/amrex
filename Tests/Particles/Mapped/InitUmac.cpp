#include <AMReX.H>
#include <AMReX_MultiFab.H>

using namespace amrex;

enum struct ProbType {
    Annulus, Stretched, Hill
};

void
InitUmac_map (MultiFab* umac, const MultiFab& a_xyz_loc, Geometry& /*geom*/, Real vert_vel, ProbType /*prob_type*/)
{
    BL_PROFILE("InitUmac");

    // auto probhi = geom.ProbHi();
    // auto problo = geom.ProbLo();

    // For right now we just define a shear flow in x, i.e. in 3D: (u,v,w) = (u(z),0.0,0.0)
    //                                                      in 2D: (u,v)   = (u(y),0.0)

    umac[1].setVal(0.);
#if (AMREX_SPACEDIM == 3)
    umac[2].setVal(0.);
#endif

    // Decide between terrain-fittedn and fully mapped
    int zcomp = (a_xyz_loc.nComp() == 1) ? 0 : AMREX_SPACEDIM-1;

    for(MFIter mfi(umac[0]); mfi.isValid(); ++mfi)
    {
        const Box& tile_box  = mfi.growntilebox();
        auto height_arr = a_xyz_loc.array(mfi);
        auto umac_x_arr = umac[0].array(mfi);

        ParallelFor( tile_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // Physical location of x-face
#if (AMREX_SPACEDIM == 2)
            Real z = Real(0.5)*(height_arr(i,j,k,zcomp) + height_arr(i,j+1,k,zcomp));
#elif (AMREX_SPACEDIM == 3)
            Real z = Real(0.25)*(height_arr(i,j,k  ,zcomp) + height_arr(i,j+1,k  ,zcomp) +
                                 height_arr(i,j,k+1,zcomp) + height_arr(i,j+1,k+1,zcomp));
#endif

            // Normal velocity on x-face based on height at face center
            umac_x_arr(i,j,k,0) = Real(1.0) + Real(2.0) * z;

#if (AMREX_SPACEDIM == 2)
            if (i == 0) amrex::Print() << "UMAC AT " << IntVect(AMREX_D_DECL(i,j,k)) << " " << z << " " << umac_x_arr(i,j,k) << std::endl;
#elif (AMREX_SPACEDIM == 3)
            if (i == 0 && j == 0) amrex::Print() << "UMAC AT " << IntVect(AMREX_D_DECL(i,j,k)) << " " << z << " " << umac_x_arr(i,j,k) << std::endl;
#endif
        });
    }

    //
    // Add an upward velocity to the shear flow above
    //
    int zdir = AMREX_SPACEDIM-1;
    for(MFIter mfi(umac[zdir]); mfi.isValid(); ++mfi)
    {
        const Box& tile_box  = mfi.growntilebox();
        auto umac_vert_arr = umac[zdir].array(mfi);

        ParallelFor( tile_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            umac_vert_arr(i,j,k) = vert_vel;
        });
    }

}

void
InitUmac_reg (MultiFab* umac, Geometry& geom, Real vert_vel, ProbType /*prob_type*/)
{
    BL_PROFILE("InitUmac");

    // auto probhi = geom.ProbHi();
    // auto problo = geom.ProbLo();

    // For right now we just define a shear flow in x, i.e. in 3D: (u,v,w) = (u(z),0.0,0.0)
    //                                                      in 2D: (u,v)   = (u(y),0.0)

    umac[1].setVal(0.);
#if (AMREX_SPACEDIM == 3)
    umac[2].setVal(0.);
#endif
    const auto dx = geom.CellSizeArray();

    for(MFIter mfi(umac[0]); mfi.isValid(); ++mfi)
    {
        const Box& tile_box  = mfi.growntilebox();
        auto umac_x_arr = umac[0].array(mfi);

        ParallelFor( tile_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // Physical location of x-face
#if (AMREX_SPACEDIM == 2)
            Real z = (static_cast<Real>(j)+0.5) * dx[1];
#elif (AMREX_SPACEDIM == 3)
            Real z = (static_cast<Real>(k)+0.5) * dx[2];
#endif

            // Normal velocity on x-face based on height at face center
            umac_x_arr(i,j,k,0) = Real(1.0) + Real(2.0) * z;

#if (AMREX_SPACEDIM == 2)
            if (i == 0) amrex::Print() << "UMAC AT " << IntVect(AMREX_D_DECL(i,j,k)) << " " << z << " " << umac_x_arr(i,j,k) << std::endl;
#elif (AMREX_SPACEDIM == 3)
            if (i == 0 && j == 0) amrex::Print() << "UMAC AT " << IntVect(AMREX_D_DECL(i,j,k)) << " " << z << " " << umac_x_arr(i,j,k) << std::endl;
#endif
        });
    }

    //
    // Add an upward velocity to the shear flow above
    //
    int zdir = AMREX_SPACEDIM-1;
    for(MFIter mfi(umac[zdir]); mfi.isValid(); ++mfi)
    {
        const Box& tile_box  = mfi.growntilebox();
        auto umac_vert_arr = umac[zdir].array(mfi);

        ParallelFor( tile_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            umac_vert_arr(i,j,k) = vert_vel;
        });
    }

}
