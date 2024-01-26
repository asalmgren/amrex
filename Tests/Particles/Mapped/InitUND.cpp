#include <AMReX.H>
#include <AMReX_MultiFab.H>

using namespace amrex;

enum struct ProbType {
    Annulus, Stretched, Hill
};

void
InitUND_map (MultiFab& und, const MultiFab& a_xyz_loc, Geometry& geom, int flow_dir, Real vert_vel, ProbType prob_type)
{
    BL_PROFILE("InitUND_map");

    auto probhi = geom.ProbHiArray();
    auto problo = geom.ProbLoArray();

    // Center of the annulus
    Real cx = 0.5 * (problo[0]+probhi[0]);
    Real cy = 0.5 * (problo[1]+probhi[1]);

    //
    // ANNULUS
    //
    if (prob_type == ProbType::Annulus) {

        // We only do this problem with fully mapped coordinates
        AMREX_ALWAYS_ASSERT(a_xyz_loc.nComp() == AMREX_SPACEDIM);

        for (MFIter mfi(und); mfi.isValid(); ++mfi)
        {
            const Box& tile_box  = mfi.tilebox();

            auto loc_arr = a_xyz_loc.const_array(mfi);
            auto und_arr = und.array(mfi);

            ParallelFor( tile_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // Physical location of cell center
                Real x = loc_arr(i,j,k,0);
                Real y = loc_arr(i,j,k,1);

                Real theta;
                if (x == cx) {
                   theta = Real(0.);
                } else {
                   theta = atan((y-cy)/(x-cx));
                }

                Real    rad = sqrt( (x-cx)*(x-cx) + (y-cy)*(y-cy));

                und_arr(i,j,k,0) =  rad*sin(theta);
                und_arr(i,j,k,1) = -rad*cos(theta);

#if (AMREX_SPACEDIM == 3)
                // Real z = loc_arr(i,j,k,2);
                und_arr(i,j,k,2) =  0.0;
#endif

                if (i == 0) amrex::Print() << "UND AT " <<  IntVect(AMREX_D_DECL(i,j,k)) << " "
                                                        << RealVect(AMREX_D_DECL(und_arr(i,j,k,0),und_arr(i,j,k,1),und_arr(i,j,k,2)))
                                                        << std::endl;


            });
        }

    } else {

        //
        // SHEAR FLOW
        //
        int zdir  = AMREX_SPACEDIM - 1;
        int zcomp = (a_xyz_loc.nComp() == 1) ? 0 : AMREX_SPACEDIM - 1;

        for (MFIter mfi(und); mfi.isValid(); ++mfi)
        {
            const Box& tile_box  = mfi.growntilebox();
            auto u_arr = und.array(mfi);
            auto loc_arr = a_xyz_loc.const_array(mfi);

            ParallelFor( tile_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // Physical location of node
                Real z = loc_arr(i,j,k,zcomp);

                // Horizontal velocity u(z)
                // Vertical   velocity constant
                u_arr(i,j,k,flow_dir) = Real(1.0) + Real(2.0) * z;
                u_arr(i,j,k,zdir    ) = vert_vel;

#if (AMREX_SPACEDIM == 2)
                if (i == 0) amrex::Print() << "UND AT " << IntVect(AMREX_D_DECL(i,j,k)) << " " << z
                                           << " " << u_arr(i,j,k,flow_dir) << " " << u_arr(i,j,k,zdir) << std::endl;
#elif (AMREX_SPACEDIM == 3)
                if (i == 0 && j == 0) amrex::Print() << "UND AT " << IntVect(AMREX_D_DECL(i,j,k)) << " " << z
                                                     << " " << u_arr(i,j,k,flow_dir) << " " << u_arr(i,j,k,zdir) << std::endl;
#endif
            });
        }
    }
}

void
InitUND_reg (MultiFab& und, Geometry& geom, int flow_dir,  Real vert_vel, ProbType /*prob_type*/)
{
    BL_PROFILE("InitUND_reg");

    const auto problo = geom.ProbLoArray();
    const auto dx     = geom.CellSizeArray();

    int zdir  = AMREX_SPACEDIM - 1;

    //
    // SHEAR FLOW
    //
    for (MFIter mfi(und); mfi.isValid(); ++mfi)
    {
        const Box& tile_box  = mfi.growntilebox();
        auto u_arr = und.array(mfi);

        ParallelFor( tile_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // Physical location of node
#if (AMREX_SPACEDIM == 2)
            Real z = problo[zdir] + static_cast<Real>(j) * dx[AMREX_SPACEDIM-1];
#elif (AMREX_SPACEDIM == 3)
            Real z = problo[zdir] + static_cast<Real>(k) * dx[AMREX_SPACEDIM-1];
#endif

            // Horizontal velocity u(z)
            // Vertical   velocity constant
            u_arr(i,j,k,flow_dir) = Real(1.0) + Real(2.0) * z;
            u_arr(i,j,k,zdir    ) = vert_vel;

#if (AMREX_SPACEDIM == 2)
            if (i == 0) amrex::Print() << "UND AT " << IntVect(AMREX_D_DECL(i,j,k)) << " " << z
                                       << " " << u_arr(i,j,k,flow_dir) << " " << u_arr(i,j,k,zdir) << std::endl;
#elif (AMREX_SPACEDIM == 3)
            if (i == 0 && j == 0) amrex::Print() << "UND AT " << IntVect(AMREX_D_DECL(i,j,k)) << " " << z
                                                 << " " << u_arr(i,j,k,flow_dir) << " " << u_arr(i,j,k,zdir) << std::endl;
#endif
        });
    }
}
