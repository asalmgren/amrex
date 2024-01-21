#include <AMReX.H>
#include <AMReX_MultiFab.H>

using namespace amrex;

enum struct ProbType {
    Annulus, Stretched, Hill
};

void
InitUCC_map (MultiFab& ucc, const MultiFab& a_xyz_loc, Geometry& geom, int flow_dir, Real vert_vel, ProbType prob_type)
{
    BL_PROFILE("InitUCC_map");

    auto probhi = geom.ProbHi();
    auto problo = geom.ProbLo();

    // Center of the annulus
    Real cx = 0.5 * (problo[0]+probhi[0]);
    Real cy = 0.5 * (problo[1]+probhi[1]);

    int zdir  = AMREX_SPACEDIM - 1;

    //
    // ANNULUS
    //
    if (prob_type == ProbType::Annulus) {

        AMREX_ALWAYS_ASSERT(a_xyz_loc.nComp() == AMREX_SPACEDIM);

        for (MFIter mfi(ucc); mfi.isValid(); ++mfi)
        {
            const Box& tile_box  = mfi.tilebox();

            auto loc_arr = a_xyz_loc.const_array(mfi);
            auto ucc_arr = ucc.array(mfi);

            ParallelFor( tile_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // Physical location of cell center
                Real x = 0.125 * (loc_arr(i,j  ,k  ,0) + loc_arr(i+1,j  ,k  ,0) +
                                  loc_arr(i,j+1,k  ,0) + loc_arr(i+1,j+1,k  ,0) +
                                  loc_arr(i,j  ,k+1,0) + loc_arr(i+1,j  ,k+1,0) +
                                  loc_arr(i,j+1,k+1,0) + loc_arr(i+1,j+1,k+1,0) );
                Real y = 0.125 * (loc_arr(i,j  ,k  ,1) + loc_arr(i+1,j  ,k  ,1) +
                                  loc_arr(i,j+1,k  ,1) + loc_arr(i+1,j+1,k  ,1) +
                                  loc_arr(i,j  ,k+1,1) + loc_arr(i+1,j  ,k+1,1) +
                                  loc_arr(i,j+1,k+1,1) + loc_arr(i+1,j+1,k+1,1) );

                Real theta;
                if (x == cx) {
                   theta = Real(0.);
                } else {
                   theta = atan((y-cy)/(x-cx));
                }

                Real    rad = sqrt( x*x + y*y);

                ucc_arr(i,j,k,0) =  rad*sin(theta);
                ucc_arr(i,j,k,1) = -rad*cos(theta);

#if (AMREX_SPACEDIM == 3)
                // Real z = loc_arr(i,j,k,2);
                ucc_arr(i,j,k,2) =  0.0;
#endif

                if (i == 0) amrex::Print() << "UCC AT " <<  IntVect(AMREX_D_DECL(i,j,k)) << " "
                                                        << RealVect(AMREX_D_DECL(ucc_arr(i,j,k,0),ucc_arr(i,j,k,1),ucc_arr(i,j,k,2)))
                                                        << std::endl;


            });
        }

    // Not the annulus
    } else {

        int zcomp = (a_xyz_loc.nComp() == 1) ? 0 : AMREX_SPACEDIM - 1;

        for (MFIter mfi(ucc); mfi.isValid(); ++mfi)
        {
            const Box& tile_box  = mfi.growntilebox();
            auto u_arr = ucc.array(mfi);
            auto loc_arr = a_xyz_loc.const_array(mfi);

            ParallelFor( tile_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // Physical location of cell center
#if (AMREX_SPACEDIM == 2)
                Real z = 0.5 * (loc_arr(i,j,k,zcomp) + loc_arr(i,j+1,k,zcomp));
#elif (AMREX_SPACEDIM == 3)
                Real z = 0.5 * (loc_arr(i,j,k,zcomp) + loc_arr(i,j,k+1,zcomp));
#endif

                // Horizontal velocity u(z)
                // Vertical   velocity constant
                u_arr(i,j,k,flow_dir) = Real(1.0) + Real(2.0) * z;
                u_arr(i,j,k,zdir    ) = vert_vel;

#if (AMREX_SPACEDIM == 2)
                if (i == 0) amrex::Print() << "UCC AT " << IntVect(AMREX_D_DECL(i,j,k)) << " " << z << " " <<
                                                        u_arr(i,j,k,flow_dir) << " " << u_arr(i,j,k,zdir) << std::endl;
#elif (AMREX_SPACEDIM == 3)
                if (i == 0 && j == 0) amrex::Print() << "UCC AT " << IntVect(AMREX_D_DECL(i,j,k)) << " " << z << " " <<
                                                        u_arr(i,j,k,flow_dir) <<" " << u_arr(i,j,k,zdir) <<  std::endl;
#endif
            });
        }
    }
}

void
InitUCC_reg (MultiFab& ucc, Geometry& geom, int flow_dir, Real vert_vel, ProbType /*prob_type*/)
{
    BL_PROFILE("InitUCC_reg");

    const auto problo = geom.ProbLo();
    const auto dx     = geom.CellSizeArray();

    int zdir = AMREX_SPACEDIM - 1;

    for (MFIter mfi(ucc); mfi.isValid(); ++mfi)
    {
        const Box& tile_box  = mfi.growntilebox();
        auto u_arr = ucc.array(mfi);

        ParallelFor( tile_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // Physical location of cell center
#if (AMREX_SPACEDIM == 2)
            Real z = problo[1] + (static_cast<Real>(j)+0.5) * dx[AMREX_SPACEDIM-1];
#elif (AMREX_SPACEDIM == 3)
            Real z = problo[2] + (static_cast<Real>(k)+0.5) * dx[AMREX_SPACEDIM-1];
#endif

            // Horizontal velocity u(z)
            // Vertical   velocity constant
            u_arr(i,j,k,flow_dir) = Real(1.0) + Real(2.0) * z;
            u_arr(i,j,k,zdir    ) = vert_vel;

#if (AMREX_SPACEDIM == 2)
                if (i == 0) amrex::Print() << "UCC AT " << IntVect(AMREX_D_DECL(i,j,k)) << " " << z << " " <<
                                                        u_arr(i,j,k,flow_dir) << " " << u_arr(i,j,k,zdir) << std::endl;
#elif (AMREX_SPACEDIM == 3)
                if (i == 0 && j == 0) amrex::Print() << "UCC AT " << IntVect(AMREX_D_DECL(i,j,k)) << " " << z << " " <<
                                                        u_arr(i,j,k,flow_dir) << " " << u_arr(i,j,k,zdir) <<  std::endl;
#endif
        });
    }
}
