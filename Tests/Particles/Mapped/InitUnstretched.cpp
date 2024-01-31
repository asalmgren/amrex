#include <AMReX.H>
#include <AMReX_MultiFab.H>

using namespace amrex;

void
InitUnstretched (MultiFab& a_xyz_loc, Geometry& geom)
{
    AMREX_ALWAYS_ASSERT(a_xyz_loc.nComp() == 1 || a_xyz_loc.nComp() == AMREX_SPACEDIM);

    bool verbose = true;

    // z_comp is 0                for terrain-fitted and
    //           AMREX_SPACEDIM-1 for generalized mapped
    int z_comp = a_xyz_loc.nComp() - 1;

    auto problo   = geom.ProbLoArray();
    const auto dx = geom.CellSizeArray();

    for (MFIter mfi(a_xyz_loc); mfi.isValid(); ++mfi)
    {
        const Box&  tbx = mfi.tilebox();
        const Box& gtbx = mfi.grownnodaltilebox(1);

        auto tlo = lbound(tbx);
        auto thi = ubound(tbx);

        auto loc_arr = a_xyz_loc.array(mfi);

#if (AMREX_SPACEDIM == 2)
        ParallelFor(makeSlab(gtbx,1,0), [=] AMREX_GPU_DEVICE (int i, int , int ) noexcept
        {
            for (int j = tlo.y; j <= thi.y; j++)
            {
                loc_arr(i,j,0,z_comp) = problo[1] + static_cast<Real>(j) * dx[1];
            }
            loc_arr(i,tlo.y-1,0,z_comp) = 2.0 * loc_arr(i,tlo.y,0,z_comp) -  loc_arr(i,tlo.y+1,0,z_comp);
            loc_arr(i,thi.y+1,0,z_comp) = 2.0 * loc_arr(i,thi.y,0,z_comp) -  loc_arr(i,thi.y-1,0,z_comp);

            // Generalized mapped coordinates -- this is x
            if (z_comp > 0) {
                for (int j = tlo.y-1; j <= thi.y+1; j++)
                {
                    loc_arr(i,j,0,0) = problo[0] + static_cast<Real>(i)  * dx[0];
                }
            }

            if (verbose && i == 0) {
                for (int j = tlo.y-1; j <= thi.y+1; j++) {
                    if (j < 0) {
                        amrex::Print() << "INITIAL MAPPING AT " << IntVect(i,j) << " " << loc_arr(i,j,0,z_comp) << std::endl;
                    } else {
                        amrex::Print() << "INITIAL MAPPING AT " << IntVect(i,j) << " " << loc_arr(i,j,0,z_comp) <<
                                          " with dz = " << loc_arr(i,j,0,z_comp) - loc_arr(i,j-1,0,z_comp) << std::endl;
                    }
                } // j
            } // i
        });
#elif (AMREX_SPACEDIM == 3)
        ParallelFor(makeSlab(gtbx,2,0), [=] AMREX_GPU_DEVICE (int i, int j, int ) noexcept
        {
            for (int k = tlo.z-1; k <= thi.z+1; k++)
            {
                loc_arr(i,j,k,z_comp) = problo[2] + static_cast<Real>(k)  * dx[2];
            }
            loc_arr(i,j,tlo.z-1,z_comp) = 2.0 * loc_arr(i,j,tlo.z,z_comp) -  loc_arr(i,j,tlo.z+1,z_comp);
            loc_arr(i,j,thi.z+1,z_comp) = 2.0 * loc_arr(i,j,thi.z,z_comp) -  loc_arr(i,j,thi.z-1,z_comp);

            // Generalized mapped coordinates -- this is x
            if (z_comp > 0) {
                for (int k = tlo.z-1; k <= thi.z+1; k++)
                {
                    loc_arr(i,j,k,0) = problo[0] + static_cast<Real>(i)  * dx[0];
                    loc_arr(i,j,k,1) = problo[1] + static_cast<Real>(j)  * dx[1];
                }
            }

            if (verbose && i == 0 && j == 0) {
                for (int k = tlo.z-1; k <= thi.z+1; k++) {
                    if (k < 0) {
                        amrex::Print() << "INITIAL MAPPING AT " << IntVect(i,j,k) << " " << loc_arr(i,j,k,z_comp) <<  std::endl;
                    } else {
                        amrex::Print() << "INITIAL MAPPING AT " << IntVect(i,j,k) << " " << loc_arr(i,j,k,z_comp) <<
                                          " with dz = " << loc_arr(i,j,k,z_comp) - loc_arr(i,j,k-1,z_comp) << std::endl;
                    }
                } // j
            } // i
        });
#endif
    }
    a_xyz_loc.FillBoundary(geom.periodicity());
}
