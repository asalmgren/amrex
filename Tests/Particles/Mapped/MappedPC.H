#ifndef MAPPED_PC_H_
#define MAPPED_PC_H_

#include <AMReX_Particles.H>

struct MappedRealIdx
{
    enum {
        vx = 0,
        vy, vz,
        ncomps
    };
};

struct MappedIntIdx
{
    enum {
        i = 0,
        j, k,
        ncomps
    };
};

struct MappedAssignor
{
    template <typename P>
    AMREX_GPU_HOST_DEVICE
    amrex::IntVect operator() (P const& p,
                               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& /*plo*/,
                               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& /*dxi*/,
                               const amrex::Box& /*domain*/) const noexcept
    {
        amrex::IntVect iv( AMREX_D_DECL(p.idata(0),
                                        p.idata(1),
                                        p.idata(2)));
        return iv;
    }
};

class MappedPC
    : public amrex::ParticleContainer<MappedRealIdx::ncomps, MappedIntIdx::ncomps, 0, 0,
                                      amrex::DefaultAllocator, MappedAssignor>
{

public:

    MappedPC (amrex::ParGDBBase* gdb)
        : amrex::ParticleContainer<MappedRealIdx::ncomps, MappedIntIdx::ncomps, 0, 0,
                                   amrex::DefaultAllocator, MappedAssignor>(gdb)
        {}

    MappedPC (const amrex::Geometry            & geom,
              const amrex::DistributionMapping & dmap,
              const amrex::BoxArray            & ba)
        : amrex::ParticleContainer<MappedRealIdx::ncomps, MappedIntIdx::ncomps, 0, 0,
                                   amrex::DefaultAllocator, MappedAssignor>(geom, dmap, ba)
        {}

    void InitParticles (amrex::MultiFab& a_xyz_loc);

    void AdvectWithUmac (amrex::MultiFab* umac, int level, amrex::Real dt,
                         const amrex::MultiFab& a_xyz_loc);
    void AdvectWithUCC  (amrex::MultiFab& u_cc, int level, amrex::Real dt,
                         const amrex::MultiFab& a_xyz_loc);
    void AdvectWithUND  (amrex::MultiFab& u_cc, int level, amrex::Real dt,
                         const amrex::MultiFab& a_xyz_loc);
    void RedistributeLocal (bool remove_neg=true)
    {
        const int lev_min = 0;
        const int lev_max = finestLevel();
        const int nGrow = 0;
        const int local = 1;
        Redistribute(lev_min, lev_max, nGrow, local, remove_neg);
    }

    void RedistributeGlobal (bool remove_neg=true)
    {
        const int lev_min = 0;
        const int lev_max = finestLevel();
        const int nGrow = 0;
        const int local = 0;
        Redistribute(lev_min, lev_max, nGrow, local, remove_neg);
    }


    /**
     * Given the vertices of a polygon (x,y), this function
     * returns true if the point(xp,yp) is inside the polygon.
     */
    bool PointInPolygon (amrex::Real x[4], amrex::Real y[4], amrex::Real xp, amrex::Real yp)
    {
        bool inside;
        int pos = 0;
        int neg = 0;
        for (int v = 0; v < 4; ++v) {
            int w = (v < 3) ? v+1 : 0;
            amrex::Real CP = (xp - x[v])*(y[w] - y[v]) - ( yp - y[v])*(x[w] - x[v]);
            if (CP > 0) { pos ++;
            } else { neg++;
            }
        }
        inside = ( (pos > 0) && (neg > 0) ) ? false : true;
        return inside;
    }

    template <typename P>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void update_mapped_idata (P& p,
                              amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
                              amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi,
                              const amrex::Array4<amrex::Real const>&  loc_arr)
    {
#if (AMREX_SPACEDIM == 2)
        amrex::IntVect iv( p.idata(0), p.idata(1) );
        amrex::Real x[5];
        amrex::Real y[5];
        int cnt = 0;
        int k = 0;
        for (int jj = 0; jj < 2; ++jj) {
            for (int ii = 0; ii < 2; ++ii) {
                // the polygon vertices are counterclockwise
                // i.e., (ii,jj), (ii+1,jj), (ii+1,jj+1), (ii,jj+1)
                int i = amrex::Math::abs( ii - jj);
                x[cnt] = loc_arr(iv[0] + i, iv[1] + jj, k, 0);
                y[cnt] = loc_arr(iv[0] + i, iv[1] + jj, k, 1);
            }
        }
        bool inside = PointInPolygon( x, y, p.pos(0), p.pos(1));

        if (inside) { return;
        } else {
            // Looping over neighboring cells
            for (int lo_j = -1; lo_j < 2; ++lo_j) {
            for (int lo_i = -1; lo_i < 2; ++lo_i) {
                if ( (lo_i == 0) && (lo_j == 0) ) continue;
                amrex::IntVect ivn( p.idata(0) + lo_i, p.idata(1) + lo_j);
                cnt = 0;
                // looping over vertices of the polygon to set x,y at each vertex
                for (int jj = 0; jj < 2; ++jj) {
                for (int ii = 0; ii < 2; ++ii) {
                    int i = amrex::Math::abs(ii - jj);
                    x[cnt] = loc_arr(ivn[0] + i, ivn[1] + jj, k, 0);
                    y[cnt] = loc_arr(ivn[0] + i, ivn[1] + jj, k, 1);
                }}
                inside = PointInPolygon( x, y, p.pos(0), p.pos(1));

                if (inside) {
                    p.idata(0) = p.idata(0) + lo_i;
                    p.idata(1) = p.idata(1) + lo_j;
                    return;
                }
            }}
        }

#elif (AMREX_SPACEDIM == 3)
        amrex::IntVect iv( int(amrex::Math::floor((p.pos(0)-plo[0])*dxi[0])),
                           int(amrex::Math::floor((p.pos(1)-plo[1])*dxi[1])),
                           p.idata(2) );

        amrex::Real lx = (p.pos(0)-plo[0])*dxi[0] - static_cast<amrex::Real>(iv[0]);
        amrex::Real ly = (p.pos(1)-plo[1])*dxi[1] - static_cast<amrex::Real>(iv[1]);
        auto zlo = loc_arr(iv[0]  ,iv[1]  ,iv[2]  ,2) * (1.0-lx) * (1.0-ly) +
                   loc_arr(iv[0]+1,iv[1]  ,iv[2]  ,2) *      lx  * (1.0-ly) +
                   loc_arr(iv[0]  ,iv[1]+1,iv[2]  ,2) * (1.0-lx) * ly +
                   loc_arr(iv[0]+1,iv[1]+1,iv[2]  ,2) *      lx  * ly;
        auto zhi = loc_arr(iv[0]  ,iv[1]  ,iv[2]+1,2) * (1.0-lx) * (1.0-ly) +
                   loc_arr(iv[0]+1,iv[1]  ,iv[2]+1,2) *      lx  * (1.0-ly) +
                   loc_arr(iv[0]  ,iv[1]+1,iv[2]+1,2) * (1.0-lx) * ly +
                   loc_arr(iv[0]+1,iv[1]+1,iv[2]+1,2) *      lx  * ly;

        if (p.pos(2) > zhi) {
            p.idata(2) += 1;
        } else if (p.pos(2) <= zlo) {
            p.idata(2) -= 1;
        }
#endif
    }

};

#endif
