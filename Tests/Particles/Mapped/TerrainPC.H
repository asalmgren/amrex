#ifndef TERRAIN_FITTED_PC_H_
#define TERRAIN_FITTED_PC_H_

#include <AMReX_Particles.H>

struct TerrainRealIdx
{
    enum {
        vx = 0,
        vy, vz,
        ncomps
    };
};

struct TerrainIntIdx
{
    enum {
        k = 0,
        ncomps
    };
};

struct TerrainAssignor
{
    template <typename P>
    AMREX_GPU_HOST_DEVICE
    amrex::IntVect operator() (P const& p,
                               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
                               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi,
                               const amrex::Box& domain) const noexcept
    {
#if (AMREX_SPACEDIM == 2)
        amrex::IntVect iv( int(amrex::Math::floor((p.pos(0)-plo[0])*dxi[0])),
                           p.idata(0) );
        iv[0] += domain.smallEnd()[0];
#elif (AMREX_SPACEDIM == 3)
        amrex::IntVect iv( int(amrex::Math::floor((p.pos(0)-plo[0])*dxi[0])),
                           int(amrex::Math::floor((p.pos(1)-plo[1])*dxi[1])),
                           p.idata(0) );
        iv[0] += domain.smallEnd()[0];
        iv[1] += domain.smallEnd()[1];
#endif
        return iv;
    }
};

class TerrainPC
    : public amrex::ParticleContainer<TerrainRealIdx::ncomps, TerrainIntIdx::ncomps, 0, 0,
                                      amrex::DefaultAllocator, TerrainAssignor>
{

public:

    TerrainPC (amrex::ParGDBBase* gdb)
        : amrex::ParticleContainer<TerrainRealIdx::ncomps, TerrainIntIdx::ncomps, 0, 0,
                                   amrex::DefaultAllocator, TerrainAssignor>(gdb)
        {}

    TerrainPC (const amrex::Geometry            & geom,
               const amrex::DistributionMapping & dmap,
               const amrex::BoxArray            & ba)
        : amrex::ParticleContainer<TerrainRealIdx::ncomps, TerrainIntIdx::ncomps, 0, 0,
                                   amrex::DefaultAllocator, TerrainAssignor>(geom, dmap, ba)
        {}

    void InitParticles (amrex::MultiFab& a_z_loc);

    void AdvectWithUmac (amrex::MultiFab* umac, int level, amrex::Real dt, const amrex::MultiFab& a_z_loc);
    void AdvectWithUCC  (amrex::MultiFab& u_cc, int level, amrex::Real dt, const amrex::MultiFab& a_z_loc);
    void AdvectWithUND  (amrex::MultiFab& u_cc, int level, amrex::Real dt, const amrex::MultiFab& a_z_loc);

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

    template <typename P>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void update_terrain_idata (P& p,
                               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
                               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi,
                               const amrex::Array4<amrex::Real const>&  height_arr)
    {
#if (AMREX_SPACEDIM == 2)
        amrex::IntVect iv( int(amrex::Math::floor((p.pos(0)-plo[0])*dxi[0])), p.idata(0) );

        amrex::Real lx = (p.pos(0)-plo[0])*dxi[0] - static_cast<amrex::Real>(iv[0]);
        auto ylo = height_arr(iv[0]  ,iv[1]  ,0) * (1.0-lx) +
                   height_arr(iv[0]+1,iv[1]  ,0) *      lx;
        auto yhi = height_arr(iv[0]  ,iv[1]+1,0) * (1.0-lx) +
                   height_arr(iv[0]+1,iv[1]+1,0) *      lx;

        if (p.pos(1) > yhi) {
            p.idata(0) += 1;
        } else if (p.pos(1) <= ylo) {
            p.idata(0) -= 1;
        }

#elif (AMREX_SPACEDIM == 3)
        amrex::IntVect iv( int(amrex::Math::floor((p.pos(0)-plo[0])*dxi[0])),
                           int(amrex::Math::floor((p.pos(1)-plo[1])*dxi[1])),
                           p.idata(0) );

        amrex::Real lx = (p.pos(0)-plo[0])*dxi[0] - static_cast<amrex::Real>(iv[0]);
        amrex::Real ly = (p.pos(1)-plo[1])*dxi[1] - static_cast<amrex::Real>(iv[1]);
        auto zlo = height_arr(iv[0]  ,iv[1]  ,iv[2]  ) * (1.0-lx) * (1.0-ly) +
                   height_arr(iv[0]+1,iv[1]  ,iv[2]  ) *      lx  * (1.0-ly) +
                   height_arr(iv[0]  ,iv[1]+1,iv[2]  ) * (1.0-lx) * ly +
                   height_arr(iv[0]+1,iv[1]+1,iv[2]  ) *      lx  * ly;
        auto zhi = height_arr(iv[0]  ,iv[1]  ,iv[2]+1) * (1.0-lx) * (1.0-ly) +
                   height_arr(iv[0]+1,iv[1]  ,iv[2]+1) *      lx  * (1.0-ly) +
                   height_arr(iv[0]  ,iv[1]+1,iv[2]+1) * (1.0-lx) * ly +
                   height_arr(iv[0]+1,iv[1]+1,iv[2]+1) *      lx  * ly;

        if (p.pos(2) > zhi) {
            p.idata(0) += 1;
        } else if (p.pos(2) <= zlo) {
            p.idata(0) -= 1;
        }
#endif
    }

};

#endif
