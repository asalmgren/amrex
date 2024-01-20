#ifndef REGULAR_PC_H_
#define REGULAR_PC_H_

#include <AMReX_Particles.H>

struct RealIdx
{
    enum {
        vx = 0,
        vy, vz,
        ncomps
    };
};

struct IntIdx
{
    enum {
        k = 0,
        ncomps
    };
};

struct Assignor
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
                           int(amrex::Math::floor((p.pos(1)-plo[1])*dxi[1])));
        iv[0] += domain.smallEnd()[0];
        iv[1] += domain.smallEnd()[0];
#elif (AMREX_SPACEDIM == 3)
        amrex::IntVect iv( int(amrex::Math::floor((p.pos(0)-plo[0])*dxi[0])),
                           int(amrex::Math::floor((p.pos(1)-plo[1])*dxi[1])),
                           int(amrex::Math::floor((p.pos(2)-plo[2])*dxi[2])));
        iv[0] += domain.smallEnd()[0];
        iv[1] += domain.smallEnd()[1];
        iv[2] += domain.smallEnd()[2];
#endif
        return iv;
    }
};

class RegularPC
    : public amrex::ParticleContainer<RealIdx::ncomps, IntIdx::ncomps, 0, 0,
                                      amrex::DefaultAllocator, Assignor>
{

public:

    RegularPC (amrex::ParGDBBase* gdb)
        : amrex::ParticleContainer<RealIdx::ncomps, IntIdx::ncomps, 0, 0,
                                   amrex::DefaultAllocator, Assignor>(gdb)
        {}

    RegularPC (const amrex::Geometry            & geom,
               const amrex::DistributionMapping & dmap,
               const amrex::BoxArray            & ba)
        : amrex::ParticleContainer<RealIdx::ncomps, IntIdx::ncomps, 0, 0,
                                   amrex::DefaultAllocator, Assignor>(geom, dmap, ba)
        {}

    void InitParticles ();

    void AdvectWithUmac (amrex::MultiFab* umac, int level, amrex::Real dt);
    void AdvectWithUCC  (amrex::MultiFab& u_cc, int level, amrex::Real dt);
    void AdvectWithUND  (amrex::MultiFab& u_cc, int level, amrex::Real dt);

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

};

#endif
