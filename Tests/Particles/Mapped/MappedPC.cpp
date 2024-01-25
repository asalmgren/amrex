#include "MappedPC.H"

#include <AMReX_TracerParticle_mod_K.H>

using namespace amrex;

static constexpr int NSR = 2*AMREX_SPACEDIM;
static constexpr int NSI = AMREX_SPACEDIM;
static constexpr int NAR = 0;
static constexpr int NAI = 0;


void
MappedPC::
InitParticles (MultiFab& a_xyz_loc)
{
    BL_PROFILE("MappedPC::InitParticles");

    const int lev = 0;

    const auto domain = Geom(lev).Domain();

    auto dom_lo = lbound(domain);
    auto dom_hi = ubound(domain);

    for(MFIter mfi(a_xyz_loc); mfi.isValid(); ++mfi)
    {
        const Box& tile_box  = enclosedCells(mfi.tilebox());

        Gpu::HostVector<ParticleType> host_particles;
        std::array<Gpu::HostVector<ParticleReal>, NAR> host_real;
        std::array<Gpu::HostVector<int>, NAI> host_int;

        auto loc_arr = a_xyz_loc.array(mfi);

        for (IntVect iv = tile_box.smallEnd(); iv <= tile_box.bigEnd(); tile_box.next(iv))
        {

#if (AMREX_SPACEDIM == 2)
            int k = 0;
            if (iv[0] == 0 && iv[1] >= dom_lo.y && iv[1] <= dom_hi.y/2) {
#elif (AMREX_SPACEDIM == 3)
            int k = iv[2];
            if (iv[0] == 0 && iv[1] == 0 && iv[2] >= dom_lo.z && iv[2] <= dom_hi.z/2) {
#endif
                int i = iv[0];
                int j = iv[1];

                // This is the physical location of the center of the cell
                Real x = 0.25*( loc_arr(i  ,j,k,0) + loc_arr(i  ,j+1,k,0)
                               +loc_arr(i+1,j,k,0) + loc_arr(i+1,j+1,k,0));
                Real y = 0.25*( loc_arr(i  ,j,k,1) + loc_arr(i  ,j+1,k,1)
                               +loc_arr(i+1,j,k,1) + loc_arr(i+1,j+1,k,1));

                ParticleType p;
                p.id()  = ParticleType::NextID();
                p.cpu() = ParallelDescriptor::MyProc();
                p.pos(0) = x;
                p.pos(1) = y;

                p.rdata(MappedRealIdx::vx) =Real(0.0);
                p.rdata(MappedRealIdx::vy) = Real(0.0);

                p.idata(MappedIntIdx::i) = iv[0];  // particles carry their i-index
                p.idata(MappedIntIdx::j) = iv[1];  // particles carry their j-index

#if (AMREX_SPACEDIM == 2)
                amrex::Print() << "Particle at (x,y) OF " << iv << " " << x << " " << y << std::endl;
#elif (AMREX_SPACEDIM == 3)
                Real z = Real(0.125)*(loc_arr(i  ,j,k  ,2) + loc_arr(i  ,j+1,k  ,2) +
                                      loc_arr(i+1,j,k  ,2) + loc_arr(i+1,j+1,k  ,2) +
                                      loc_arr(i  ,j,k+1,2) + loc_arr(i  ,j+1,k+1,2) +
                                      loc_arr(i+1,j,k+1,2) + loc_arr(i+1,j+1,k+1,2));
                p.pos(2) = z;
                p.rdata(MappedRealIdx::vz) = Real(0.);
                p.idata(MappedIntIdx::k) = iv[2];  // particles carry their k-index

                amrex::Print() << "Particle at (x,y,z) OF " << iv << " " << x << " " << y << " " << z << std::endl;
#endif

                host_particles.push_back(p);
                for (int nr = 0; nr < NAR; ++nr)
                    host_real[nr].push_back(p.rdata(nr));
                for (int ni = 0; ni < NAI; ++ni)
                    host_int[ni].push_back(p.idata(ni));

           }
        }

            auto& particle_tile = DefineAndReturnParticleTile(lev, mfi.index(), mfi.LocalTileIndex());
            auto old_size = particle_tile.GetArrayOfStructs().size();
            auto new_size = old_size + host_particles.size();
            particle_tile.resize(new_size);

            Gpu::copyAsync(Gpu::hostToDevice,
                           host_particles.begin(),
                           host_particles.end(),
                           particle_tile.GetArrayOfStructs().begin() + old_size);

            auto& soa = particle_tile.GetStructOfArrays();
            for (int i = 0; i < NAR; ++i)
            {
                Gpu::copyAsync(Gpu::hostToDevice,
                               host_real[i].begin(),
                               host_real[i].end(),
                               soa.GetRealData(i).begin() + old_size);
            }

            for (int i = 0; i < NAI; ++i)
            {
                Gpu::copyAsync(Gpu::hostToDevice,
                               host_int[i].begin(),
                               host_int[i].end(),
                               soa.GetIntData(i).begin() + old_size);
            }

            Gpu::streamSynchronize();
    }
    RedistributeLocal();
}

/*
  /brief Uses midpoint method to advance particles using cell-centered velocity.
*/
void
MappedPC::AdvectWithUCC (MultiFab& vel_cc, int lev, Real dt, const MultiFab& a_xyz_loc)
{
    BL_PROFILE("MappedPC::AdvectWithCC()");
    Abort("Not implemented yet!");
    AMREX_ASSERT(lev >= 0 && lev < GetParticles().size());

    const auto plo = this->ParticleContainerBase::Geom(0).ProbLoArray();
    const auto dxi = this->ParticleContainerBase::Geom(0).InvCellSizeArray();

    for (int ipass = 0; ipass < 2; ipass++)
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (ParIterType pti(*this, lev); pti.isValid(); ++pti)
        {
            auto& ptile = ParticlesAt(lev, pti);
            auto& aos  = ptile.GetArrayOfStructs();
            const int n = aos.numParticles();
            auto *p_pbox = aos().data();

            const auto loc_arr = a_xyz_loc.array(pti);
            const auto vel_cc_arr = vel_cc.const_array(pti);

            ParallelFor(n, [=] AMREX_GPU_DEVICE (int i)
            {
                ParticleType& p = p_pbox[i];

                if (p.id() <= 0) { return; }

                ParticleReal v[AMREX_SPACEDIM];
                cic_interpolate_mapped(p, vel_cc_arr, loc_arr, v);

                if (ipass == 0)
                {
#if (AMREX_SPACEDIM == 2)
                    amrex::Print() << "FROM " << p.pos(0) << " " << p.pos(AMREX_SPACEDIM-1) << std::endl;
#elif (AMREX_SPACEDIM == 3)
                    amrex::Print() << "FROM " << p.pos(0) << " " << p.pos(1) << " " << p.pos(AMREX_SPACEDIM-1) << std::endl;
#endif
                    for (int dim=0; dim < AMREX_SPACEDIM; dim++)
                    {
                        p.rdata(dim) = p.pos(dim);
                        p.pos(dim) += static_cast<ParticleReal>(ParticleReal(0.5)*dt*v[dim]);
                    }
                    update_mapped_idata(p,plo,dxi,loc_arr);
                }
                else
                {
                    for (int dim=0; dim < AMREX_SPACEDIM; dim++)
                    {
                        p.pos(dim) = p.rdata(dim) + static_cast<ParticleReal>(dt*v[dim]);
                        p.rdata(dim) = v[dim];
                    }
                    update_mapped_idata(p,plo,dxi,loc_arr);

#if (AMREX_SPACEDIM == 2)
                    amrex::Print() << "TO   " << p.pos(0) << " " << p.pos(AMREX_SPACEDIM-1) << std::endl;
#elif (AMREX_SPACEDIM == 3)
                    amrex::Print() << "TO   " << p.pos(0) << " " << p.pos(1) << " " << p.pos(AMREX_SPACEDIM-1) << std::endl;
#endif
                }
            });
        } // ParIter
    } // ipass

    Redistribute();
}

/*
  /brief Uses midpoint method to advance particles using cell-centered velocity.
*/
void
MappedPC::AdvectWithUND (MultiFab& vel_nd, int lev, Real dt, const MultiFab& a_xyz_loc)
{
    BL_PROFILE("MappedPC::AdvectWithND()");
    AMREX_ASSERT(lev >= 0 && lev < GetParticles().size());

    const auto plo = this->ParticleContainerBase::Geom(0).ProbLoArray();
    const auto dxi = this->ParticleContainerBase::Geom(0).InvCellSizeArray();

    for (int ipass = 0; ipass < 2; ipass++)
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (ParIterType pti(*this, lev); pti.isValid(); ++pti)
        {
            auto& ptile = ParticlesAt(lev, pti);
            auto& aos  = ptile.GetArrayOfStructs();
            const int n = aos.numParticles();
            auto *p_pbox = aos().data();

            const auto loc_arr = a_xyz_loc.array(pti);
            const auto vel_nd_arr = vel_nd.const_array(pti);

            ParallelFor(n, [=] AMREX_GPU_DEVICE (int i)
            {
                ParticleType& p = p_pbox[i];

                if (p.id() <= 0) { return; }

                ParticleReal v[AMREX_SPACEDIM];

                cic_interpolate_nd_mapped(p, vel_nd_arr, loc_arr, v);

                if (ipass == 0)
                {
#if (AMREX_SPACEDIM == 2)
                    amrex::Print() << "FROM " << p.pos(0) << " " << p.pos(AMREX_SPACEDIM-1) << std::endl;
#elif (AMREX_SPACEDIM == 3)
                    amrex::Print() << "FROM " << p.pos(0) << " " << p.pos(1) << " " << p.pos(AMREX_SPACEDIM-1) << std::endl;
#endif
                    for (int dim=0; dim < AMREX_SPACEDIM; dim++)
                    {
                        p.rdata(dim) = p.pos(dim);
                        p.pos(dim) += static_cast<ParticleReal>(ParticleReal(0.5)*dt*v[dim]);
                    }
                    update_mapped_idata(p,plo,dxi,loc_arr);
                }
                else
                {
                    for (int dim=0; dim < AMREX_SPACEDIM; dim++)
                    {
                        p.pos(dim) = p.rdata(dim) + static_cast<ParticleReal>(dt*v[dim]);
                        p.rdata(dim) = v[dim];
                    }
#if (AMREX_SPACEDIM == 2)
                    amrex::Print() << "TO   " << p.pos(0) << " " << p.pos(AMREX_SPACEDIM-1) <<
                                      " WITH VEL " << v[0] << std::endl;
#elif (AMREX_SPACEDIM == 3)
                    amrex::Print() << "TO   " << p.pos(0) << " " << p.pos(1) << " " << p.pos(AMREX_SPACEDIM-1) <<
                                      " WITH VEL " << v[0] << std::endl;
#endif
                    update_mapped_idata(p,plo,dxi,loc_arr);
                }
            });
        } // ParIter
    } // ipass

    Redistribute();
}
