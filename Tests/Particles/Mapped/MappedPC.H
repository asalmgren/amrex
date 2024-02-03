
#include <AMReX_Particles.H>
#include <AMReX_TracerParticle_mod_K.H>

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

    // The xyz are in order (i,j,k) (i+1,j,k) (i,j+1,k) (i+1,j+1,k)
    // (i,j,k+1) (i+1,j,k+1) (i,j+1,k+1) (i+1,j+1,k+1)
    bool PointInHexahedron (amrex::Real x[8], amrex::Real y[8], amrex::Real z[8],
                       amrex::Real xp, amrex::Real yp, amrex::Real zp)
    {
        // look up for vertex order to take cross product and determine normal
        // 6 faces and 4 vertices per face
        amrex::Array2D<int, 0, 3, 0, 5> inode{4, 2, 6, 0, 7, 1, 5, 3,
                                              5, 0, 4, 1, 6, 3, 7, 2,
                                              2, 1, 3, 0, 7, 4, 6, 5};
        bool inside = false;
        // Define ray 1 : particle to lower corner
        amrex::Real r1x = x[0] - xp;
        amrex::Real r1y = y[0] - yp;
        amrex::Real r1z = z[0] - zp;
        // Define ray 2 : particle to upper corner
        amrex::Real r2x = x[7] - xp;
        amrex::Real r2y = y[7] - yp;
        amrex::Real r2z = z[7] - zp;

        int numfaces = 6;
        int numinside_counter = 0;
        for (int i = 0; i < numfaces; ++i) {
            amrex::Real nx = 0.5 * ( ( (y[inode(0,i)] - y[inode(1,i)]) * (z[inode(2,i)] - z[inode(3,i)]) )
                                 -   ( (z[inode(0,i)] - z[inode(1,i)]) * (y[inode(2,i)] - y[inode(3,i)]) ) );
            amrex::Real ny = 0.5 * ( ( (z[inode(0,i)] - z[inode(1,i)]) * (x[inode(2,i)] - x[inode(3,i)]) )
                                 -   ( (x[inode(0,i)] - x[inode(1,i)]) * (z[inode(2,i)] - z[inode(3,i)]) ) );
            amrex::Real nz = 0.5 * ( ( (x[inode(0,i)] - x[inode(1,i)]) * (y[inode(2,i)] - y[inode(3,i)]) )
                                 -   ( (y[inode(0,i)] - y[inode(1,i)]) * (x[inode(2,i)] - x[inode(3,i)]) ) );
            amrex::Real raydotn = (i%2 == 0) ? (r1x * nx + r1y * ny + r1z * nz)
                                             : (r2x * nx + r2y * ny + r2z * nz);
            if (raydotn < 0) {
                amrex::Print() << " iface : " << i << " outside \n";
                inside = false;
                return inside;
            }
            numinside_counter++;
        }
        if (numinside_counter == 6) inside = true;
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
                cnt++;
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
                    cnt++;
                }}
                inside = PointInPolygon( x, y, p.pos(0), p.pos(1));

                if (inside) {
                    p.idata(0) = p.idata(0) + lo_i;
                    p.idata(1) = p.idata(1) + lo_j;
                    return;
                }
            }}
        }

        if (!inside) amrex::Abort("particle has not been mapped in 2D!\n");
#elif (AMREX_SPACEDIM == 3)
        //amrex::IntVect iv( int(amrex::Math::floor((p.pos(0)-plo[0])*dxi[0])),
        //                   int(amrex::Math::floor((p.pos(1)-plo[1])*dxi[1])),
        //                   p.idata(2) );

        //amrex::Real lx = (p.pos(0)-plo[0])*dxi[0] - static_cast<amrex::Real>(iv[0]);
        //amrex::Real ly = (p.pos(1)-plo[1])*dxi[1] - static_cast<amrex::Real>(iv[1]);
        //auto zlo = loc_arr(iv[0]  ,iv[1]  ,iv[2]  ,2) * (1.0-lx) * (1.0-ly) +
        //           loc_arr(iv[0]+1,iv[1]  ,iv[2]  ,2) *      lx  * (1.0-ly) +
        //           loc_arr(iv[0]  ,iv[1]+1,iv[2]  ,2) * (1.0-lx) * ly +
        //           loc_arr(iv[0]+1,iv[1]+1,iv[2]  ,2) *      lx  * ly;
        //auto zhi = loc_arr(iv[0]  ,iv[1]  ,iv[2]+1,2) * (1.0-lx) * (1.0-ly) +
        //           loc_arr(iv[0]+1,iv[1]  ,iv[2]+1,2) *      lx  * (1.0-ly) +
        //           loc_arr(iv[0]  ,iv[1]+1,iv[2]+1,2) * (1.0-lx) * ly +
        //           loc_arr(iv[0]+1,iv[1]+1,iv[2]+1,2) *      lx  * ly;

        //if (p.pos(2) > zhi) {
        //    p.idata(2) += 1;
        //} else if (p.pos(2) <= zlo) {
        //    p.idata(2) -= 1;
        //}

        amrex::IntVect iv( p.idata(0), p.idata(1), p.idata(2) );
        amrex::Real x[8];
        amrex::Real y[8];
        amrex::Real z[8];

        int cnt = 0;
        for (int k = 0; k < 2; ++k) {
        for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            x[cnt] = loc_arr(iv[0] + i, iv[1] + j, iv[2] + k, 0);
            y[cnt] = loc_arr(iv[0] + i, iv[1] + j, iv[2] + k, 1);
            z[cnt] = loc_arr(iv[0] + i, iv[1] + j, iv[2] + k, 2);
            cnt++;
        }}}

        bool inside = PointInHexahedron (x, y, z, p.pos(0), p.pos(1), p.pos(2));
        if (inside) {
            return;
        } else {
            // looping over neighboring cells
            for (int lo_k = -1; lo_k <=1; ++lo_k){
            for (int lo_j = -1; lo_j <=1; ++lo_j){
            for (int lo_i = -1; lo_i <=1; ++lo_i){
                if ( (lo_i == 0) && (lo_j == 0) && lo_k == 0 ) continue;
                amrex::IntVect ivn(p.idata(0) + lo_i, p.idata(1) + lo_j, p.idata(2) + lo_k);
                cnt = 0;
                for (int k = 0; k < 2; ++k) {
                for (int j = 0; j < 2; ++j) {
                for (int i = 0; i < 2; ++i) {
                    x[cnt] = loc_arr(ivn[0] + i, ivn[1] + j, ivn[2] + k, 0);
                    y[cnt] = loc_arr(ivn[0] + i, ivn[1] + j, ivn[2] + k, 1);
                    z[cnt] = loc_arr(ivn[0] + i, ivn[1] + j, ivn[2] + k, 2);
                    cnt++;
                }}}
                inside = PointInHexahedron (x, y, z, p.pos(0), p.pos(1), p.pos(2));
                if (inside) {
                    p.idata(0) = p.idata(0) + lo_i;
                    p.idata(1) = p.idata(1) + lo_j;
                    p.idata(2) = p.idata(2) + lo_k;
                    if (p.idata(0) == -1) p.idata(0) = 31;
                    if (p.idata(1) == -1) p.idata(1) = 31;
                    if (p.idata(0) == 32) p.idata(0) = 0;
                    if (p.idata(1) == 32) p.idata(1) = 0;
                    return;
                }
            }}}
        }
        if (!inside) { 
        //    // looping over neighboring cells
        //    for (int lo_k = -2; lo_k <=2; ++lo_k){
        //    for (int lo_j = -2; lo_j <=2; ++lo_j){
        //    for (int lo_i = -2; lo_i <=2; ++lo_i){
        //        if ( (lo_i == 0) && (lo_j == 0) && lo_k == 0 ) continue;
        //        amrex::IntVect ivn(p.idata(0) + lo_i, p.idata(1) + lo_j, p.idata(2) + lo_k);
        //        cnt = 0;
        //        for (int k = 0; k < 2; ++k) {
        //        for (int j = 0; j < 2; ++j) {
        //        for (int i = 0; i < 2; ++i) {
        //            x[cnt] = loc_arr(ivn[0] + i, ivn[1] + j, ivn[2] + k, 0);
        //            y[cnt] = loc_arr(ivn[0] + i, ivn[1] + j, ivn[2] + k, 1);
        //            z[cnt] = loc_arr(ivn[0] + i, ivn[1] + j, ivn[2] + k, 2);
        //            cnt++;
        //        }}}
        //        amrex::Print() << " ngbr cnt : " << lo_i << " " << lo_j << " " << lo_k  << " " << ivn<< " " << "\n";
        //        for (int ii = 0; ii < 8; ++ii ) {amrex::Print() << " " << x[ii] << ", ";}
        //        amrex::Print() << "\n";
        //        for (int ii = 0; ii < 8; ++ii ) {amrex::Print() << " " << y[ii] << ", ";}
        //        amrex::Print() << "\n";
        //        for (int ii = 0; ii < 8; ++ii ) {amrex::Print() << " " << z[ii] << ", ";}
        //        amrex::Print() << "\n";
        //        inside = PointInHexahedron (x, y, z, p.pos(0), p.pos(1), p.pos(2));
        //        if (inside) {
        //            p.idata(0) = p.idata(0) + lo_i;
        //            p.idata(1) = p.idata(1) + lo_j;
        //            p.idata(2) = p.idata(2) + lo_k;
        //            if (p.idata(0) == -1) p.idata(0) = 31;
        //            if (p.idata(1) == -1) p.idata(1) = 31;
        //            if (p.idata(0) == 32) p.idata(0) = 0;
        //            if (p.idata(1) == 32) p.idata(1) = 0;
        //            return;
        //        }
        //    }}}
        //    amrex::Print() << " pos : " << p.pos(0) << " " << p.pos(1) << " " << p.pos(2)  << " iv " << iv << "\n";
            if (!inside) amrex::Abort("particle has not been mapped!\n");
        }
#endif
    }

};

