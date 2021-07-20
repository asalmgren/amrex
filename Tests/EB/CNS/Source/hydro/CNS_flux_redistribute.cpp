
#include <CNS.H>
#include <CNS_hydro_K.H>
#include <CNS_hydro_eb_K.H>

#include <AMReX_EBFluxRegister.H>
#include <AMReX_YAFluxRegister.H>
#include <AMReX_EBFArrayBox.H>
#include <AMReX_MultiCutFab.H>
#include <AMReX_EBMultiFabUtil_3D_C.H>

using namespace amrex;

void
CNS::cns_flux_redistribute (const Box& bx,
                            Array4<Real       const> const& q,
                            Array4<Real            > const& dqdt,
                            Array4<Real            > const& divc,
                            Array4<Real            > const& optmp,
                            Array4<Real            > const& delm,
                            Array4<Real            > const& redistwgt,
                            Array4<Real       const> const& vfrac,
                            Array4<EBCellFlag const> const& flag,
                            int as_crse,
                            Array4<Real            > const& rr_drho_crse,
                            Array4<int             > const& rr_flag_crse,
                            int as_fine,
                            Array4<Real            > const& dm_as_fine,
                            Array4<int        const> const& levmsk,
                            Real dt)
{
    const Box& bxg1 = amrex::grow(bx,1);
    const Box& bxg2 = amrex::grow(bx,2);

    Real reredistribution_threshold = amrex_eb_get_reredistribution_threshold();

    int covered_by_fine   = YAFluxRegister::fine_cell;
    int levmsk_notcovered = 2;

    int bx_ilo = bx.smallEnd()[0];
    int bx_ihi = bx.bigEnd()[0];
    int bx_jlo = bx.smallEnd()[1];
    int bx_jhi = bx.bigEnd()[1];
    int bx_klo = bx.smallEnd()[2];
    int bx_khi = bx.bigEnd()[2];

    amrex::ParallelFor(bxg2,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        if (eb_weights_type == 0)
            { redistwgt(i,j,k) = 1.0; }
        else if (eb_weights_type == 1)
            { redistwgt(i,j,k) = q(i,j,k,QRHO)*( q(i,j,k,QEINT) +
                            0.5*(q(i,j,k,QU)*q(i,j,k,QU) + q(i,j,k,QV)*q(i,j,k,QV) + q(i,j,k,QW)*q(i,j,k,QW)) ); }
        else if (eb_weights_type == 2)
            { redistwgt(i,j,k) = q(i,j,k,QRHO); }
        else if (eb_weights_type == 3)
            { redistwgt(i,j,k) = vfrac(i,j,k); }
    });

    amrex::ParallelFor(bxg1, NEQNS,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        if (flag(i,j,k).isSingleValued())
        {
            Real vtot(0.);
            Real divnc(0.);
            for (int kk = -1; kk <= 1; kk++)
             for (int jj = -1; jj <= 1; jj++)
              for (int ii = -1; ii <= 1; ii++)
                  if ( (ii != 0 || jj != 0 || kk != 0) && flag(i,j,k).isConnected(i+ii,j+jj,k+kk) )
                  {
                      vtot  += vfrac(i+ii,j+jj,k+kk);
                      divnc += vfrac(i+ii,j+jj,k+kk)*divc(i+ii,j+jj,k+kk,n);
                  }
            divnc /= vtot;
            optmp(i,j,k,n) = (1.0-vfrac(i,j,k))*(divnc-divc(i,j,k,n));
            delm(i,j,k,n) = -vfrac(i,j,k)*optmp(i,j,k,n);
        } else {
            delm(i,j,k,n) = 0.;
        }
    });

    amrex::ParallelFor(bxg1, NEQNS,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        bool valid_dst_cell;
        if (flag(i,j,k).isSingleValued())
        {
            Real wtot = 0.;
            for (int kk = -1; kk <= 1; kk++)
             for (int jj = -1; jj <= 1; jj++)
              for (int ii = -1; ii <= 1; ii++)
                  if ( (ii != 0 || jj != 0 || kk != 0) && flag(i,j,k).isConnected(i+ii,j+jj,k+kk) )
                  {
                      wtot += vfrac(i+ii,j+jj,k+kk)*redistwgt(i+ii,j+jj,k+kk);
                  }
            wtot = 1.0 / wtot;

            bool as_crse_crse_cell    = false;
            bool as_crse_covered_cell = false;
            if (as_crse)
            {
               as_crse_crse_cell = (
                ( (i >= bx_ilo) && (i <= bx_ihi) && (j >= bx_jlo) && (j <= bx_jhi) && (k >= bx_klo) && (k <= bx_khi) ) &&
                  (rr_flag_crse(i,j,k) == YAFluxRegister::crse_fine_boundary_cell) );
               as_crse_covered_cell = (rr_flag_crse(i,j,k) == covered_by_fine);
            }

            bool as_fine_valid_cell = false;  // valid cells near box boundary
            bool as_fine_ghost_cell = false;  // ghost cells just outside valid region
            if (as_fine)
            {
               as_fine_valid_cell =
                ( (i >= bx_ilo) && (i <= bx_ihi) && (j >= bx_jlo) && (j <= bx_jhi) && (k >= bx_klo) && (k <= bx_khi) );
               as_fine_ghost_cell = (levmsk(i,j,k) == levmsk_notcovered); // not covered by other grids
            }

            for (int kk = -1; kk <= 1; kk++)
             for (int jj = -1; jj <= 1; jj++)
              for (int ii = -1; ii <= 1; ii++)
                  if ( (ii != 0 || jj != 0 || kk != 0) && flag(i,j,k).isConnected(i+ii,j+jj,k+kk) )
                  {
                      int iii = i + ii;
                      int jjj = j + jj;
                      int kkk = k + kk;

                      Real drho = delm(i,j,k,n)*wtot*redistwgt(iii,jjj,kkk);
                      optmp(iii,jjj,kkk,n) += drho;

                      valid_dst_cell = ( (iii >= bx_ilo) && (iii <= bx_ihi) &&
                                         (jjj >= bx_jlo) && (jjj <= bx_jhi) &&
                                         (kkk >= bx_klo) && (kkk <= bx_khi) );

                      if (as_crse_crse_cell)
                      {
                         if ( (rr_flag_crse(iii,jjj,kkk) == covered_by_fine) &&
                              (vfrac(i,j,k) > reredistribution_threshold) )
                         {
                            rr_drho_crse(i,j,k,n) += dt*drho*(vfrac(iii,jjj,kkk)/vfrac(i,j,k));
                         }
                      }

                      if (as_crse_covered_cell)
                      {
                          if (valid_dst_cell)
                          {
                             if ( (rr_flag_crse(iii,jjj,kkk) == YAFluxRegister::crse_fine_boundary_cell) &&
                                         (vfrac(iii,jjj,kkk) > reredistribution_threshold) )
                             {
                                // the recipient is a crse/fine boundary cell
                                rr_drho_crse(iii,jjj,kkk,n) = rr_drho_crse(iii,jjj,kkk,n) - dt*drho;
                             }
                          }
                      }

                      if (as_fine_valid_cell)
                      {
                         if (!valid_dst_cell)
                             dm_as_fine(iii,jjj,kkk,n) = dm_as_fine(iii,jjj,kkk,n) + dt*drho*vfrac(iii,jjj,kkk);
                      }

                      if (as_fine_ghost_cell)
                      {
                         if (valid_dst_cell)
                             dm_as_fine(i,j,k,n) = dm_as_fine(i,j,k,n) - dt*drho*vfrac(iii,jjj,kkk);
                      }

                  } // isConnected
        } // isSingleValued
    });

    amrex::ParallelFor(bx, NEQNS,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        if (!flag(i,j,k).isCovered())
            dqdt(i,j,k,n) = divc(i,j,k,n) + optmp(i,j,k,n);
    });
}
