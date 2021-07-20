#ifndef CNS_DIFFUSION_EB_K_H_
#define CNS_DIFFUSION_EB_K_H_

#include "CNS_index_macros.H"
#include "CNS_parm.H"
#include <AMReX_FArrayBox.H>
#include <AMReX_CONSTANTS.H>
#include <cmath>

AMREX_GPU_DEVICE
inline
void
cns_diffcoef_eb (int i, int j, int k,
                 amrex::Array4<amrex::Real const> const& q,
                 amrex::Array4<amrex::EBCellFlag const> const& flag,
                 amrex::Array4<amrex::Real> const& coefs,
                 Parm const& parm) noexcept
{
    using amrex::Real;

     bool cov = flag(i,j,k).isCovered();
     coefs(i,j,k,CETA) = cov ? -1.e10 : parm.C_S * std::sqrt(q(i,j,k,QTEMP)) * q(i,j,k,QTEMP) / (q(i,j,k,QTEMP)+parm.T_S);
     coefs(i,j,k,CXI)  = cov ? -1.e10 : Real(0.0);
     coefs(i,j,k,CLAM) = cov ? -1.e10 : coefs(i,j,k,CETA)*parm.cp/parm.Pr;
}

AMREX_GPU_DEVICE
inline
void
cns_constcoef_eb (int i, int j, int k,
                 amrex::Array4<amrex::EBCellFlag const> const& flag,
                 amrex::Array4<amrex::Real> const& coefs,
                 Parm const& parm) noexcept
{
    using amrex::Real;

     bool cov = flag(i,j,k).isCovered();
     coefs(i,j,k,CETA) = cov ? -1.e01 : parm.const_visc_mu;
     coefs(i,j,k,CXI)  = cov ? -1.e01 : parm.const_visc_ki;
     coefs(i,j,k,CLAM) = cov ? -1.e01 : parm.const_lambda;
}

AMREX_GPU_DEVICE
inline
void
cns_diff_eb_x (int i, int j, int k,
               amrex::Array4<amrex::Real const> const& q,
               amrex::Array4<amrex::Real const> const& coeffs,
               amrex::Array4<amrex::EBCellFlag const> const& flag,
               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
               amrex::GpuArray<amrex::Real,3> const& weights,
               amrex::Array4<amrex::Real> const& fx) noexcept
{
    using amrex::Real;

    Real  dTdx = (q(i,j,k,QTEMP)-q(i-1,j,k,QTEMP))*dxinv[0];
    Real  dudx = (q(i,j,k,QU)-q(i-1,j,k,QU))*dxinv[0];
    Real  dvdx = (q(i,j,k,QV)-q(i-1,j,k,QV))*dxinv[0];
    Real  dwdx = (q(i,j,k,QW)-q(i-1,j,k,QW))*dxinv[0];

    int  jhip = j + (flag(i  ,j,k).isConnected(0, 1,0) ? 1 : 0 );
    int  jhim = j - (flag(i  ,j,k).isConnected(0,-1,0) ? 1 : 0 );
    int  jlop = j + (flag(i-1,j,k).isConnected(0, 1,0) ? 1 : 0 );
    int  jlom = j - (flag(i-1,j,k).isConnected(0,-1,0) ? 1 : 0 );

    Real  whi = weights[jhip-jhim];
    Real  wlo = weights[jlop-jlom];
    Real  dudy = (0.5*dxinv[1]) * ((q(i  ,jhip,k,QU)-q(i  ,jhim,k,QU))*whi 
                                    +(q(i-1,jlop,k,QU)-q(i-1,jlom,k,QU))*wlo);
    Real  dvdy = (0.5*dxinv[1]) * ((q(i  ,jhip,k,QV)-q(i  ,jhim,k,QV))*whi 
                                    +(q(i-1,jlop,k,QV)-q(i-1,jlom,k,QV))*wlo);

    int  khip = k + (flag(i  ,j,k).isConnected(0,0, 1) ? 1 : 0 );
    int  khim = k - (flag(i  ,j,k).isConnected(0,0,-1) ? 1 : 0 );
    int  klop = k + (flag(i-1,j,k).isConnected(0,0, 1) ? 1 : 0 );
    int  klom = k - (flag(i-1,j,k).isConnected(0,0,-1) ? 1 : 0 );

         whi = weights[khip-khim];
         wlo = weights[klop-klom];
    Real  dudz = (0.5*dxinv[2]) * ((q(i  ,j,khip,QU)-q(i  ,j,khim,QU))*whi 
                                    +(q(i-1,j,klop,QU)-q(i-1,j,klom,QU))*wlo);
    Real  dwdz = (0.5*dxinv[2]) * ((q(i  ,j,khip,QW)-q(i  ,j,khim,QW))*whi 
                                    +(q(i-1,j,klop,QW)-q(i-1,j,klom,QW))*wlo);


    Real  divu = dudx + dvdy + dwdz;
    Real  etaf = 0.5*(coeffs(i,j,k,CETA)+coeffs(i-1,j,k,CETA));
    Real  xif  = 0.5*(coeffs(i,j,k,CXI)+coeffs(i-1,j,k,CXI));
    Real  tauxx = etaf*(2.0*dudx-(2.0/3.0)*divu) + xif*divu;
    Real  tauxy = etaf*(dudy+dvdx);
    Real  tauxz = etaf*(dudz+dwdx);

    fx(i,j,k,UMX)   += -tauxx;
    fx(i,j,k,UMY)   += -tauxy;
    fx(i,j,k,UMZ)   += -tauxz;
    fx(i,j,k,UEDEN) += -0.5*( (q(i,j,k,QU)+q(i-1,j,k,QU))*tauxx+
                              (q(i,j,k,QV)+q(i-1,j,k,QV))*tauxy+
                              (q(i,j,k,QW)+q(i-1,j,k,QW))*tauxz+
                              (coeffs(i,j,k,CLAM) +coeffs(i-1,j,k,CLAM))*dTdx );

}

AMREX_GPU_DEVICE
inline
void
cns_diff_eb_y (int i, int j, int k, amrex::Array4<amrex::Real const> const& q,
               amrex::Array4<amrex::Real const> const& coeffs,
               amrex::Array4<amrex::EBCellFlag const> const& flag,
               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
               amrex::GpuArray<amrex::Real,3> const& weights,
               amrex::Array4<amrex::Real> const& fy) noexcept
{
    using amrex::Real;

    Real  dTdy = (q(i,j,k,QTEMP)-q(i,j-1,k,QTEMP))*dxinv[1];
    Real  dudy = (q(i,j,k,QU)-q(i,j-1,k,QU))*dxinv[1];
    Real  dvdy = (q(i,j,k,QV)-q(i,j-1,k,QV))*dxinv[1];
    Real  dwdy = (q(i,j,k,QW)-q(i,j-1,k,QW))*dxinv[1];

    int  ihip = i + (flag(i,j  ,k).isConnected( 1,0,0) ? 1 : 0 );
    int  ihim = i - (flag(i,j  ,k).isConnected(-1,0,0) ? 1 : 0 );
    int  ilop = i + (flag(i,j-1,k).isConnected( 1,0,0) ? 1 : 0 );
    int  ilom = i - (flag(i,j-1,k).isConnected(-1,0,0) ? 1 : 0 );

    Real  whi = weights[ihip-ihim];
    Real  wlo = weights[ilop-ilom];
    Real  dudx = (0.5*dxinv[0]) * ((q(ihip,j  ,k,QU)-q(ihim,j  ,k,QU))*whi 
                                    +(q(ilop,j-1,k,QU)-q(ilom,j-1,k,QU))*wlo);
    Real  dvdx = (0.5*dxinv[0]) * ((q(ihip,j  ,k,QV)-q(ihim,j  ,k,QV))*whi 
                                    +(q(ilop,j-1,k,QV)-q(ilom,j-1,k,QV))*wlo);

    int  khip = k + (flag(i  ,j,k).isConnected(0,0, 1) ? 1 : 0 );
    int  khim = k - (flag(i  ,j,k).isConnected(0,0,-1) ? 1 : 0 );
    int  klop = k + (flag(i-1,j,k).isConnected(0,0, 1) ? 1 : 0 );
    int  klom = k - (flag(i-1,j,k).isConnected(0,0,-1) ? 1 : 0 );

          whi = weights[khip-khim];
          wlo = weights[klop-klom];
    Real  dvdz = (0.5*dxinv[2]) * ((q(i,j  ,khip,QV)-q(i,j  ,khim,QV))*whi 
                                    +(q(i,j-1,klop,QV)-q(i,j-1,klom,QV))*wlo);
    Real  dwdz = (0.5*dxinv[2]) * ((q(i,j  ,khip,QW)-q(i,j  ,khim,QW))*whi 
                                    +(q(i,j-1,klop,QW)-q(i,j-1,klom,QW))*wlo) ;


    Real  divu = dudx + dvdy + dwdz;
    Real  etaf = 0.5*(coeffs(i,j,k,CETA)+coeffs(i,j-1,k,CETA));
    Real  xif  = 0.5*(coeffs(i,j,k,CXI)+coeffs(i,j-1,k,CXI));
    Real  tauyy = etaf*(2.0*dvdy-(2.0/3.0)*divu) + xif*divu;
    Real  tauxy = etaf*(dudy+dvdx);
    Real  tauyz = etaf*(dwdy+dvdz);


    fy(i,j,k,UMX)   += -tauxy;
    fy(i,j,k,UMY)   += -tauyy;
    fy(i,j,k,UMZ)   += -tauyz;
    fy(i,j,k,UEDEN) += -0.5*( (q(i,j,k,QU)+q(i,j-1,k,QU))*tauxy
                             +(q(i,j,k,QV)+q(i,j-1,k,QV))*tauyy
                             +(q(i,j,k,QW)+q(i,j-1,k,QW))*tauyz
                             +(coeffs(i,j,k,CLAM) +coeffs(i,j-1,k,CLAM))*dTdy );

}

AMREX_GPU_DEVICE
inline
void
cns_diff_eb_z (int i, int j, int k,
               amrex::Array4<amrex::Real const> const& q,
               amrex::Array4<amrex::Real const> const& coeffs,
               amrex::Array4<amrex::EBCellFlag const> const& flag,
               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
               amrex::GpuArray<amrex::Real,3> const& weights,
               amrex::Array4<amrex::Real> const& fz) noexcept
{
    using amrex::Real;

    Real  dTdz = (q(i,j,k,QTEMP)-q(i,j,k-1,QTEMP))*dxinv[2];
    Real  dudz = (q(i,j,k,QU)-q(i,j,k-1,QU))*dxinv[2];
    Real  dvdz = (q(i,j,k,QV)-q(i,j,k-1,QV))*dxinv[2];
    Real  dwdz = (q(i,j,k,QW)-q(i,j,k-1,QW))*dxinv[2];

    int  ihip = i + (flag(i,j  ,k).isConnected( 1,0,0) ? 1 : 0 );
    int  ihim = i - (flag(i,j  ,k).isConnected(-1,0,0) ? 1 : 0 );
    int  ilop = i + (flag(i,j-1,k).isConnected( 1,0,0) ? 1 : 0 );
    int  ilom = i - (flag(i,j-1,k).isConnected(-1,0,0) ? 1 : 0 );

    Real   whi = weights[ihip-ihim];
    Real   wlo = weights[ilop-ilom];
    Real   dudx = (0.5*dxinv[0]) * ((q(ihip,j,k  ,QU)-q(ihim,j,k  ,QU))*whi 
                                     +(q(ilop,j,k-1,QU)-q(ilom,j,k-1,QU))*wlo) ;
     Real  dwdx = (0.5*dxinv[0]) * ((q(ihip,j,k  ,QW)-q(ihim,j,k  ,QW))*whi 
                                     +(q(ilop,j,k-1,QW)-q(ilom,j,k-1,QW))*wlo) ;


    int  jhip = j + (flag(i  ,j,k).isConnected(0, 1,0) ? 1 : 0 );
    int  jhim = j - (flag(i  ,j,k).isConnected(0,-1,0) ? 1 : 0 );
    int  jlop = j + (flag(i-1,j,k).isConnected(0, 1,0) ? 1 : 0 );
    int  jlom = j - (flag(i-1,j,k).isConnected(0,-1,0) ? 1 : 0 );

           whi = weights[jhip-jhim];
           wlo = weights[jlop-jlom];
    Real   dvdy = (0.5*dxinv[1]) * ((q(i,jhip,k  ,QV)-q(i,jhim,k  ,QV))*whi 
                                     +(q(i,jlop,k-1,QV)-q(i,jlom,k-1,QV))*wlo) ;
     Real  dwdy = (0.5*dxinv[1]) *  ((q(i,jhip,k  ,QW)-q(i,jhim,k  ,QW))*whi 
                                      +(q(i,jlop,k-1,QW)-q(i,jlom,k-1,QW))*wlo) ;



    Real  divu = dudx + dvdy + dwdz;
    Real  etaf = 0.5*(coeffs(i,j,k,CETA)+coeffs(i,j,k-1,CETA));
    Real  xif  = 0.5*(coeffs(i,j,k,CXI)+coeffs(i,j,k-1,CXI));
    Real  tauxz = etaf*(dudz+dwdx);
    Real  tauyz = etaf*(dvdz+dwdy);
    Real  tauzz = etaf*(2.0*dwdz-(2.0/3.0)*divu) + xif*divu;

    fz(i,j,k,UMX)   += -tauxz;
    fz(i,j,k,UMY)   += -tauyz;
    fz(i,j,k,UMZ)   += -tauzz;
    fz(i,j,k,UEDEN) += -0.5*( (q(i,j,k,QU)+q(i,j,k-1,QU))*tauxz
                             +(q(i,j,k,QV)+q(i,j,k-1,QV))*tauyz
                             +(q(i,j,k,QW)+q(i,j,k-1,QW))*tauzz
                             +(coeffs(i,j,k,CLAM) +coeffs(i,j,k-1,CLAM))*dTdz );
}

#endif
