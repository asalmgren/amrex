module probdata_module
  use amrex_fort_module, only : rt => amrex_real
  use cns_module, only : center, nvar, urho, umx, umy, umz, ueden, ueint, utemp
  implicit none
  real(rt), save :: inflow_T = 300.d0
  real(rt), save :: inflow_p = 1.0d6
  real(rt), save :: inflow_mach = 0.8
  real(rt), save :: interior_T = 1500.d0
  real(rt), save :: interior_p = 1.0d6
  !
  real(rt), save :: inflow_state(nvar)
end module probdata_module


subroutine amrex_probinit (init,name,namlen,problo,probhi) bind(c)
  use amrex_fort_module, only : rt => amrex_real
  use probdata_module
  use cns_physics_module, only : cv, gamma
  implicit none
  integer, intent(in) :: init, namlen
  integer, intent(in) :: name(namlen)
  real(rt), intent(in) :: problo(*), probhi(*)
  real(rt) :: rho, v, rhoe, cs

  rhoe = inflow_p / (gamma-1.d0)
  rho = rhoe/(cv*inflow_T)
  cs = sqrt(gamma*inflow_p/rho)
  v = inflow_mach * cs
  inflow_state(urho) = rho
  inflow_state(umx) = 0.d0
  inflow_state(umy) = 0.d0
  inflow_state(umz) = rho*v
  inflow_state(ueden) = rhoe + 0.5d0*rho*v*v
  inflow_state(ueint) = rhoe
  inflow_state(utemp) = inflow_T

end subroutine amrex_probinit

