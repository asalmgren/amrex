module cns_nd_module

  use amrex_fort_module, only : rt=>amrex_real
  implicit none

  private

  public :: cns_nullfill

contains

  subroutine cns_nullfill(adv,adv_lo,adv_hi,domlo,domhi,delta,xlo,time,bc) &
       bind(C, name="cns_nullfill")
    use amrex_fort_module, only: dim=>amrex_spacedim
    use amrex_error_module, only : amrex_error
    implicit none
    include 'AMReX_bc_types.fi'
    integer          :: adv_lo(3),adv_hi(3)
    integer          :: bc(dim,2,*)
    integer          :: domlo(3), domhi(3)
    double precision :: delta(3), xlo(3), time
    double precision :: adv(adv_lo(1):adv_hi(1),adv_lo(2):adv_hi(2),adv_lo(3):adv_hi(3))
    call amrex_error("How did this happen?")
  end subroutine cns_nullfill

end module cns_nd_module
