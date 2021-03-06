module averagedown_module

  use amrex_amr_module

  use amr_data_module, only : phi_new

  implicit none
  private
  
  public :: averagedown

contains

  subroutine averagedown ()
    integer :: lev, finest_level
    finest_level = amrex_get_finest_level()
    do lev = finest_level-1, 0, -1
       call amrex_average_down(phi_new(lev+1), phi_new(lev), amrex_geom(lev+1), amrex_geom(lev), &
            1, 1, amrex_ref_ratio(lev))
    end do
  end subroutine averagedown

end module averagedown_module
