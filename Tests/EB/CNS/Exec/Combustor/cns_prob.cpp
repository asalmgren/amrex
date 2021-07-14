
#include <AMReX_PROB_AMR_F.H>
#include <AMReX_ParmParse.H>
#include "cns_prob_parm.H"
#include "CNS.H"

extern "C" {
    void amrex_probinit (const int* /*init*/,
                         const int* /*name*/,
                         const int* /*namelen*/,
                         const amrex_real* /*problo*/,
                         const amrex_real* /*probhi*/)
    {
        amrex::ParmParse pp("prob");

        pp.query("inflow_T"   , CNS::h_prob_parm->inflow_T);
        pp.query("inflow_p"   , CNS::h_prob_parm->inflow_p);
        pp.query("inflow_mach", CNS::h_prob_parm->inflow_mach);
        pp.query("interior_T" , CNS::h_prob_parm->interior_T);
        pp.query("interior_P" , CNS::h_prob_parm->interior_p);

        amrex::Gpu::copy(amrex::Gpu::hostToDevice, CNS::h_prob_parm, CNS::h_prob_parm+1,
                         CNS::d_prob_parm);
    }
}
