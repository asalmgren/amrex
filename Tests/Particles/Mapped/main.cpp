#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include "AMReX_PlotFileUtil.H"
#include <AMReX_Particles.H>
#include <AMReX_BoxArray.H>
#include <MappedPC.H>
#include <RegularPC.H>
#include <TerrainPC.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void
WriteMultiLevelPlotfileWithMapped (const std::string& plotfilename, int nlevels,
                                         const Vector<const MultiFab*>& mf,
                                         const Vector<const MultiFab*>& mf_nd,
                                         const Vector<std::string>& varnames,
                                         Real time,
                                         const Vector<int>& level_steps,
                                         const std::string &versionName,
                                         const std::string &levelPrefix,
                                         const std::string &mfPrefix,
                                         const Vector<std::string>& extra_dirs,
                                         const Vector<Geometry> geom);


void
WriteGenericPlotfileHeaderWithMapped (std::ostream &HeaderFile,
                                            int nlevels,
                                            const Vector<BoxArray> &bArray,
                                            const Vector<std::string> &varnames,
                                            Real time,
                                            const Vector<int> &level_steps,
                                            const std::string &versionName,
                                            const std::string &levelPrefix,
                                            const std::string &mfPrefix,
                                            const Vector<Geometry> geom);
enum struct GridType {
    Regular, Terrain, Mapped
};

enum struct ProbType {
    Helix, Torus, Annulus, Stretched, Unstretched, Hill
};

enum struct VelType {
    mac, cc, nd
};

struct TestParams
{
    // Number of grid cells in each direction
    IntVect size;

    // Periodicity of each coordinate direction
    IntVect is_periodic;

    // Maximum size of any grid in each coordinate direction
    int max_grid_size;

    // Number of particles to be initialized in each cell
    int num_ppc;

    // How many time steps to run
    int nsteps;

    // How many levels of refinement to run (default = 1, aka "single level")
    int nlevs;

    // Types of meshes currently include
    //    "terrain" -- the grid is regularly spaced in the first (AMREX_SPACEDIM-1) directions,
    //                 but the last direction is irregularly spaced and the height at nodes is
    //                 stored in the a_z_loc array
    //    "mapped"  -- the grid is irregularly spaced in all directions and the locations at nodes
    //                 are stored in the a_xyz_loc array
    //    "regular" -- the grid is regularly spaced in all directions there is no array of positions
    GridType grid_type = GridType::Terrain;

    // Types of velocity locations currently include
    //    "cc"  -- all components of the velocity are stored at cell centers
    //    "nd"  -- all components of the velocity are stored at cell corners (nodes)
    //    "mac" -- normal components of velocity are stored on the cell faces
    VelType vel_type = VelType::mac;

    // Combinations that are currently allowed and can be tested here
    // "regular" + {cc, nd, mac}
    // "terrain" + {cc, nd, mac}
    // "mapped"  + {nd}

    // Sample mesh initializations currently include
    // "unstretched" -- the grid spacing dz is constant
    // "stretched"   -- the grid spacing dz only depends on z
    // "hill"        -- the grid spacing dz varies with x and y as well as z
    // "annulus"     -- this is only for grid_type GridType::Mapped and is a periodic rectangle mapped into an annulus
    ProbType prob_type = ProbType::Hill;

    // Currently the domain size is hard-wired to [0:1, 0.5] if 2D and [0:1, 0:0.5, 0:0.5] if 3D
};

void Test ();

void InitHelix       (MultiFab& a_xyz_loc, Geometry& geom);
void InitTorus       (MultiFab& a_xyz_loc, Geometry& geom);
void InitAnnulus     (MultiFab& a_xyz_loc, Geometry& geom);
void InitUnstretched (MultiFab& a_xyz_loc  , Geometry& geom);
void InitStretched   (MultiFab& a_xyz_loc  , Geometry& geom);
void InitHill        (MultiFab& a_z_loc  , Geometry& geom);

void InitUmac_map (MultiFab* umac, const MultiFab& a_xyz_loc, Geometry& geom, int flow_dir, Real vert_vel, ProbType prob_type);
void InitUmac_reg (MultiFab* umac,                            Geometry& geom, int flow_dir, Real vert_vel, ProbType prob_type);
void InitUCC_map  (MultiFab& u   , const MultiFab& a_xyz_loc, Geometry& geom, int flow_dir, Real vert_vel, ProbType prob_type);
void InitUCC_reg  (MultiFab& u   ,                            Geometry& geom, int flow_dir, Real vert_vel, ProbType prob_type);
void InitUND_map  (MultiFab& u   , const MultiFab& a_xyz_loc, Geometry& geom, int flow_dir, Real vert_vel, ProbType prob_type);
void InitUND_reg  (MultiFab& u   ,                            Geometry& geom, int flow_dir, Real vert_vel, ProbType prob_type);


void get_test_params(TestParams& params)
{
    ParmParse pp("");
    pp.get("size", params.size);
    pp.get("max_grid_size", params.max_grid_size);
    pp.get("num_ppc", params.num_ppc);
    pp.get("is_periodic", params.is_periodic);
    pp.get("nsteps", params.nsteps);
    pp.get("nlevs", params.nlevs);

    std::string vel_type_string;
    pp.get("vel_type"     , vel_type_string);
    AMREX_ALWAYS_ASSERT(vel_type_string == "cc" || vel_type_string == "mac" || vel_type_string == "nd");
    if (vel_type_string == "cc") {
        params.vel_type = VelType::cc;
    } else if (vel_type_string == "nd") {
        params.vel_type = VelType::nd;
    } else if (vel_type_string == "mac") {
        params.vel_type = VelType::mac;
    }

    std::string grid_type_string;
    pp.get("grid_type"     , grid_type_string);
    AMREX_ALWAYS_ASSERT(grid_type_string == "mapped" || grid_type_string == "terrain" || grid_type_string == "regular");
    if (grid_type_string == "mapped") {
        params.grid_type = GridType::Mapped;
    } else if (grid_type_string == "regular") {
        params.grid_type = GridType::Regular;
    } else {
        params.grid_type = GridType::Terrain;
    }

    std::string prob_type_string;
    pp.get("prob_type", prob_type_string);
    AMREX_ALWAYS_ASSERT(prob_type_string == "helix"       ||
                        prob_type_string == "torus"     ||
                        prob_type_string == "annulus"     ||
                        prob_type_string == "stretched"   ||
                        prob_type_string == "unstretched" ||
                        prob_type_string == "hill");
    if (prob_type_string == "helix"      ) params.prob_type = ProbType::Helix;
    if (prob_type_string == "torus"      ) params.prob_type = ProbType::Torus;
    if (prob_type_string == "annulus"    ) params.prob_type = ProbType::Annulus;
    if (prob_type_string == "unstretched") params.prob_type = ProbType::Unstretched;
    if (prob_type_string == "stretched"  ) params.prob_type = ProbType::Stretched;
    if (prob_type_string == "hill"       ) params.prob_type = ProbType::Hill;

    if (params.grid_type == GridType::Terrain) {
        amrex::Print() << "GRID TYPE = TERRAIN" << std::endl;
    } else if (params.grid_type == GridType::Regular) {
        amrex::Print() << "GRID TYPE = REGULAR" << std::endl;
    } else if (params.grid_type == GridType::Mapped) {
        amrex::Print() << "GRID TYPE = MAPPED" << std::endl;
    }
    if (params.vel_type == VelType::mac) {
        amrex::Print() << "VEL  TYPE = MAC" << std::endl;
    } else if (params.vel_type == VelType::cc) {
        amrex::Print() << "VEL  TYPE = CC" << std::endl;
    } else if (params.vel_type == VelType::nd) {
        amrex::Print() << "VEL  TYPE = ND" << std::endl;
    }
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    Test();

    amrex::Finalize();
}

void
WriteMultiLevelPlotfileWithMapped (const std::string& plotfilename, int nlevels,
                                         const Vector<const MultiFab*>& mf,
                                         const Vector<const MultiFab*>& mf_nd,
                                         const Vector<std::string>& varnames,
                                         Real time,
                                         const Vector<int>& level_steps,
                                         const std::string &versionName,
                                         const std::string &levelPrefix,
                                         const std::string &mfPrefix,
                                         const Vector<std::string>& extra_dirs,
                                         const Vector<Geometry> geom)
{
    BL_PROFILE("WriteMultiLevelPlotfileWithTerrain()");
    amrex::Vector<IntVect> ref_ratio(nlevels);
    for (int i = 0 ; i <nlevels; ++i) {
        ref_ratio[i] = amrex::IntVect::TheUnitVector();
    }
    BL_ASSERT(nlevels <= mf.size());
    BL_ASSERT(nlevels <= ref_ratio.size()+1);
    BL_ASSERT(nlevels <= level_steps.size());
    BL_ASSERT(mf[0]->nComp() == varnames.size());
    int finest_level = 0;
    bool callBarrier(false);
    PreBuildDirectorHierarchy(plotfilename, levelPrefix, nlevels, callBarrier);
    if (!extra_dirs.empty()) {
        for (const auto& d : extra_dirs) {
            const std::string ed = plotfilename+"/"+d;
            PreBuildDirectorHierarchy(ed, levelPrefix, nlevels, callBarrier);
        }
    }
    ParallelDescriptor::Barrier();

    if (ParallelDescriptor::MyProc() == ParallelDescriptor::NProcs()-1) {
        Vector<BoxArray> boxArrays(nlevels);
        for(int level(0); level < boxArrays.size(); ++level) {
            boxArrays[level] = mf[level]->boxArray();
        }

        auto f = [=]() {
            VisMF::IO_Buffer io_buffer(VisMF::IO_Buffer_Size);
            std::string HeaderFileName(plotfilename + "/Header");
            std::ofstream HeaderFile;
            HeaderFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
            HeaderFile.open(HeaderFileName.c_str(), std::ofstream::out   |
                                                    std::ofstream::trunc |
                                                    std::ofstream::binary);
            if( ! HeaderFile.good()) FileOpenFailed(HeaderFileName);
            WriteGenericPlotfileHeaderWithMapped(HeaderFile, nlevels, boxArrays, varnames,
                                                  time, level_steps, versionName,
                                                  levelPrefix, mfPrefix, geom);
        };

        if (AsyncOut::UseAsyncOut()) {
            AsyncOut::Submit(std::move(f));
        } else {
            f();
        }
    }

    std::string mf_nodal_prefix = "Nu_nd";
    for (int level = 0; level <= finest_level; ++level)
    {
        if (AsyncOut::UseAsyncOut()) {
            VisMF::AsyncWrite(*mf[level],
                              MultiFabFileFullPrefix(level, plotfilename, levelPrefix, mfPrefix),
                              true);
            VisMF::AsyncWrite(*mf_nd[level],
                              MultiFabFileFullPrefix(level, plotfilename, levelPrefix, mf_nodal_prefix),
                              true);
        } else {
            const MultiFab* data;
            std::unique_ptr<MultiFab> mf_tmp;
            if (mf[level]->nGrowVect() != 0) {
                mf_tmp = std::make_unique<MultiFab>(mf[level]->boxArray(),
                                                    mf[level]->DistributionMap(),
                                                    mf[level]->nComp(), 0, MFInfo(),
                                                    mf[level]->Factory());
                MultiFab::Copy(*mf_tmp, *mf[level], 0, 0, mf[level]->nComp(), 0);
                data = mf_tmp.get();
            } else {
                data = mf[level];
            }
            VisMF::Write(*data        , MultiFabFileFullPrefix(level, plotfilename, levelPrefix, mfPrefix));
            VisMF::Write(*mf_nd[level], MultiFabFileFullPrefix(level, plotfilename, levelPrefix, mf_nodal_prefix));
        }
    }
}

void
WriteGenericPlotfileHeaderWithMapped (std::ostream &HeaderFile,
                                            int nlevels,
                                            const Vector<BoxArray> &bArray,
                                            const Vector<std::string> &varnames,
                                            Real time,
                                            const Vector<int> &level_steps,
                                            const std::string &versionName,
                                            const std::string &levelPrefix,
                                            const std::string &mfPrefix,
                                            const Vector<Geometry> geom)
{
        amrex::Vector<IntVect> ref_ratio(nlevels);
        for (int i = 0 ; i <nlevels; ++i) {
            ref_ratio[i] = amrex::IntVect::TheUnitVector();
        } 
        BL_ASSERT(nlevels <= bArray.size());
        BL_ASSERT(nlevels <= ref_ratio.size()+1);
        BL_ASSERT(nlevels <= level_steps.size());
        int finest_level = 0;
        HeaderFile.precision(17);

        // ---- this is the generic plot file type name
        HeaderFile << versionName << '\n';

        HeaderFile << varnames.size() << '\n';

        for (int ivar = 0; ivar < varnames.size(); ++ivar) {
            HeaderFile << varnames[ivar] << "\n";
        }
        HeaderFile << AMREX_SPACEDIM << '\n';
        HeaderFile << time << '\n';
        HeaderFile << finest_level << '\n';
        for (int i = 0; i < AMREX_SPACEDIM; ++i) {
            HeaderFile << geom[0].ProbLo(i) << ' ';
        }
        HeaderFile << '\n';
        for (int i = 0; i < AMREX_SPACEDIM; ++i) {
            HeaderFile << geom[0].ProbHi(i) << ' ';
        }
        HeaderFile << '\n';
        for (int i = 0; i < finest_level; ++i) {
            HeaderFile << ref_ratio[i][0] << ' ';
        }
        HeaderFile << '\n';
        for (int i = 0; i <= finest_level; ++i) {
            HeaderFile << geom[i].Domain() << ' ';
        }
        HeaderFile << '\n';
        for (int i = 0; i <= finest_level; ++i) {
            HeaderFile << level_steps[i] << ' ';
        }
        HeaderFile << '\n';
        for (int i = 0; i <= finest_level; ++i) {
            for (int k = 0; k < AMREX_SPACEDIM; ++k) {
                HeaderFile << geom[i].CellSize()[k] << ' ';
            }
            HeaderFile << '\n';
        }
        HeaderFile << (int) geom[0].Coord() << '\n';
        HeaderFile << "0\n";

        for (int level = 0; level <= finest_level; ++level) {
            HeaderFile << level << ' ' << bArray[level].size() << ' ' << time << '\n';
            HeaderFile << level_steps[level] << '\n';

            const IntVect& domain_lo = geom[level].Domain().smallEnd();
            for (int i = 0; i < bArray[level].size(); ++i)
            {
                // Need to shift because the RealBox ctor we call takes the
                // physical location of index (0,0,0).  This does not affect
                // the usual cases where the domain index starts with 0.
                const Box& b = shift(bArray[level][i], -domain_lo);
                RealBox loc = RealBox(b, geom[level].CellSize(), geom[level].ProbLo());
                for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                    HeaderFile << loc.lo(n) << ' ' << loc.hi(n) << '\n';
                }
            }

            HeaderFile << MultiFabHeaderPath(level, levelPrefix, mfPrefix) << '\n';
        }
        HeaderFile << "1" << "\n";
        HeaderFile << "3" << "\n";
        HeaderFile << "amrexvec_nu_x" << "\n";
        HeaderFile << "amrexvec_nu_y" << "\n";
        HeaderFile << "amrexvec_nu_z" << "\n";
        std::string mf_nodal_prefix = "Nu_nd";
        for (int level = 0; level <= finest_level; ++level) {
            HeaderFile << MultiFabHeaderPath(level, levelPrefix, mf_nodal_prefix) << '\n';
        }
}



void Test()
{
    BL_PROFILE("Test");
    TestParams params;
    get_test_params(params);

    int is_per[] = {AMREX_D_DECL(params.is_periodic[0],
                                 params.is_periodic[1],
                                 params.is_periodic[2])};

    RealBox real_box;
    real_box.setLo(0, 0.0);
    real_box.setHi(0, 1.0);

    real_box.setLo(1, 0.0);
    real_box.setHi(1, 1.0);

#if (AMREX_SPACEDIM == 3)
    real_box.setLo(2, 0.0);
    real_box.setHi(2, 5.0);
#endif

    IntVect domain_lo(AMREX_D_DECL(0, 0, 0));
    IntVect domain_hi(AMREX_D_DECL(params.size[0]-1,params.size[1]-1,params.size[2]-1));
    const Box base_domain(domain_lo, domain_hi);

    Vector<Geometry> geom(params.nlevs);
    geom[0].define(base_domain, &real_box, CoordSys::cartesian, is_per);

    Vector<BoxArray> ba(params.nlevs);
    Vector<DistributionMapping> dm(params.nlevs);
    IntVect lo(0);
    IntVect size = params.size;
    for (int lev = 0; lev < params.nlevs; ++lev)
    {
        ba[lev].define(Box(domain_lo, domain_hi));
        ba[lev].maxSize(params.max_grid_size);
        dm[lev].define(ba[lev]);
        lo += size/2;
        size *= 2;
    }

    // We currently assume a single-level problem
    int lev = 0;

    // We define both types of particles here, but in separate particle containers
    MappedPC   mapped_pc(geom[lev], dm[lev], ba[lev]);
    TerrainPC terrain_pc(geom[lev], dm[lev], ba[lev]);
    RegularPC regular_pc(geom[lev], dm[lev], ba[lev]);

    IntVect nppc(params.num_ppc);

    // **************************************************************************************
    // Define xyz_phys on nodes
    // **************************************************************************************
    BoxArray ba_nd(ba[lev]); ba_nd.surroundingNodes();

    // This has one component, the "height"
    MultiFab   a_z_loc(ba_nd,dm[lev],1,1);

    // This has AMREX_SPACEDIM components to define (x,y,z) as a function of (i,j,k)
    MultiFab a_xyz_loc(ba_nd,dm[lev],AMREX_SPACEDIM,2);

    // Annulus
    if (params.prob_type == ProbType::Helix) {
        AMREX_ALWAYS_ASSERT(params.grid_type == GridType::Mapped);
        AMREX_ALWAYS_ASSERT(AMREX_SPACEDIM==3);
        InitHelix(a_xyz_loc, geom[lev]);
    } else if (params.prob_type == ProbType::Torus) {
        AMREX_ALWAYS_ASSERT(params.grid_type == GridType::Mapped);
        AMREX_ALWAYS_ASSERT(AMREX_SPACEDIM==3);
        InitTorus(a_xyz_loc, geom[lev]);
    } else if (params.prob_type == ProbType::Annulus) {
        AMREX_ALWAYS_ASSERT(params.grid_type == GridType::Mapped);
        InitAnnulus(a_xyz_loc, geom[lev]);

    // No stretching
    } else if (params.prob_type == ProbType::Unstretched) {
        if (params.grid_type == GridType::Terrain) {
            InitUnstretched(a_z_loc, geom[lev]);
        } else if (params.grid_type == GridType::Mapped) {
            InitUnstretched(a_xyz_loc, geom[lev]);
        } else {
            // We don't need to initialize any arrays if GridType::Regular
        }

    // Stretched but flat
    } else if (params.prob_type == ProbType::Stretched) {
        AMREX_ALWAYS_ASSERT(params.grid_type != GridType::Regular);
        if (params.grid_type == GridType::Terrain) {
            InitStretched(a_z_loc, geom[lev]);
        } else if (params.grid_type == GridType::Mapped) {
            InitStretched(a_xyz_loc, geom[lev]);
        }

    // Hill
    } else if (params.prob_type == ProbType::Hill) {
        AMREX_ALWAYS_ASSERT(params.grid_type != GridType::Regular);
        if (params.grid_type == GridType::Terrain) {
            InitHill(a_z_loc, geom[lev]);
        } else if (params.grid_type == GridType::Mapped) {
            InitHill(a_xyz_loc, geom[lev]);
        }

    } else {
        amrex::Abort("Don't know this prob_type");
    }

    // **************************************************************************************
    // Define velocity components (for now it is constant velocity in "i" direction)
    // **************************************************************************************
    MultiFab ucc;                  // Cell-centered
    MultiFab und;                  // Node-centered
    MultiFab umac[AMREX_SPACEDIM]; // Face-centered

    // Choose between flow in x- and y-directions
    int flow_dir = 0;
#if (AMREX_SPACEDIM == 3)
    // This is an option in 3d
    // int flow_dir = 1;
#endif

    // Hard-wire the vertical velocity
    Real vert_vel = 0.1;

    if (params.vel_type == VelType::mac)
    {
        BoxArray ba_x(ba[lev]); ba_x.convert(IntVect(AMREX_D_DECL(1,0,0)));
        umac[0].define(ba_x,dm[lev],1,1); umac[0].setVal(0.0);
        umac[0].setVal(0.);

        BoxArray ba_y(ba[lev]); ba_y.convert(IntVect(AMREX_D_DECL(0,1,0)));
        umac[1].define(ba_y,dm[lev],1,1); umac[1].setVal(0.0);
        umac[1].setVal(0.);

#if (AMREX_SPACEDIM == 3)
        BoxArray ba_z(ba[lev]); ba_z.convert(IntVect(AMREX_D_DECL(0,0,1)));
        umac[2].define(ba_z,dm[lev],1,1); umac[2].setVal(0.0);
        umac[2].setVal(0.);
#endif

        if (params.grid_type == GridType::Regular) {
            InitUmac_reg(&umac[0],          geom[lev], flow_dir, vert_vel, params.prob_type);
        } else if (params.grid_type == GridType::Terrain) {
            InitUmac_map(&umac[0], a_z_loc, geom[lev], flow_dir, vert_vel, params.prob_type);
        } else {
            amrex::Error("Mapped grids aren't allowed with MAC velocities");
        }
        umac[0].FillBoundary(geom[lev].periodicity());
        umac[1].FillBoundary(geom[lev].periodicity());
#if (AMREX_SPACEDIM == 3)
        umac[2].FillBoundary(geom[lev].periodicity());
#endif

    } else if (params.vel_type == VelType::cc) {
        ucc.define(ba[lev],dm[lev],AMREX_SPACEDIM,1);
        ucc.setVal(0.);

        if (params.grid_type == GridType::Regular) {
            InitUCC_reg(ucc,          geom[lev], flow_dir, vert_vel, params.prob_type);
        } else {
            InitUCC_map(ucc, a_z_loc, geom[lev], flow_dir, vert_vel, params.prob_type);
        }
        ucc.FillBoundary(geom[lev].periodicity());

    } else if (params.vel_type == VelType::nd) {
        und.define(ba_nd,dm[lev],AMREX_SPACEDIM,1);
        und.setVal(0.);

        if (params.grid_type == GridType::Regular) {
            InitUND_reg(und,          geom[lev], flow_dir, vert_vel, params.prob_type);
        } else if (params.grid_type == GridType::Terrain) {
            InitUND_map(und, a_z_loc, geom[lev], flow_dir, vert_vel, params.prob_type);
        } else if (params.grid_type == GridType::Mapped) {
            InitUND_map(und, a_xyz_loc, geom[lev], flow_dir, vert_vel, params.prob_type);
        }
        und.FillBoundary(geom[lev].periodicity());

    } else {
        amrex::Abort("Unknown vel_type");
    }

    // **************************************************************************************
    // Initialize and write out particle locations
    // **************************************************************************************
    if (params.grid_type == GridType::Mapped)
    {
        mapped_pc.InitParticles(a_xyz_loc);
        // mapped_pc.WritePlotFile("plt", "particles");
    }
    else if (params.grid_type == GridType::Terrain)
    {
        terrain_pc.InitParticles(a_z_loc);
        // terrain_pc.WritePlotFile("plt", "particles");
    }
    else if (params.grid_type == GridType::Regular)
    {
        regular_pc.InitParticles();
        // regular_pc.WritePlotFile("plt", "particles");
    }

    // **************************************************************************************
    // Advance the particle positions in time based on the face-based velocity
    // **************************************************************************************
    std::string plotfilename;

#if 0
    amrex::Real max_vel;
    if (params.vel_type == VelType::mac) {
        max_vel = umac[0].max(0,0,false);
    } else if (params.vel_type == VelType::cc) {
        max_vel = ucc.max(0,0,false);
    } else if (params.vel_type == VelType::nd) {
        max_vel = und.max(0,0,false);
    } else {
        amrex::Error("What is this velocity type??");
    }

    // This is assuming velocity only in x-direction
    auto dx = geom[lev].CellSize();
    amrex::Real dt = 0.9 * dx[0] / max_vel;
    amrex::Print() << "COMPUTING DT TO BE " << dt << " BASED ON MAX VEL " << max_vel << std::endl;
#else
    auto dx = geom[0].CellSize();
    amrex::Real dt = 0.01;
    amrex::Print() << "SETTING DT TO BE " << dt << std::endl;
#endif

    for (int nt = 0; nt < params.nsteps; nt++)
    {
        if (params.grid_type == GridType::Regular && params.vel_type == VelType::mac) {
            amrex::Print() << "Advecting at time " << nt << " using MAC velocities" << std::endl;
            regular_pc.AdvectWithUmac(&umac[0], 0, dt);

        } else if (params.grid_type == GridType::Regular && params.vel_type == VelType::cc) {
            amrex::Print() << "Advecting at time " << nt << " using cell-centered velocities" << std::endl;
            regular_pc.AdvectWithUCC(ucc, 0, dt);

        } else if (params.grid_type == GridType::Regular && params.vel_type == VelType::nd) {
            amrex::Print() << "Advecting at time " << nt << " using node-centered velocities" << std::endl;
            regular_pc.AdvectWithUND(und, 0, dt);

        } else if (params.grid_type == GridType::Terrain && params.vel_type == VelType::mac) {
            amrex::Print() << "Advecting at time " << nt << " using MAC velocities" << std::endl;
            terrain_pc.AdvectWithUmac(&umac[0], 0, dt, a_z_loc);

        } else if (params.grid_type == GridType::Terrain && params.vel_type == VelType::cc) {
            amrex::Print() << "Advecting at time " << nt << " using cell-centered velocities" << std::endl;
            terrain_pc.AdvectWithUCC(ucc, 0, dt, a_z_loc);

        } else if (params.grid_type == GridType::Terrain && params.vel_type == VelType::nd) {
            amrex::Print() << "Advecting at time " << nt << " using node-centered velocities" << std::endl;
            terrain_pc.AdvectWithUND(und, 0, dt, a_z_loc);

        } else if (params.grid_type == GridType::Mapped && params.vel_type == VelType::nd) {
            ucc.FillBoundary(geom[0].periodicity());
            amrex::Print() << "Advecting at time " << nt << " using node-centered velocities" << std::endl;
            mapped_pc.AdvectWithUND(und, 0, dt, a_xyz_loc);
        }


    if (nt%2 ==0){
        amrex::Print() << " writing plotfile \n";
        plotfilename = Concatenate("plt", nt, 5);
        Vector<std::string> varname = {"ux", "uy", "uz"};
        amrex::MultiFab plotmf(ba[0], dm[0], varname.size(), 0 );
        amrex::average_node_to_cellcenter (plotmf, 0, und, 0, 3, 0);
        // if (params.grid_type == GridType::Terrain) {
        //     terrain_pc.WritePlotFile(plotfilename, "particles");
        // } else if (params.grid_type == GridType::Mapped) {
        //     mapped_pc.WritePlotFile(plotfilename, "particles");
        // } else if (params.grid_type == GridType::Regular) {
        //     regular_pc.WritePlotFile(plotfilename, "particles");
        // }
        // Make a nodal multifab that stores the location of the grid as x,y,z (3 components)
        amrex::Vector<amrex::MultiFab> mf_nd(1);
        if (params.grid_type == GridType::Mapped) {
            amrex::BoxArray nodal_ba(ba[0]);
            nodal_ba.surroundingNodes();
            mf_nd[0].define(nodal_ba,dm[0],3,0);
            mf_nd[0].setVal(0.);
            amrex::Print() << nodal_ba[0] << "\n";
            //Copy data from xyz location mf (copy 3 components)
            amrex::MultiFab::Copy(mf_nd[0],a_xyz_loc,0,0,3,0);
            amrex::Real dx = geom[0].CellSizeArray()[0];
            amrex::Real dy = geom[0].CellSizeArray()[1];
            amrex::Real dz = geom[0].CellSizeArray()[2];
            for (amrex::MFIter mfi(mf_nd[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const amrex::Box & bx = mfi.tilebox();
                amrex::Array4<amrex::Real> mf_arr = mf_nd[0].array(mfi);
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    mf_arr(i,j,k,0) -= i*dx;
                    mf_arr(i,j,k,1) -= j*dy;
                    mf_arr(i,j,k,2) -= k*dz;
                });
            }
        }


        const std::string versionName = "HyperCLaw-V1.1";
        const std::string levelPrefix = "Level_";
        const std::string mfPrefix = "Cell";
        const amrex::Vector<std::string> extra_dirs = amrex::Vector<std::string>();
        Vector<const MultiFab*> mfarr(1,&plotmf);
        Vector<Geometry> geomarr(1,geom[0]);
        Vector<int> level_steps(1,nt);
        Vector<IntVect> ref_ratio;
        ///WriteSingleLevelPlotfile(plotfilename, plotmf, varname, geom[0],0.0,0);
        WriteMultiLevelPlotfileWithMapped(plotfilename, 1,
                                           GetVecOfConstPtrs(mfarr),
                                           GetVecOfConstPtrs(mf_nd),
                                           varname,
                                           nt, level_steps, versionName, levelPrefix, mfPrefix, extra_dirs, geomarr); 
        mapped_pc.WritePlotFile(plotfilename,"particles");

        //WriteSingleLevelPlotfile("plt_grid",a_xyz_loc,{"gridmap",AMREX_D_DECL("x1","y1","z1")},geom,0.0,0);
        }
    } // nt
//    plotfilename = Concatenate("plt", nt, 5);
//    Vector<std::string> varname = {"dummy"};
//    amrex::MultiFab plotmf(ba[0], dm[0], varname.size(), 0 );
//    plotmf.setval(0.0);
//    WriteSingleLevelPlotfileWithTerrain(plotfilename, plotmf, varname, geom[0],0.0,0);
//    //WriteSingleLevelPlotfile("plt_grid",a_xyz_loc,{"gridmap",AMREX_D_DECL("x1","y1","z1")},geom,0.0,0);
//    mapped_pc.WritePlotFile(plotfilename,"particles");
}
