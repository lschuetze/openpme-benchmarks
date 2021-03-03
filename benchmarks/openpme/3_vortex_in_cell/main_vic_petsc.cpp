#include <cmath>
#include <Vector/vector_dist.hpp>
#include <Grid/grid_dist_id.hpp>
#include <Plot/GoogleChart.hpp>
#include <Plot/util.hpp>
#include <timer.hpp>
#include <interpolation/interpolation.hpp>
#include <Matrix/SparseMatrix.hpp>
#include <Vector/Vector.hpp>
#include <FiniteDifference/FDScheme.hpp>
#include <Solvers/petsc_solver.hpp>
#include <interpolation/mp4_kernel.hpp>
#include <Solvers/petsc_solver_AMG_report.hpp>
#include <Decomposition/Distribution/SpaceDistribution.hpp>

// Property indices
constexpr unsigned int vorticity_m = 0;
constexpr unsigned int vorticity = 0;
constexpr unsigned int velocity = 1;
constexpr unsigned int vortex_stretching = 2;
constexpr unsigned int old_vorticity = 3;
constexpr unsigned int old_position = 4;
constexpr unsigned int velocity_m = 0;
constexpr unsigned int velocity_phi = 0;
constexpr unsigned int vortex_stretching_m = 0;

constexpr unsigned int phi_poisson = 0;

struct poisson_nn
{
static const unsigned int dims = 3;
static const unsigned int nvar = 1;
static const bool boundary[];
typedef double stype;
typedef grid_dist_id<3, double, aggregate<double>> b_grid;
typedef SparseMatrix<double, int, PETSC_BASE> SparseMatrix_type;
typedef Vector<double, PETSC_BASE> Vector_type;
static const int grid_type = NORMAL_GRID;
};
const bool poisson_nn::boundary[] = {PERIODIC, PERIODIC, PERIODIC};

int main(int argc, char* argv[])
{
  
  openfpm_init(&argc, &argv);
  
  // Initialization
  double r_cut = 2;
  Box<3, double> box({0.0, 0.0, 0.0}, {3.57, 3.57, 3.57});
   size_t bc_particle[3] = {PERIODIC, PERIODIC, PERIODIC};
  periodicity<3> bc_grid = {{PERIODIC, PERIODIC, PERIODIC}};
  Ghost<3, long int> ghost(r_cut);
  
  
  // Hybrid Simulation
  
{
  
  // Field containers
   size_t sz_vorticity_mesh[3] = {128, 128, 128};
  grid_dist_id<3, double, aggregate<double[3]>> vorticity_mesh(sz_vorticity_mesh, box, ghost, bc_grid);
   size_t sz_velocity_mesh[3] = {128, 128, 128};
  grid_dist_id<3, double, aggregate<double[3]>> velocity_mesh(sz_velocity_mesh, box, ghost, bc_grid);
   size_t sz_phi[3] = {128, 128, 128};
  grid_dist_id<3, double, aggregate<double[3]>> phi(sz_phi, box, ghost, bc_grid);
   size_t sz_rhs[3] = {128, 128, 128};
  grid_dist_id<3, double, aggregate<double[3]>> rhs(sz_rhs, box, ghost, bc_grid);
  vector_dist<3, double, aggregate<double[3], double[3], double[3], double[3], double[3]>> particles(vorticity_mesh.getDecomposition(), 0);
  
  // Init particles
  // Init meshes (TODO)
  
  // Instantiate interpolation object
  interpolate<vector_dist<3, double, aggregate<double[3], double[3], double[3], double[3], double[3]>>, grid_dist_id<3, double, aggregate<double[3]>>, mp4_kernel<double>> interpol(particles, vorticity_mesh);
  
  // Poisson solver prerequisites
  grid_dist_id<3, double, aggregate<double>> psi_poisson(vorticity_mesh.getDecomposition(), vorticity_mesh.getGridInfo().getSize(), ghost);
  Ghost<3, long int> stencil_max_poisson(2);
  typedef Field<phi_poisson, poisson_nn> phi_f_poisson;
  typedef Lap<phi_f_poisson, poisson_nn, CENTRAL_SYM> poisson;
  FDScheme<poisson_nn> fd_poisson(stencil_max_poisson, box, psi_poisson);
  fd_poisson.template impose_dit<0>(poisson(), psi_poisson, psi_poisson.getDomainIterator());
   Vector<double, PETSC_BASE> phi_solution_poisson[3];
  petsc_solver<double> poisson_solver;
  phi_solution_poisson[0].resize(vorticity_mesh.size(), vorticity_mesh.getLocalDomainSize());
  phi_solution_poisson[1].resize(vorticity_mesh.size(), vorticity_mesh.getLocalDomainSize());
  phi_solution_poisson[2].resize(vorticity_mesh.size(), vorticity_mesh.getLocalDomainSize());
  phi_solution_poisson[0].setZero();
  phi_solution_poisson[1].setZero();
  phi_solution_poisson[2].setZero();
  poisson_solver.setSolver(KSPBCGS);
  poisson_solver.setMaxIter(500);
  poisson_solver.setAbsTol(0.01);
  for (size_t init_poisson_solver_dim = 0; init_poisson_solver_dim < 3; init_poisson_solver_dim++)
  {
    poisson_solver.solve(fd_poisson.getA(), phi_solution_poisson[init_poisson_solver_dim], fd_poisson.getB());
  }
  
  // Commands
  double nu = (1 / 3000);
  double dt = 0.0125;
  vorticity_mesh.load("init_vort_double.hdf5");
  // Remesh
  particles.clear();
  auto remesh_iterator_d0a = particles.getGridIterator(vorticity_mesh.getGridInfo().getSize());
  while (remesh_iterator_d0a.isNext())
  {
    auto local_key_d0a = remesh_iterator_d0a.get();
    auto global_key_d0a = remesh_iterator_d0a.get_dist();
    particles.add();
    particles.getLastPos()[0] = vorticity_mesh.spacing(0) * local_key_d0a.get(0) + box.getLow(0);
    particles.getLastPos()[1] = vorticity_mesh.spacing(1) * local_key_d0a.get(1) + box.getLow(1);
    particles.getLastPos()[2] = vorticity_mesh.spacing(2) * local_key_d0a.get(2) + box.getLow(2);
    particles.template getLastProp<vorticity>()[0] = vorticity_mesh.template get<vorticity_m>(global_key_d0a)[0];
    particles.template getLastProp<vorticity>()[1] = vorticity_mesh.template get<vorticity_m>(global_key_d0a)[1];
    particles.template getLastProp<vorticity>()[2] = vorticity_mesh.template get<vorticity_m>(global_key_d0a)[2];
    ++remesh_iterator_d0a;
  }
  particles.map();
  int i = 0;
  
  // Time loop
  for (size_t time_step = 1; time_step < 10; time_step++)
  {
    
    // Mesh loop
    auto mloop_iterator_a5a0 = vorticity_mesh.getDomainIterator();
    while (mloop_iterator_a5a0.isNext())
    {
      auto loopNodeM = mloop_iterator_a5a0.get();
      vorticity_mesh.template get<vorticity_m>(loopNodeM)[0] = 0.0;
      vorticity_mesh.template get<vorticity_m>(loopNodeM)[1] = 0.0;
      vorticity_mesh.template get<vorticity_m>(loopNodeM)[2] = 0.0;
      ++mloop_iterator_a5a0;
    }
    interpol.template p2m<vorticity, vorticity_m>(particles, vorticity_mesh);
    
    // Solve poisson equation
    for (size_t dim_poisson = 0; dim_poisson < 3; dim_poisson++)
    {
      auto poisson_copy_prop_iterator = psi_poisson.getDomainIterator();
      while (poisson_copy_prop_iterator.isNext())
      {
        auto poisson_copy_node = poisson_copy_prop_iterator.get();
        psi_poisson.get<phi_poisson>(poisson_copy_node) = vorticity_mesh.template get<vorticity_m>(poisson_copy_node)[dim_poisson];
        ++poisson_copy_prop_iterator;
      }
      fd_poisson.new_b();
      fd_poisson.template impose_dit_b<phi_poisson>(psi_poisson, psi_poisson.getDomainIterator());
      poisson_solver.setAbsTol(0.01);
      Vector<double, PETSC_BASE> &poisson_b = fd_poisson.getB();
      poisson_solver.solve(phi_solution_poisson[dim_poisson], poisson_b);
      fd_poisson.template copy<phi_poisson>(phi_solution_poisson[dim_poisson], psi_poisson);
      auto poisson_copy_prop_back_iterator = psi_poisson.getDomainIterator();
      while (poisson_copy_prop_back_iterator.isNext())
      {
        auto poisson_copy_back_node = poisson_copy_prop_back_iterator.get();
        phi.get<velocity_phi>(poisson_copy_back_node)[dim_poisson] = psi_poisson.get<phi_poisson>(poisson_copy_back_node);
        ++poisson_copy_prop_back_iterator;
      }
    }
    phi.ghost_get<velocity_phi>();
    
    // Mesh loop
    auto mloop_iterator_d5a0 = velocity_mesh.getDomainIterator();
    while (mloop_iterator_d5a0.isNext())
    {
      auto loopNodeM = mloop_iterator_d5a0.get();
      velocity_mesh.template get<velocity_m>(loopNodeM)[0] = (((phi.template get<velocity_phi>(loopNodeM.move(1, 1))[2] - phi.template get<velocity_phi>(loopNodeM.move(1, -1))[2]) * (0.5 / phi.spacing(1))) - ((phi.template get<velocity_phi>(loopNodeM.move(2, 1))[1] - phi.template get<velocity_phi>(loopNodeM.move(2, -1))[1]) * (0.5 / phi.spacing(2))));
      velocity_mesh.template get<velocity_m>(loopNodeM)[1] = (((phi.template get<velocity_phi>(loopNodeM.move(2, 1))[0] - phi.template get<velocity_phi>(loopNodeM.move(2, -1))[0]) * (0.5 / phi.spacing(2))) - ((phi.template get<velocity_phi>(loopNodeM.move(0, 1))[2] - phi.template get<velocity_phi>(loopNodeM.move(0, -1))[2]) * (0.5 / phi.spacing(0))));
      velocity_mesh.template get<velocity_m>(loopNodeM)[2] = (((phi.template get<velocity_phi>(loopNodeM.move(0, 1))[1] - phi.template get<velocity_phi>(loopNodeM.move(0, -1))[1]) * (0.5 / phi.spacing(0))) - ((phi.template get<velocity_phi>(loopNodeM.move(1, 1))[0] - phi.template get<velocity_phi>(loopNodeM.move(1, -1))[0]) * (0.5 / phi.spacing(1))));
      ++mloop_iterator_d5a0;
    }
    
    // Mesh loop
    auto mloop_iterator_e5a0 = rhs.getDomainIterator();
    while (mloop_iterator_e5a0.isNext())
    {
      auto loopNodeM = mloop_iterator_e5a0.get();
      rhs.template get<vortex_stretching_m>(loopNodeM)[0] = ((((vorticity_mesh.template get<vorticity_m>(loopNodeM)[0] * ((0.5 / velocity_mesh.spacing(0)) * (velocity_mesh.template get<velocity_m>(loopNodeM.move(0, 1))[0] - velocity_mesh.template get<velocity_m>(loopNodeM.move(0, -1))[0]))) + (vorticity_mesh.template get<vorticity_m>(loopNodeM)[1] * ((0.5 / velocity_mesh.spacing(1)) * (velocity_mesh.template get<velocity_m>(loopNodeM.move(1, 1))[0] - velocity_mesh.template get<velocity_m>(loopNodeM.move(1, -1))[0])))) + (vorticity_mesh.template get<vorticity_m>(loopNodeM)[2] * ((0.5 / velocity_mesh.spacing(2)) * (velocity_mesh.template get<velocity_m>(loopNodeM.move(2, 1))[0] - velocity_mesh.template get<velocity_m>(loopNodeM.move(2, -1))[0])))) + (nu * (((((vorticity_mesh.template get<vorticity_m>(loopNodeM.move(0, 1))[0] + vorticity_mesh.template get<vorticity_m>(loopNodeM.move(0, -1))[0]) / (vorticity_mesh.spacing(0) * vorticity_mesh.spacing(0))) + ((vorticity_mesh.template get<vorticity_m>(loopNodeM.move(1, 1))[0] + vorticity_mesh.template get<vorticity_m>(loopNodeM.move(1, -1))[0]) / (vorticity_mesh.spacing(1) * vorticity_mesh.spacing(1)))) + ((vorticity_mesh.template get<vorticity_m>(loopNodeM.move(2, 1))[0] + vorticity_mesh.template get<vorticity_m>(loopNodeM.move(2, -1))[0]) / (vorticity_mesh.spacing(2) * vorticity_mesh.spacing(2)))) - (2 * (vorticity_mesh.template get<vorticity_m>(loopNodeM)[0] * (((1 / (vorticity_mesh.spacing(0) * vorticity_mesh.spacing(0))) + (1 / (vorticity_mesh.spacing(1) * vorticity_mesh.spacing(1)))) + (1 / (vorticity_mesh.spacing(2) * vorticity_mesh.spacing(2)))))))));
      rhs.template get<vortex_stretching_m>(loopNodeM)[1] = ((((vorticity_mesh.template get<vorticity_m>(loopNodeM)[0] * ((0.5 / velocity_mesh.spacing(0)) * (velocity_mesh.template get<velocity_m>(loopNodeM.move(0, 1))[1] - velocity_mesh.template get<velocity_m>(loopNodeM.move(0, -1))[1]))) + (vorticity_mesh.template get<vorticity_m>(loopNodeM)[1] * ((0.5 / velocity_mesh.spacing(1)) * (velocity_mesh.template get<velocity_m>(loopNodeM.move(1, 1))[1] - velocity_mesh.template get<velocity_m>(loopNodeM.move(1, -1))[1])))) + (vorticity_mesh.template get<vorticity_m>(loopNodeM)[2] * ((0.5 / velocity_mesh.spacing(2)) * (velocity_mesh.template get<velocity_m>(loopNodeM.move(2, 1))[1] - velocity_mesh.template get<velocity_m>(loopNodeM.move(2, -1))[1])))) + (nu * (((((vorticity_mesh.template get<vorticity_m>(loopNodeM.move(0, 1))[1] + vorticity_mesh.template get<vorticity_m>(loopNodeM.move(0, -1))[1]) / (vorticity_mesh.spacing(0) * vorticity_mesh.spacing(0))) + ((vorticity_mesh.template get<vorticity_m>(loopNodeM.move(1, 1))[1] + vorticity_mesh.template get<vorticity_m>(loopNodeM.move(1, -1))[1]) / (vorticity_mesh.spacing(1) * vorticity_mesh.spacing(1)))) + ((vorticity_mesh.template get<vorticity_m>(loopNodeM.move(2, 1))[1] + vorticity_mesh.template get<vorticity_m>(loopNodeM.move(2, -1))[1]) / (vorticity_mesh.spacing(2) * vorticity_mesh.spacing(2)))) - (2 * (vorticity_mesh.template get<vorticity_m>(loopNodeM)[1] * (((1 / (vorticity_mesh.spacing(0) * vorticity_mesh.spacing(0))) + (1 / (vorticity_mesh.spacing(1) * vorticity_mesh.spacing(1)))) + (1 / (vorticity_mesh.spacing(2) * vorticity_mesh.spacing(2)))))))));
      rhs.template get<vortex_stretching_m>(loopNodeM)[2] = ((((vorticity_mesh.template get<vorticity_m>(loopNodeM)[0] * ((0.5 / velocity_mesh.spacing(0)) * (velocity_mesh.template get<velocity_m>(loopNodeM.move(0, 1))[2] - velocity_mesh.template get<velocity_m>(loopNodeM.move(0, -1))[2]))) + (vorticity_mesh.template get<vorticity_m>(loopNodeM)[1] * ((0.5 / velocity_mesh.spacing(1)) * (velocity_mesh.template get<velocity_m>(loopNodeM.move(1, 1))[2] - velocity_mesh.template get<velocity_m>(loopNodeM.move(1, -1))[2])))) + (vorticity_mesh.template get<vorticity_m>(loopNodeM)[2] * ((0.5 / velocity_mesh.spacing(2)) * (velocity_mesh.template get<velocity_m>(loopNodeM.move(2, 1))[2] - velocity_mesh.template get<velocity_m>(loopNodeM.move(2, -1))[2])))) + (nu * (((((vorticity_mesh.template get<vorticity_m>(loopNodeM.move(0, 1))[2] + vorticity_mesh.template get<vorticity_m>(loopNodeM.move(0, -1))[2]) / (vorticity_mesh.spacing(0) * vorticity_mesh.spacing(0))) + ((vorticity_mesh.template get<vorticity_m>(loopNodeM.move(1, 1))[2] + vorticity_mesh.template get<vorticity_m>(loopNodeM.move(1, -1))[2]) / (vorticity_mesh.spacing(1) * vorticity_mesh.spacing(1)))) + ((vorticity_mesh.template get<vorticity_m>(loopNodeM.move(2, 1))[2] + vorticity_mesh.template get<vorticity_m>(loopNodeM.move(2, -1))[2]) / (vorticity_mesh.spacing(2) * vorticity_mesh.spacing(2)))) - (2 * (vorticity_mesh.template get<vorticity_m>(loopNodeM)[2] * (((1 / (vorticity_mesh.spacing(0) * vorticity_mesh.spacing(0))) + (1 / (vorticity_mesh.spacing(1) * vorticity_mesh.spacing(1)))) + (1 / (vorticity_mesh.spacing(2) * vorticity_mesh.spacing(2)))))))));
      ++mloop_iterator_e5a0;
    }
    
    // Particle loop
    auto ploop_iterator_f5a0 = particles.getDomainIterator();
    while (ploop_iterator_f5a0.isNext())
    {
      auto particleIterator = ploop_iterator_f5a0.get();
      particles.template getProp<vortex_stretching>(particleIterator)[0] = 0.0;
      particles.template getProp<vortex_stretching>(particleIterator)[1] = 0.0;
      particles.template getProp<vortex_stretching>(particleIterator)[2] = 0.0;
      ++ploop_iterator_f5a0;
    }
    
    // Particle loop
    auto ploop_iterator_g5a0 = particles.getDomainIterator();
    while (ploop_iterator_g5a0.isNext())
    {
      auto particleIterator = ploop_iterator_g5a0.get();
      particles.template getProp<velocity>(particleIterator)[0] = 0.0;
      particles.template getProp<velocity>(particleIterator)[1] = 0.0;
      particles.template getProp<velocity>(particleIterator)[2] = 0.0;
      ++ploop_iterator_g5a0;
    }
    interpol.template m2p<vortex_stretching_m, vortex_stretching>(rhs, particles);
    interpol.template m2p<velocity_m, velocity>(velocity_mesh, particles);
    
    // Particle loop
    auto ploop_iterator_j5a0 = particles.getDomainIterator();
    while (ploop_iterator_j5a0.isNext())
    {
      auto particleIterator = ploop_iterator_j5a0.get();
      particles.template getProp<old_vorticity>(particleIterator)[0] = particles.template getProp<vorticity>(particleIterator)[0];
      particles.template getProp<old_vorticity>(particleIterator)[1] = particles.template getProp<vorticity>(particleIterator)[1];
      particles.template getProp<old_vorticity>(particleIterator)[2] = particles.template getProp<vorticity>(particleIterator)[2];
      ++ploop_iterator_j5a0;
    }
    
    // Particle loop
    auto ploop_iterator_k5a0 = particles.getDomainIterator();
    while (ploop_iterator_k5a0.isNext())
    {
      auto particleIterator = ploop_iterator_k5a0.get();
      particles.template getProp<vorticity>(particleIterator)[0] = (particles.template getProp<vorticity>(particleIterator)[0] + (0.5 * (dt * particles.template getProp<vortex_stretching>(particleIterator)[0])));
      particles.template getProp<vorticity>(particleIterator)[1] = (particles.template getProp<vorticity>(particleIterator)[1] + (0.5 * (dt * particles.template getProp<vortex_stretching>(particleIterator)[1])));
      particles.template getProp<vorticity>(particleIterator)[2] = (particles.template getProp<vorticity>(particleIterator)[2] + (0.5 * (dt * particles.template getProp<vortex_stretching>(particleIterator)[2])));
      ++ploop_iterator_k5a0;
    }
    
    // Particle loop
    auto ploop_iterator_l5a0 = particles.getDomainIterator();
    while (ploop_iterator_l5a0.isNext())
    {
      auto particleIterator = ploop_iterator_l5a0.get();
      particles.template getProp<old_position>(particleIterator)[0] = particles.getPos(particleIterator)[0];
      particles.template getProp<old_position>(particleIterator)[1] = particles.getPos(particleIterator)[1];
      particles.template getProp<old_position>(particleIterator)[2] = particles.getPos(particleIterator)[2];
      ++ploop_iterator_l5a0;
    }
    
    // Particle loop
    auto ploop_iterator_m5a0 = particles.getDomainIterator();
    while (ploop_iterator_m5a0.isNext())
    {
      auto particleIterator = ploop_iterator_m5a0.get();
      particles.getPos(particleIterator)[0] = (particles.getPos(particleIterator)[0] + (0.5 * (dt * particles.template getProp<velocity>(particleIterator)[0])));
      particles.getPos(particleIterator)[1] = (particles.getPos(particleIterator)[1] + (0.5 * (dt * particles.template getProp<velocity>(particleIterator)[1])));
      particles.getPos(particleIterator)[2] = (particles.getPos(particleIterator)[2] + (0.5 * (dt * particles.template getProp<velocity>(particleIterator)[2])));
      ++ploop_iterator_m5a0;
    }
    particles.map();
    particles.template ghost_get<>();
    
    // Mesh loop
    auto mloop_iterator_p5a0 = vorticity_mesh.getDomainIterator();
    while (mloop_iterator_p5a0.isNext())
    {
      auto loopNodeM = mloop_iterator_p5a0.get();
      vorticity_mesh.template get<vorticity_m>(loopNodeM)[0] = 0.0;
      vorticity_mesh.template get<vorticity_m>(loopNodeM)[1] = 0.0;
      vorticity_mesh.template get<vorticity_m>(loopNodeM)[2] = 0.0;
      ++mloop_iterator_p5a0;
    }
    interpol.template p2m<vorticity, vorticity_m>(particles, vorticity_mesh);
    
    // Solve poisson equation
    for (size_t dim_poisson = 0; dim_poisson < 3; dim_poisson++)
    {
      auto poisson_copy_prop_iterator = psi_poisson.getDomainIterator();
      while (poisson_copy_prop_iterator.isNext())
      {
        auto poisson_copy_node = poisson_copy_prop_iterator.get();
        psi_poisson.get<phi_poisson>(poisson_copy_node) = vorticity_mesh.template get<vorticity_m>(poisson_copy_node)[dim_poisson];
        ++poisson_copy_prop_iterator;
      }
      fd_poisson.new_b();
      fd_poisson.template impose_dit_b<phi_poisson>(psi_poisson, psi_poisson.getDomainIterator());
      poisson_solver.setAbsTol(0.01);
      Vector<double, PETSC_BASE> &poisson_b = fd_poisson.getB();
      poisson_solver.solve(phi_solution_poisson[dim_poisson], poisson_b);
      fd_poisson.template copy<phi_poisson>(phi_solution_poisson[dim_poisson], psi_poisson);
      auto poisson_copy_prop_back_iterator = psi_poisson.getDomainIterator();
      while (poisson_copy_prop_back_iterator.isNext())
      {
        auto poisson_copy_back_node = poisson_copy_prop_back_iterator.get();
        phi.get<velocity_phi>(poisson_copy_back_node)[dim_poisson] = psi_poisson.get<phi_poisson>(poisson_copy_back_node);
        ++poisson_copy_prop_back_iterator;
      }
    }
    phi.ghost_get<velocity_phi>();
    
    // Mesh loop
    auto mloop_iterator_s5a0 = velocity_mesh.getDomainIterator();
    while (mloop_iterator_s5a0.isNext())
    {
      auto loopNodeM = mloop_iterator_s5a0.get();
      velocity_mesh.template get<velocity_m>(loopNodeM)[0] = (((phi.template get<velocity_phi>(loopNodeM.move(1, 1))[2] - phi.template get<velocity_phi>(loopNodeM.move(1, -1))[2]) * (0.5 / phi.spacing(1))) - ((phi.template get<velocity_phi>(loopNodeM.move(2, 1))[1] - phi.template get<velocity_phi>(loopNodeM.move(2, -1))[1]) * (0.5 / phi.spacing(2))));
      velocity_mesh.template get<velocity_m>(loopNodeM)[1] = (((phi.template get<velocity_phi>(loopNodeM.move(2, 1))[0] - phi.template get<velocity_phi>(loopNodeM.move(2, -1))[0]) * (0.5 / phi.spacing(2))) - ((phi.template get<velocity_phi>(loopNodeM.move(0, 1))[2] - phi.template get<velocity_phi>(loopNodeM.move(0, -1))[2]) * (0.5 / phi.spacing(0))));
      velocity_mesh.template get<velocity_m>(loopNodeM)[2] = (((phi.template get<velocity_phi>(loopNodeM.move(0, 1))[1] - phi.template get<velocity_phi>(loopNodeM.move(0, -1))[1]) * (0.5 / phi.spacing(0))) - ((phi.template get<velocity_phi>(loopNodeM.move(1, 1))[0] - phi.template get<velocity_phi>(loopNodeM.move(1, -1))[0]) * (0.5 / phi.spacing(1))));
      ++mloop_iterator_s5a0;
    }
    
    // Mesh loop
    auto mloop_iterator_t5a0 = rhs.getDomainIterator();
    while (mloop_iterator_t5a0.isNext())
    {
      auto loopNodeM = mloop_iterator_t5a0.get();
      rhs.template get<vortex_stretching_m>(loopNodeM)[0] = ((((vorticity_mesh.template get<vorticity_m>(loopNodeM)[0] * ((0.5 / velocity_mesh.spacing(0)) * (velocity_mesh.template get<velocity_m>(loopNodeM.move(0, 1))[0] - velocity_mesh.template get<velocity_m>(loopNodeM.move(0, -1))[0]))) + (vorticity_mesh.template get<vorticity_m>(loopNodeM)[1] * ((0.5 / velocity_mesh.spacing(1)) * (velocity_mesh.template get<velocity_m>(loopNodeM.move(1, 1))[0] - velocity_mesh.template get<velocity_m>(loopNodeM.move(1, -1))[0])))) + (vorticity_mesh.template get<vorticity_m>(loopNodeM)[2] * ((0.5 / velocity_mesh.spacing(2)) * (velocity_mesh.template get<velocity_m>(loopNodeM.move(2, 1))[0] - velocity_mesh.template get<velocity_m>(loopNodeM.move(2, -1))[0])))) + (nu * (((((vorticity_mesh.template get<vorticity_m>(loopNodeM.move(0, 1))[0] + vorticity_mesh.template get<vorticity_m>(loopNodeM.move(0, -1))[0]) / (vorticity_mesh.spacing(0) * vorticity_mesh.spacing(0))) + ((vorticity_mesh.template get<vorticity_m>(loopNodeM.move(1, 1))[0] + vorticity_mesh.template get<vorticity_m>(loopNodeM.move(1, -1))[0]) / (vorticity_mesh.spacing(1) * vorticity_mesh.spacing(1)))) + ((vorticity_mesh.template get<vorticity_m>(loopNodeM.move(2, 1))[0] + vorticity_mesh.template get<vorticity_m>(loopNodeM.move(2, -1))[0]) / (vorticity_mesh.spacing(2) * vorticity_mesh.spacing(2)))) - (2 * (vorticity_mesh.template get<vorticity_m>(loopNodeM)[0] * (((1 / (vorticity_mesh.spacing(0) * vorticity_mesh.spacing(0))) + (1 / (vorticity_mesh.spacing(1) * vorticity_mesh.spacing(1)))) + (1 / (vorticity_mesh.spacing(2) * vorticity_mesh.spacing(2)))))))));
      rhs.template get<vortex_stretching_m>(loopNodeM)[1] = ((((vorticity_mesh.template get<vorticity_m>(loopNodeM)[0] * ((0.5 / velocity_mesh.spacing(0)) * (velocity_mesh.template get<velocity_m>(loopNodeM.move(0, 1))[1] - velocity_mesh.template get<velocity_m>(loopNodeM.move(0, -1))[1]))) + (vorticity_mesh.template get<vorticity_m>(loopNodeM)[1] * ((0.5 / velocity_mesh.spacing(1)) * (velocity_mesh.template get<velocity_m>(loopNodeM.move(1, 1))[1] - velocity_mesh.template get<velocity_m>(loopNodeM.move(1, -1))[1])))) + (vorticity_mesh.template get<vorticity_m>(loopNodeM)[2] * ((0.5 / velocity_mesh.spacing(2)) * (velocity_mesh.template get<velocity_m>(loopNodeM.move(2, 1))[1] - velocity_mesh.template get<velocity_m>(loopNodeM.move(2, -1))[1])))) + (nu * (((((vorticity_mesh.template get<vorticity_m>(loopNodeM.move(0, 1))[1] + vorticity_mesh.template get<vorticity_m>(loopNodeM.move(0, -1))[1]) / (vorticity_mesh.spacing(0) * vorticity_mesh.spacing(0))) + ((vorticity_mesh.template get<vorticity_m>(loopNodeM.move(1, 1))[1] + vorticity_mesh.template get<vorticity_m>(loopNodeM.move(1, -1))[1]) / (vorticity_mesh.spacing(1) * vorticity_mesh.spacing(1)))) + ((vorticity_mesh.template get<vorticity_m>(loopNodeM.move(2, 1))[1] + vorticity_mesh.template get<vorticity_m>(loopNodeM.move(2, -1))[1]) / (vorticity_mesh.spacing(2) * vorticity_mesh.spacing(2)))) - (2 * (vorticity_mesh.template get<vorticity_m>(loopNodeM)[1] * (((1 / (vorticity_mesh.spacing(0) * vorticity_mesh.spacing(0))) + (1 / (vorticity_mesh.spacing(1) * vorticity_mesh.spacing(1)))) + (1 / (vorticity_mesh.spacing(2) * vorticity_mesh.spacing(2)))))))));
      rhs.template get<vortex_stretching_m>(loopNodeM)[2] = ((((vorticity_mesh.template get<vorticity_m>(loopNodeM)[0] * ((0.5 / velocity_mesh.spacing(0)) * (velocity_mesh.template get<velocity_m>(loopNodeM.move(0, 1))[2] - velocity_mesh.template get<velocity_m>(loopNodeM.move(0, -1))[2]))) + (vorticity_mesh.template get<vorticity_m>(loopNodeM)[1] * ((0.5 / velocity_mesh.spacing(1)) * (velocity_mesh.template get<velocity_m>(loopNodeM.move(1, 1))[2] - velocity_mesh.template get<velocity_m>(loopNodeM.move(1, -1))[2])))) + (vorticity_mesh.template get<vorticity_m>(loopNodeM)[2] * ((0.5 / velocity_mesh.spacing(2)) * (velocity_mesh.template get<velocity_m>(loopNodeM.move(2, 1))[2] - velocity_mesh.template get<velocity_m>(loopNodeM.move(2, -1))[2])))) + (nu * (((((vorticity_mesh.template get<vorticity_m>(loopNodeM.move(0, 1))[2] + vorticity_mesh.template get<vorticity_m>(loopNodeM.move(0, -1))[2]) / (vorticity_mesh.spacing(0) * vorticity_mesh.spacing(0))) + ((vorticity_mesh.template get<vorticity_m>(loopNodeM.move(1, 1))[2] + vorticity_mesh.template get<vorticity_m>(loopNodeM.move(1, -1))[2]) / (vorticity_mesh.spacing(1) * vorticity_mesh.spacing(1)))) + ((vorticity_mesh.template get<vorticity_m>(loopNodeM.move(2, 1))[2] + vorticity_mesh.template get<vorticity_m>(loopNodeM.move(2, -1))[2]) / (vorticity_mesh.spacing(2) * vorticity_mesh.spacing(2)))) - (2 * (vorticity_mesh.template get<vorticity_m>(loopNodeM)[2] * (((1 / (vorticity_mesh.spacing(0) * vorticity_mesh.spacing(0))) + (1 / (vorticity_mesh.spacing(1) * vorticity_mesh.spacing(1)))) + (1 / (vorticity_mesh.spacing(2) * vorticity_mesh.spacing(2)))))))));
      ++mloop_iterator_t5a0;
    }
    
    // Particle loop
    auto ploop_iterator_u5a0 = particles.getDomainIterator();
    while (ploop_iterator_u5a0.isNext())
    {
      auto particleIterator = ploop_iterator_u5a0.get();
      particles.template getProp<vortex_stretching>(particleIterator)[0] = 0.0;
      particles.template getProp<vortex_stretching>(particleIterator)[1] = 0.0;
      particles.template getProp<vortex_stretching>(particleIterator)[2] = 0.0;
      ++ploop_iterator_u5a0;
    }
    
    // Particle loop
    auto ploop_iterator_v5a0 = particles.getDomainIterator();
    while (ploop_iterator_v5a0.isNext())
    {
      auto particleIterator = ploop_iterator_v5a0.get();
      particles.template getProp<velocity>(particleIterator)[0] = 0.0;
      particles.template getProp<velocity>(particleIterator)[1] = 0.0;
      particles.template getProp<velocity>(particleIterator)[2] = 0.0;
      ++ploop_iterator_v5a0;
    }
    interpol.template m2p<vortex_stretching_m, vortex_stretching>(rhs, particles);
    interpol.template m2p<velocity_m, velocity>(velocity_mesh, particles);
    
    // Particle loop
    auto ploop_iterator_y5a0 = particles.getDomainIterator();
    while (ploop_iterator_y5a0.isNext())
    {
      auto particleIterator = ploop_iterator_y5a0.get();
      particles.template getProp<vorticity>(particleIterator)[0] = (particles.template getProp<old_vorticity>(particleIterator)[0] + (0.5 * (dt * particles.template getProp<vortex_stretching>(particleIterator)[0])));
      particles.template getProp<vorticity>(particleIterator)[1] = (particles.template getProp<old_vorticity>(particleIterator)[1] + (0.5 * (dt * particles.template getProp<vortex_stretching>(particleIterator)[1])));
      particles.template getProp<vorticity>(particleIterator)[2] = (particles.template getProp<old_vorticity>(particleIterator)[2] + (0.5 * (dt * particles.template getProp<vortex_stretching>(particleIterator)[2])));
      ++ploop_iterator_y5a0;
    }
    
    // Particle loop
    auto ploop_iterator_z5a0 = particles.getDomainIterator();
    while (ploop_iterator_z5a0.isNext())
    {
      auto particleIterator = ploop_iterator_z5a0.get();
      particles.getPos(particleIterator)[0] = (particles.template getProp<old_position>(particleIterator)[0] + (0.5 * (dt * particles.template getProp<velocity>(particleIterator)[0])));
      particles.getPos(particleIterator)[1] = (particles.template getProp<old_position>(particleIterator)[1] + (0.5 * (dt * particles.template getProp<velocity>(particleIterator)[1])));
      particles.getPos(particleIterator)[2] = (particles.template getProp<old_position>(particleIterator)[2] + (0.5 * (dt * particles.template getProp<velocity>(particleIterator)[2])));
      ++ploop_iterator_z5a0;
    }
    particles.map();
    particles.template ghost_get<>();
    
    // Mesh loop
    auto mloop_iterator_cb5a0 = vorticity_mesh.getDomainIterator();
    while (mloop_iterator_cb5a0.isNext())
    {
      auto loopNodeM = mloop_iterator_cb5a0.get();
      vorticity_mesh.template get<vorticity>(loopNodeM)[0] = 0.0;
      vorticity_mesh.template get<vorticity>(loopNodeM)[1] = 0.0;
      vorticity_mesh.template get<vorticity>(loopNodeM)[2] = 0.0;
      ++mloop_iterator_cb5a0;
    }
    interpol.template p2m<vorticity, vorticity_m>(particles, vorticity_mesh);
    // Remesh
    particles.clear();
    auto remesh_iterator_eb5a0 = particles.getGridIterator(vorticity_mesh.getGridInfo().getSize());
    while (remesh_iterator_eb5a0.isNext())
    {
      auto local_key_eb5a0 = remesh_iterator_eb5a0.get();
      auto global_key_eb5a0 = remesh_iterator_eb5a0.get_dist();
      particles.add();
      particles.getLastPos()[0] = vorticity_mesh.spacing(0) * local_key_eb5a0.get(0) + box.getLow(0);
      particles.getLastPos()[1] = vorticity_mesh.spacing(1) * local_key_eb5a0.get(1) + box.getLow(1);
      particles.getLastPos()[2] = vorticity_mesh.spacing(2) * local_key_eb5a0.get(2) + box.getLow(2);
      particles.template getLastProp<vorticity>()[0] = vorticity_mesh.template get<vorticity_m>(global_key_eb5a0)[0];
      particles.template getLastProp<vorticity>()[1] = vorticity_mesh.template get<vorticity_m>(global_key_eb5a0)[1];
      particles.template getLastProp<vorticity>()[2] = vorticity_mesh.template get<vorticity_m>(global_key_eb5a0)[2];
      ++remesh_iterator_eb5a0;
    }
    particles.map();
    if (i % 50 == 0)
    {
      particles.deleteGhost();
      particles.write("particles");
    }
    i += 1;
  }
}

  openfpm_finalize();
}

