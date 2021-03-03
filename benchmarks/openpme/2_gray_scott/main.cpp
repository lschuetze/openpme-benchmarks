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
constexpr unsigned int Uo = 0;
constexpr unsigned int Vo = 1;
constexpr unsigned int Un = 0;
constexpr unsigned int Vn = 1;


int main(int argc, char* argv[])
{
  
  openfpm_init(&argc, &argv);
  
  // Initialization
  double r_cut = 1;
  Box<3, double> box({0.0, 0.0, 0.0}, {2.5, 2.5, 2.5});
   size_t bc_particle[3] = {PERIODIC, PERIODIC, PERIODIC};
  periodicity<3> bc_grid = {{PERIODIC, PERIODIC, PERIODIC}};
  Ghost<3, long int> ghost(r_cut);
  
  // Mesh-based Simulation
  
  // Field containers
   size_t sz_Old[3] = {128, 128, 128};
  grid_dist_id<3, double, aggregate<double, double>> Old(sz_Old, box, ghost, bc_grid);
   size_t sz_New[3] = {128, 128, 128};
  grid_dist_id<3, double, aggregate<double, double>> New(sz_New, box, ghost, bc_grid);
  
  // Commands
  double dt = 1.0;
  double du = (2.0 * 1e-5);
  double dv = 1e-5;
  double K = 0.053;
  double F = 0.014;
  
  // Mesh loop
  auto mloop_iterator_f0a = New.getDomainIterator();
  while (mloop_iterator_f0a.isNext())
  {
    auto loopNodeM = mloop_iterator_f0a.get();
    New.template get<Un>(loopNodeM) = 0.0;
    ++mloop_iterator_f0a;
  }
  
  // Mesh loop
  auto mloop_iterator_g0a = New.getDomainIterator();
  while (mloop_iterator_g0a.isNext())
  {
    auto loopNodeM = mloop_iterator_g0a.get();
    New.template get<Vn>(loopNodeM) = 0.0;
    ++mloop_iterator_g0a;
  }
  Old.load("./2_gray_scott/init_mesh.hdf5");
  Old.template ghost_get<Uo, Vo>();
  int i = 0;
  
  // Time loop
  for (size_t time_step = 0; time_step < 5000; time_step++)
  {
    
    // Mesh loop
    auto mloop_iterator_a01a0 = New.getDomainIterator();
    while (mloop_iterator_a01a0.isNext())
    {
      auto loopNodeM = mloop_iterator_a01a0.get();
      New.template get<Un>(loopNodeM) = (Old.template get<Uo>(loopNodeM) + (dt * ((du * (((((Old.template get<Uo>(loopNodeM.move(0, 1)) + Old.template get<Uo>(loopNodeM.move(0, -1))) / (Old.spacing(0) * Old.spacing(0))) + ((Old.template get<Uo>(loopNodeM.move(1, 1)) + Old.template get<Uo>(loopNodeM.move(1, -1))) / (Old.spacing(1) * Old.spacing(1)))) + ((Old.template get<Uo>(loopNodeM.move(2, 1)) + Old.template get<Uo>(loopNodeM.move(2, -1))) / (Old.spacing(2) * Old.spacing(2)))) - (2 * (Old.template get<Uo>(loopNodeM) * (((1 / (Old.spacing(0) * Old.spacing(0))) + (1 / (Old.spacing(1) * Old.spacing(1)))) + (1 / (Old.spacing(2) * Old.spacing(2)))))))) - ((Old.template get<Uo>(loopNodeM) * (Old.template get<Vo>(loopNodeM) * Old.template get<Vo>(loopNodeM))) - (F * (1.0 - Old.template get<Uo>(loopNodeM)))))));
      ++mloop_iterator_a01a0;
    }
    
    // Mesh loop
    auto mloop_iterator_b01a0 = New.getDomainIterator();
    while (mloop_iterator_b01a0.isNext())
    {
      auto loopNodeM = mloop_iterator_b01a0.get();
      New.template get<Vn>(loopNodeM) = (Old.template get<Vo>(loopNodeM) + (dt * ((dv * (((((Old.template get<Vo>(loopNodeM.move(0, 1)) + Old.template get<Vo>(loopNodeM.move(0, -1))) / (Old.spacing(0) * Old.spacing(0))) + ((Old.template get<Vo>(loopNodeM.move(1, 1)) + Old.template get<Vo>(loopNodeM.move(1, -1))) / (Old.spacing(1) * Old.spacing(1)))) + ((Old.template get<Vo>(loopNodeM.move(2, 1)) + Old.template get<Vo>(loopNodeM.move(2, -1))) / (Old.spacing(2) * Old.spacing(2)))) - (2 * (Old.template get<Vo>(loopNodeM) * (((1 / (Old.spacing(0) * Old.spacing(0))) + (1 / (Old.spacing(1) * Old.spacing(1)))) + (1 / (Old.spacing(2) * Old.spacing(2)))))))) + ((Old.template get<Uo>(loopNodeM) * (Old.template get<Vo>(loopNodeM) * Old.template get<Vo>(loopNodeM))) - ((F + K) * Old.template get<Vo>(loopNodeM))))));
      ++mloop_iterator_b01a0;
    }
    Old.copy(New);
    Old.template ghost_get<Uo, Vo>();
    if (i % 500 == 0)
    {
      Old.write("mesh" + std::to_string(time_step));
    }
    i += 1;
  }
  openfpm_finalize();
}

