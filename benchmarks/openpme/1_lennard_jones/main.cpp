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
constexpr unsigned int velocity = 0;
constexpr unsigned int force = 1;


int main(int argc, char* argv[])
{
  
  openfpm_init(&argc, &argv);
  
  // Initialization
  double r_cut = 0.3;
  Box<3, double> box({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
   size_t bc_particle[3] = {PERIODIC, PERIODIC, PERIODIC};
  periodicity<3> bc_grid = {{PERIODIC, PERIODIC, PERIODIC}};
  Ghost<3, double> ghost(r_cut);
  
  // Discrete Simulation
  
  // Field containers
  vector_dist<3, double, aggregate<double[3], double[3]>> particles(0, box, bc_particle, ghost);
  
  // Init particles
  
  // Initialize and create particle grid on vector_dist
  // works in 3D only right now
   size_t sz_a0a[3] = {10, 10, 10};
  auto grid_iterator_init_a0a = particles.getGridIterator(sz_a0a);
  while (grid_iterator_init_a0a.isNext())
  {
    particles.add();
    auto key = grid_iterator_init_a0a.get();
    
    particles.getLastPos()[0] = key.get(0) * grid_iterator_init_a0a.getSpacing(0);
    particles.getLastPos()[1] = key.get(1) * grid_iterator_init_a0a.getSpacing(1);
    particles.getLastPos()[2] = key.get(2) * grid_iterator_init_a0a.getSpacing(2);
    
    particles.template getLastProp<velocity>()[0] = 0.0;
    particles.template getLastProp<force>()[0] = 0.0;
    particles.template getLastProp<velocity>()[1] = 0.0;
    particles.template getLastProp<force>()[1] = 0.0;
    particles.template getLastProp<velocity>()[2] = 0.0;
    particles.template getLastProp<force>()[2] = 0.0;
    
    ++grid_iterator_init_a0a;
  }
  
  particles.map();
  particles.ghost_get<>();
  
  // Commands
  auto particles_cellList_0 = particles.getCellList<CELL_MEMBAL(3, double)>(r_cut);
  double dt = 0.0005;
  double sigma = 0.1;
  double sigma6 = (((((sigma * sigma) * sigma) * sigma) * sigma) * sigma);
  double sigma12 = (((((((((((sigma * sigma) * sigma) * sigma) * sigma) * sigma) * sigma) * sigma) * sigma) * sigma) * sigma) * sigma);
  double r_cut2 = (0.3 * 0.3);
  double shift = (2.0 * ((sigma12 / (((((r_cut2 * r_cut2) * r_cut2) * r_cut2) * r_cut2) * r_cut2)) - (sigma6 / ((r_cut2 * r_cut2) * r_cut2))));
  int i = 0;
  double E = 0.0;
  particles.updateCellList(particles_cellList_0);
  
  // Particle loop
  auto ploop_iterator_k0a = particles.getDomainIterator();
  while (ploop_iterator_k0a.isNext())
  {
    auto p_force = ploop_iterator_k0a.get();
    Point<3, double> selfPosition = particles.getPos(p_force);
    auto nlist = particles_cellList_0.template getNNIterator<NO_CHECK>(particles_cellList_0.getCell(selfPosition));
    
    // Neighborhood loop
    while (nlist.isNext())
    {
      auto q_force = nlist.get();
      if (p_force.getKey() == q_force)
      {
        ++nlist;
        continue;
      }
      Point<3, double> neighborPosition = particles.getPos(q_force);
      Point<3, double> diff = (selfPosition - neighborPosition);
      double norm = norm2(diff);
      if (norm > (0.3 * 0.3))
      {
        ++nlist;
        continue;
      }
      double r = norm;
      particles.template getProp<force>(p_force)[0] += ((24.0 * (((2.0 * sigma12) / ((((((r * r) * r) * r) * r) * r) * r)) - (sigma6 / (((r * r) * r) * r)))) * diff.get(0));
      particles.template getProp<force>(p_force)[1] += ((24.0 * (((2.0 * sigma12) / ((((((r * r) * r) * r) * r) * r) * r)) - (sigma6 / (((r * r) * r) * r)))) * diff.get(1));
      particles.template getProp<force>(p_force)[2] += ((24.0 * (((2.0 * sigma12) / ((((((r * r) * r) * r) * r) * r) * r)) - (sigma6 / (((r * r) * r) * r)))) * diff.get(2));
      ++nlist;
    }
    ++ploop_iterator_k0a;
  }
  
  // Time loop
  for (size_t time_step = 0; time_step < 10000; time_step++)
  {
    
    // Particle loop
    auto ploop_iterator_a11a0 = particles.getDomainIterator();
    while (ploop_iterator_a11a0.isNext())
    {
      auto p_velocity = ploop_iterator_a11a0.get();
      particles.template getProp<velocity>(p_velocity)[0] += ((0.5 * dt) * particles.template getProp<force>(p_velocity)[0]);
      particles.template getProp<velocity>(p_velocity)[1] += ((0.5 * dt) * particles.template getProp<force>(p_velocity)[1]);
      particles.template getProp<velocity>(p_velocity)[2] += ((0.5 * dt) * particles.template getProp<force>(p_velocity)[2]);
      ++ploop_iterator_a11a0;
    }
    
    // Particle loop
    auto ploop_iterator_b11a0 = particles.getDomainIterator();
    while (ploop_iterator_b11a0.isNext())
    {
      auto p_position = ploop_iterator_b11a0.get();
      particles.getPos(p_position)[0] += (dt * particles.template getProp<velocity>(p_position)[0]);
      particles.getPos(p_position)[1] += (dt * particles.template getProp<velocity>(p_position)[1]);
      particles.getPos(p_position)[2] += (dt * particles.template getProp<velocity>(p_position)[2]);
      ++ploop_iterator_b11a0;
    }
    particles.map();
    particles.template ghost_get<>();
    
    // Particle loop
    auto ploop_iterator_e11a0 = particles.getDomainIterator();
    while (ploop_iterator_e11a0.isNext())
    {
      auto particleIterator = ploop_iterator_e11a0.get();
      particles.template getProp<force>(particleIterator)[0] = 0.0;
      particles.template getProp<force>(particleIterator)[1] = 0.0;
      particles.template getProp<force>(particleIterator)[2] = 0.0;
      ++ploop_iterator_e11a0;
    }
    particles.updateCellList(particles_cellList_0);
    
    // Particle loop
    auto ploop_iterator_g11a0 = particles.getDomainIterator();
    while (ploop_iterator_g11a0.isNext())
    {
      auto p_force = ploop_iterator_g11a0.get();
      Point<3, double> selfPosition = particles.getPos(p_force);
      auto nlist = particles_cellList_0.template getNNIterator<NO_CHECK>(particles_cellList_0.getCell(selfPosition));
      
      // Neighborhood loop
      while (nlist.isNext())
      {
        auto q_force = nlist.get();
        if (p_force.getKey() == q_force)
        {
          ++nlist;
          continue;
        }
        Point<3, double> neighborPosition = particles.getPos(q_force);
        Point<3, double> diff = (selfPosition - neighborPosition);
        double norm = norm2(diff);
        if (norm > (0.3 * 0.3))
        {
          ++nlist;
          continue;
        }
        double r = norm;
        particles.template getProp<force>(p_force)[0] += ((24.0 * (((2.0 * sigma12) / ((((((r * r) * r) * r) * r) * r) * r)) - (sigma6 / (((r * r) * r) * r)))) * diff.get(0));
        particles.template getProp<force>(p_force)[1] += ((24.0 * (((2.0 * sigma12) / ((((((r * r) * r) * r) * r) * r) * r)) - (sigma6 / (((r * r) * r) * r)))) * diff.get(1));
        particles.template getProp<force>(p_force)[2] += ((24.0 * (((2.0 * sigma12) / ((((((r * r) * r) * r) * r) * r) * r)) - (sigma6 / (((r * r) * r) * r)))) * diff.get(2));
        ++nlist;
      }
      ++ploop_iterator_g11a0;
    }
    
    // Particle loop
    auto ploop_iterator_h11a0 = particles.getDomainIterator();
    while (ploop_iterator_h11a0.isNext())
    {
      auto p_velocity = ploop_iterator_h11a0.get();
      particles.template getProp<velocity>(p_velocity)[0] += ((0.5 * dt) * particles.template getProp<force>(p_velocity)[0]);
      particles.template getProp<velocity>(p_velocity)[1] += ((0.5 * dt) * particles.template getProp<force>(p_velocity)[1]);
      particles.template getProp<velocity>(p_velocity)[2] += ((0.5 * dt) * particles.template getProp<force>(p_velocity)[2]);
      ++ploop_iterator_h11a0;
    }
    if (i % 100 == 0)
    {
      E = 0.0;
      
      // Particle loop
      auto ploop_iterator_b0i11a0 = particles.getDomainIterator();
      while (ploop_iterator_b0i11a0.isNext())
      {
        auto p_energy = ploop_iterator_b0i11a0.get();
        Point<3, double> xp_e = particles.getPos(p_energy);
        auto nlist = particles_cellList_0.template getNNIterator<NO_CHECK>(particles_cellList_0.getCell(xp_e));
        
        // Neighborhood loop
        while (nlist.isNext())
        {
          auto nlist_e = nlist.get();
          if (nlist_e == p_energy.getKey())
          {
            ++nlist;
            continue;
          }
          Point<3, double> xq_e = particles.getPos(nlist_e);
          Point<3, double> diff = (xp_e - xq_e);
          double rn_e = norm2(diff);
          if (rn_e > r_cut2)
          {
            ++nlist;
            continue;
          }
          E += ((2.0 * ((sigma12 / (((((rn_e * rn_e) * rn_e) * rn_e) * rn_e) * rn_e)) - (sigma6 / ((rn_e * rn_e) * rn_e)))) - shift);
          ++nlist;
        }
        ++ploop_iterator_b0i11a0;
      }
      
      // Particle loop
      auto ploop_iterator_c0i11a0 = particles.getDomainIterator();
      while (ploop_iterator_c0i11a0.isNext())
      {
        auto particleIterator = ploop_iterator_c0i11a0.get();
        E += (((particles.template getProp<velocity>(particleIterator)[0] * particles.template getProp<velocity>(particleIterator)[0]) + ((particles.template getProp<velocity>(particleIterator)[1] * particles.template getProp<velocity>(particleIterator)[1]) + (particles.template getProp<velocity>(particleIterator)[2] * particles.template getProp<velocity>(particleIterator)[2]))) / 2);
        ++ploop_iterator_c0i11a0;
      }
      auto &vcl_d0i11a0 = create_vcluster();
      vcl_d0i11a0.sum(E);
      vcl_d0i11a0.execute();
      if (vcl_d0i11a0.getProcessUnitID() == 0)
      {
        std::cout << "Output: " << E << std::endl;
      }
      particles.deleteGhost();
      particles.write("particles");
    }
    i += 1;
  }
  openfpm_finalize();
}

