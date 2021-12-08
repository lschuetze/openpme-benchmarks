
module {
  func @grayScottCpu(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) {
    %0 = stencil.cast %arg0([0, 0, 0] : [256, 256, 256]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<?x?x?xf64>
    %1 = stencil.cast %arg1([0, 0, 0] : [256, 256, 256]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<?x?x?xf64>
    // launch the kernel
    // not sure about the arguments for the launch_func, in the paper not very clear what they call, check that again
    gpu.launch_func(%0,%1)
    {kernel = "kernel", kernel_module = @outlined} 
    : (!stencil.field<?x?x?xf64>, !stencil.field<?x?x?xf64>) -> ()

  }
 // GPU function outlined 
gpu.module @outlined {
    // kernel outlined to separate module
    gpu.func @kernel((%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>))
      workgroup(%shared_buffer: memref<32 xf64, 3>),
      private(%private_buffer: memref<32 xf64, 3>){
      // Here we implement the kernel as a separate function
      %0 = stencil.cast %arg0([0, 0, 0] : [256, 256, 256]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<?x?x?xf64>
      %1 = stencil.cast %arg1([0, 0, 0] : [256, 256, 256]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<?x?x?xf64>
      %2 = stencil.load %0 : (!stencil.field<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    // loop over the stencil
      %3 = stencil.apply (%arg2 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      
      // Operations inside the loop over the stencil equivalent to scf.for or scf.parallel in MLIR
      // This computation is to be implemented
      //u_out = uc + uFactor *(u(i-1,j,k) + u(i+1,j,k) +
      //                                                 u(i,j-1,k) + u(i,j+1,k) +
      //                                                 u(i,j,k-1) + u(i,j,k+1) - 6.0*uc) - deltaT * uc*vc*vc
      //                                                 - deltaT * F * (uc - 1.0);


			//v_out = vc + vFactor *(v(i-1,j,k) + v(i+1,j,k) +
      //                                                 v(i,j+1,k) + v(i,j-1,k) +
      //                                                 v(i,j,k-1) + v(i,j,k+1) - 6.0*vc) + deltaT * uc*vc*vc
			//		               - deltaT * (F+K) * vc;
      

      // Operations inside the loop over the stencil equivalent to scf.for or scf.parallel in MLIR

      //%9 = addf %4, %5 : f64
      //%10 = addf %6, %7 : f64
      //%11 = addf %9, %10 : f64
      // define constants
      %deltaT = constant 0.25 : f64
	    // Diffusion constant for specie U
	    %du = constant 2*1e-5 : f64
      // Diffusion constant for specie V
	    %dv = constant 1*1e-5 : f64
      // Spacing
      %space = constant 0.01 : f64
      // calcul
      //float uFactor = deltaT * du/(spacing[x]*spacing[x]);
	    //float vFactor = deltaT * dv/(spacing[x]*spacing[x]);
      %tempu = divf %du, %space : f64
      %tempv = divf %dv, %space : f64
      %uFactor = mulf %deltaT, %tempu : f64
      %vFactor = mulf %deltaT, %tempv : f64

      // Unrolling the stencil in all dimensions
      %4 = stencil.access %arg2 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %5 = stencil.access %arg2 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %6 = stencil.access %arg2 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %7 = stencil.access %arg2 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %8 = stencil.access %arg2 [0, 0, 1] : (!stencil.temp<?x?x?xf64>) -> f64
      %9 = stencil.access %arg2 [0, 0, -1] : (!stencil.temp<?x?x?xf64>) -> f64
      
      

      %14 = stencil.store_result %13 : (f64) -> !stencil.result<f64>
      stencil.return %14 : !stencil.result<f64>
    }
    stencil.store %3 to %1([0, 0, 0] : [256, 256, 256]) : !stencil.temp<?x?x?xf64> to !stencil.field<?x?x?xf64>
    return



      }
    
  }

}
