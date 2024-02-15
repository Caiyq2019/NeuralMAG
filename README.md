<div align="center">
  <img width="500px" src="https://github.com/Caiyq2019/NN-MAG/blob/ad4c2cff5ade3bc209df62c28ca67bf96ff424a8/figs/magsimu.png"/>
</div>


## Introduction to NeuralMAG Project

The NeuralMAG Project is an open-source neural network framework designed for micromagnetic simulation. It employs a multi-scale U-Net neural network model to learn the physical relationship between magnetization states and demagnetizing fields, effectively extending traditional micromagnetic simulations into the realm of advanced deep learning frameworks. This approach fully leverages the cutting-edge technologies available on deep learning platforms.

![alt text](framework.png)

The project finds wide application in the study of magnetic materials among other areas.

## Capabilities of NeuralMAG

- **Integration with PyTorch Platform**: NeuralMAG integrates MAG micromagnetic simulations into the PyTorch framework, utilizing the extensive parallelization capabilities of GPUs for enhanced performance.
- **Cross-Scale Neural Network Simulation**: By employing a multi-scale Unet neural network architecture, the system can efficiently simulate magnetic material systems on a large scale.
- **Generalization Across Material Systems**: The model has been designed to generalize effectively to previously unseen material systems, thereby broadening the scope for exploration of new materials.
- **Versatility in Micromagnetic Simulation Tasks**: NeuralMAG is adept at performing a wide array of micromagnetic simulations, including but not limited to predicting magnetic ground states, simulating hysteresis loops, and accommodating arbitrary geometric shapes in simulations.
- **Utilization of Advanced Deep Learning Optimization Techniques**: The framework incorporates the latest advancements in model compression and acceleration optimization technologies that are prevalent in modern deep learning platforms.



## Getting Started

### Installation Requirements

Ensure your system meets the following prerequisites:

- **Python Version**: 3.9.0
- **PyTorch Version**: 2.0.1 with CUDA 11.7 support
- **Additional Dependencies**: Refer to the `requirements.txt` file for a complete list of required libraries and their versions.

### Usage Instructions

#### Example Tasks (`./egs`)

This directory houses sample tasks, including:
- **NMI**: Replicates the main experimental results presented in the manuscript.
- **Demo**: Contains code for quick experimentation and familiarization with the tool. 
- **Checkpoint**: Pre-trained Unet model parameters used in the manuscript are located in `./egs/NMI/ckpt/k16`,`./egs/demo/ckpt/k16`.

#### Libraries (`./libs`)

Contains the core libraries of the project:
- Traditional micromagnetic simulation frameworks based on RK4 and LLG equations.
- Unet neural network model architecture.
- Auxiliary functions pertinent to this project.

#### Utilities (`./utils`)

This directory includes scripts for data generation, essential for training the Unet model:
- Scripts generate (spin, Hdemag) pair data required for Unet training.

By following these instructions, users can set up the necessary environment to run simulations, replicate study findings, or train the Unet model with custom data.



## Example Execution

### Running MH Simulations

#### Quick Trial

To expediently initiate the simulation of the MH curve for magnetic thin film materials, such as a material configured into a triangular shape with a dual-layer thickness, characterized by magnetic properties delineated by `{ --Ms 1000, --Ax 0.5e-6, --Ku 0.0 }`, please execute the following script:

```bash
./egs/demo/MH/runMH.sh
```

This script facilitates a comparative analysis of outcomes derived from FFT-based and Unet-based micromagnetic simulation frameworks.


#### Replication of Published Results

To replicate the MH experimental results detailed in the manuscript, please use the following script:

```bash
./egs/NMI/MH_evaluate/runMH.sh
```

This script facilitates the adjustment of the film's dimensions via the `--width` parameter and is configured to test 18 unique combinations of magnetic property parameters:

- `--Ms` for saturation magnetization, with values: {1200, 1000, 800, 600, 400} (in arbitrary units),
- `--Ku` for uniaxial anisotropy constant, with values: {1e5, 2e5, 3e5, 4e5} (in arbitrary units),
- `--Ax` for exchange stiffness constant, with values: {0.7e-6, 0.6e-6, 0.4e-6, 0.3e-6} (in meters).

The `--mask` parameter specifies the shape of the magnetic film, which can include:

- Triangular films,
- Films with a central hole,
- Films of random polygonal shapes.


#### Sample Result Images

Triangular film MH result | Film with a central hole MH result | Random polygonal film MH result
:-------------------------:|:-----------------------------------:|:---------------------------------:
![Triangular film MH result](MH_triangle.png) | ![Film with a central hole MH result](MH_hole.png) | ![Random polygonal film MH result](MH_square.png)


### Running Vortex Simulations

#### Quick Trial

To commence a micromagnetic dynamical analysis based on the LLG (Landau-Lifshitz-Gilbert) equation from a random initial condition, execute:

```bash
./egs/demo/vortex/run.sh
```

This command initiates a simulation applying FFT (Fast Fourier Transform) until the vortex count meets the `--pre_core` parameter's specification, whereupon Unet modeling commences. The `--pre_core` parameter signifies the initial vortex count for transitioning from FFT to Unet simulation.

#### Replication of Published Results

For exact replication of the manuscript's vortex simulation outcomes, utilize:

```bash
./egs/NMI/vortex_evaluate/run.sh
```

This script provides a detailed evaluation of Unet's prediction accuracy across initial vortex counts (`--pre_core=5,10,20”), comparing results from 100 dynamical experiments per test condition to assess Unet's performance.

#### Sample Result Images

Varying Materials | Random Shapes | Square Films
:-----------------------------------------:|:---------------------------------------:|:--------------------------------------:
![Varying Materials](vortex_rdparam.png) | ![Random Shapes](vortex_rdshape.png) | ![Square Films](vortex_square.png)



### Computational Speed Assessment

To assess the computational efficiency of micromagnetic simulations within distinct frameworks, the following command is recommended:

```bash
./egs/NMI/speed_evaluate/run.sh
```

This command facilitates the setup of comparative analyses for micromagnetic films of variable dimensions through the `--width` parameter. It systematically evaluates the computational demands of three distinct simulation frameworks:

- A traditional FFT-based approach,
- The Unet framework, employing deep neural networks,
- TensorRT-accelerated Unet modeling.

This assessment elucidates the potential for efficiency gains and performance enhancements through the adoption of deep learning and acceleration technologies in micromagnetic simulation workflows.


### Data Generation and Model Training Process

#### Data Generation

The dataset referenced in our study is generated via:

```bash
./utils/run.sh
```

The script automatically generates datasets in four sizes: 32, 64, 96, and 128. The first three sizes are for cross-scale training, while the 128 size assesses generalization across scales. Each size's dataset is developed under three conditions: a default square shape, and two scenarios involving random shape masks for augmentation, each additionally subjected to two magnitudes of external magnetic fields. Therefore, each size features three conditions with 100 samples each, facilitating a comprehensive evaluation of the model's robustness and scalability.

For data inspection and analysis, a visualization utility is provided:

```bash
./utils/visualize_data.py
```

Executing this script produces visual representations of the dataset, showcasing magnetic vector fields, RGB imagery, and histograms of numerical statistics, thereby facilitating a comprehensive overview of the training data's characteristics.


Data Visualization Samples |  | 
:-------------------------:|:-------------------------:
![demagnetizing field vector](Hds_vector.png) | ![demagnetizing field RGB](Hd_rgb.png)| ![demagnetizing field histogram](Hd_hist.png)
![Spins RGB](Spins_rgb.png) | ![Spin vector](Spins_vector.png) | ![Spin histogram](Spins_hist.png)


#### Model Training

Once the data is prepared, you can commence training your own Unet model by executing:

```bash
./egs/demo/train/run_train.sh
```

Adjust the volume of training data with `--ntrain` and set the test dataset size via `--ntest`. Training hyperparameters including `--batch-size`, `--lr` (learning rate), and `--epochs` are customizable to optimize performance, though default settings are provided to replicate manuscript results. Parameters `--kc` and `--inch`, crucial for the model's network architecture, remain fixed to preserve model dimensions. Training generates automated logs of model progress, including intermediate models, convergence metrics, and performance evaluations.

To apply the trained model to micromagnetic simulations, replace the existing model at `./egs/NMI/ckpt/k16/model.pt` with the newly trained model.


Model Training Visualizations |  | 
:-----------------------------:|:-----------------------------:
![Training loss example](loss_ex1.0-1.png) | ![Layer 1 RGB output after training](epoch820_L1_rgb.png) | ![Layer 1 Vector output after training](epoch820_L1_vec.png)



### standproblem



## Documentation

- [Paper Citation](LINK_TO_YOUR_DOCUMENTATION)
- [API参考](LINK_TO_YOUR_DOCUMENTATION)


## License

本项目在Apache License (Version 2.0)下分发。

详情请见 [Apache License](LINK_TO_YOUR_LICENSE).


