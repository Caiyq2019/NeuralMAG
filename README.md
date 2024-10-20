<div align="center">
  <img width="500px" src="./figs/logol.png"/>
</div>


## Introduction to NeuralMAG Project

The NeuralMAG Project is an open-source neural network framework designed for micromagnetic simulation. It employs a multi-scale U-Net neural network model to learn the physical relationship between magnetization states and demagnetizing fields, effectively extending traditional micromagnetic simulations into the realm of advanced deep learning frameworks. This approach fully leverages the cutting-edge technologies available on deep learning platforms.

<div align="center">
<figure>
    <img src="figs/framework.png" width="500" height="300">
</figure>
</div>


The project finds wide application in the study of magnetic materials among other areas.

## Capabilities of NeuralMAG

- **Integration with PyTorch Platform**: NeuralMAG integrates MAG micromagnetic simulations into the PyTorch framework, utilizing the extensive parallelization capabilities of GPUs for enhanced performance.
- **Cross-Scale Neural Network Simulation**: By employing a multi-scale Unet neural network architecture, the system can efficiently simulate magnetic material systems on a large scale.
- **Generalization Across Material Systems**: The model has been designed to generalize effectively to previously unseen material systems, thereby broadening the scope for exploration of new materials.
- **Versatility in Micromagnetic Simulation Tasks**: NeuralMAG is adept at performing a wide array of micromagnetic simulations, including but not limited to predicting magnetic ground states, simulating hysteresis loops, and accommodating arbitrary geometric shapes in simulations.
- **Utilization of Advanced Deep Learning Optimization Techniques**: The framework incorporates the latest advancements in model compression and acceleration optimization technologies that are prevalent in modern deep learning platforms.



## Getting Started

### Installation Requirements

Hardware requirements
- **GPU**: Nvidia GTX 1070 or above (Nvidia RTX-3090 or above for TensorRT)

Ensure your system meets the following prerequisites:
- **Linux**: Ubuntu 20.04.6, Ubuntu 22.04 or above
- **Python Version**: 3.9.0 or above
- **PyTorch Version**: 2.0.1 with CUDA 11.7 support or above
- **Additional Dependencies**: For a complete list of required libraries and their versions, refer to the [`requirements.txt`](./requirements.txt) file.

### Usage Instructions

#### Example Tasks (`./egs`)

This directory houses sample tasks, including:
- **NMI**: Replicates the main experimental results presented in the manuscript.
- **Demo**: Contains code for quick experimentation and familiarization with the tool. 
- **Checkpoint**: Pre-trained Unet model parameters used in the manuscript are located in [`Checkpoint`](./egs/NMI/ckpt/k16)

#### Libraries (`./libs`)

Contains the core libraries of the project:
- Traditional micromagnetic simulation frameworks based on RK4 and LLG equations[`MAG2305`](./libs/MAG2305.py).
- Unet neural network model architecture[`Unet`](./libs/Unet.py).
- Auxiliary functions pertinent to this project[`misc`](./libs/misc.py).

#### Utilities (`./utils`)

This directory includes scripts for data generation, essential for training the Unet model:
- Scripts generate the Magnetic spin & Demagnetizing field data pairs \( \vec{m}, H_{\text{demag}} \) required for Unet training.

By following these instructions, users can set up the necessary environment to run simulations, replicate study findings, or train the Unet model with custom data.


## Example Execution [chrome browser preferred]

[MH demo quick trial in Colab](https://colab.research.google.com/drive/1ppVSR1Wwan5zVr_lg4UfkjMbTWjlCMDw?usp=sharing)

[Vortex demo quick trial in Colab](https://colab.research.google.com/drive/18YaOxSH2XWY4StrxsyacGeYwIxU-c-_E?usp=sharing)

### Running MH Simulations

#### Quick Trial

To expediently initiate the simulation of the MH curve for magnetic thin film materials, such as a material configured into a triangular shape with a dual-layer thickness, characterized by magnetic properties delineated by `{ --Ms 1000, --Ax 0.5e-6, --Ku 0.0 }`, please execute the following script:

```bash
cd ./egs/demo/MH
sh runMH.sh
```

This script facilitates a comparative analysis of outcomes derived from FFT-based and Unet-based micromagnetic simulation frameworks.


#### Replication of Published Results

To replicate the MH experimental results detailed in the manuscript, please use the following script:

```bash
cd ./egs/NMI/MH_evaluate
sh runMH.sh
```

This script facilitates the adjustment of the film's dimensions via the `--width` parameter and is configured to test 13 unique combinations of magnetic property parameters:

- `--Ms` for saturation magnetization, with values: {1200, 1000, 800, 600, 400} (in arbitrary units),
- `--Ku` for uniaxial anisotropy constant, with values: {1e5, 2e5, 3e5, 4e5} (in arbitrary units),
- `--Ax` for exchange stiffness constant, with values: {0.7e-6, 0.6e-6, 0.4e-6, 0.3e-6} (in meters).

The `--mask` parameter specifies the shape of the magnetic film, which can include:

- Triangular films,
- Films with a central hole,
- Films of random polygonal shapes.


#### Sample: MH result Images

Triangular film MH result | Film with a central hole MH result | Random polygonal film MH result
:-------------------------:|:-----------------------------------:|:---------------------------------:
![Triangular film](./figs/MH_triangle.png) | ![Film with a central hole](./figs/MH_hole.png) | ![Random polygonal film](./figs/MH_square.png)


### Running Vortex Simulations

#### Quick Trial

To commence a micromagnetic dynamical analysis based on the LLG (Landau-Lifshitz-Gilbert) equation from a random initial condition, execute:

```bash
cd ./egs/demo/vortex
sh run.sh
```

This command initiates a simulation applying FFT (Fast Fourier Transform) until the vortex count meets the `--InitCore` parameter's specification, whereupon Unet modeling commences. The `--InitCore` parameter signifies the initial vortex count for transitioning from FFT to Unet simulation.

#### Replication of Published Results

For exact replication of the manuscript's vortex simulation outcomes, utilize:

```bash
cd ./egs/NMI/vortex_evaluate
sh run.sh
```

This script provides a detailed evaluation of Unet's prediction accuracy across initial vortex counts `--InitCore=5,10,20`, comparing results from 100 dynamical experiments per test condition to assess Unet's performance.

#### Sample: Vortex Simulations Result Images

Varying Materials | Random Shapes | Square Films
:-----------------------------------------:|:---------------------------------------:|:--------------------------------------:
![Varying Materials](./figs/vortex_rdparam.png) | ![Random Shapes](./figs/vortex_rdshape.png) | ![Square Films](./figs/vortex_square.png)



### Computational Speed Assessment

To assess the computational efficiency of micromagnetic simulations within distinct frameworks, the following command is recommended:

```bash
cd ./egs/NMI/speed_evaluate
sh run.sh
```

This command facilitates the setup of comparative analyses for micromagnetic films of variable dimensions through the `--width` parameter. It systematically evaluates the computational demands of three distinct simulation frameworks:

- A traditional FFT-based approach,
- The Unet framework, employing deep neural networks,
- TensorRT-accelerated Unet modeling.

This assessment elucidates the potential for efficiency gains and performance enhancements through the adoption of deep learning and acceleration technologies in micromagnetic simulation workflows.

### Standard Problems

The [standard problems](https://www.ctcms.nist.gov/~rdm/mumag.org.html) are proposed by micromagnetic modeling activity group ($\mu$MAG in NIST, USA), allowing micromagnetic researchers to "compare techniques, identify problems, and detect programming bugs".

In current version of NeuralMAG, the problems \#1 and \#4 designed for magnetic thin films could be verified. 
The simulation demo for the standard problem \#1 could be launched via following commands:

```bash
# Standard problem #1
cd ./egs/NMI/standard_problem1
sh run.sh
```
and for standard problem \#4:
```bash
# Standard problem #4
cd ./egs/NMI/standard_problem4
sh run.sh
```


### Data Generation and Model Training Process

#### Data Generation

The dataset referenced in our study is generated via:

```bash
cd ./utils
sh run.sh 
```

The script automatically generates datasets in four sizes: 32, 64, 96, and 128. The first three sizes are for cross-scale training, while the 128 size assesses generalization across scales. Each size's dataset is developed under three conditions: a default square shape, and two scenarios involving random shape masks for augmentation, each additionally subjected to two magnitudes of external magnetic fields. Therefore, each size features three conditions with 100 cases each, facilitating a comprehensive evaluation of the model's robustness and scalability.

For data inspection and analysis, a visualization utility is provided:

```bash
cd 
./utils
python visualize_data.py
```

Executing this script produces visual representations of the dataset, showcasing magnetic vector fields, RGB imagery, and histograms of numerical statistics, thereby facilitating a comprehensive overview of the training data's characteristics.

#### Sample: Data Visualization 
Spin vector | Spins RGB | Spin histogram 
:-----------------------------------------:|:---------------------------------------:|:--------------------------------------:
![Spins RGB](./figs/Spins_vector.png) | ![Spin vector](./figs/Spins_rgb.png) | ![Spin histogram](./figs/Spins_hist.png)

demagnetizing field vector | demagnetizing field RGB | demagnetizing field histogram 
:-----------------------------------------:|:---------------------------------------:|:--------------------------------------:
![demagnetizing field vector](./figs/Hds_vector.png) | ![demagnetizing field RGB](./figs/Hds_rgb.png)| ![demagnetizing field histogram](./figs/Hd_hist.png)

#### Model Training

Once the data is prepared, you can commence training your own Unet model by executing:

```bash
cd ./egs/demo/train
sh run_train.sh
```

Adjust the volume of training data with `--ntrain` and set the test dataset size via `--ntest`. Training hyperparameters including `--batch-size`, `--lr` (learning rate), and `--epochs` are customizable to optimize performance, though default settings are provided to replicate manuscript results. Parameters `--kc` and `--inch`, crucial for the model's network architecture, remain fixed to preserve model dimensions. Training generates automated logs of model progress, including intermediate models, convergence metrics, and performance evaluations.

To apply the trained model to micromagnetic simulations, replace the existing model at `./egs/NMI/ckpt/k16/model.pt` with the newly trained model.

#### Sample: Model Training Visualizations
 Training loss | Input(spin) and output(Hd) data RGB | Input(spin) and output(Hd) data vector
:-----------------------------:|:-----------------------------:|:-----------------------------:
![Training loss example](./figs/loss.png) | ![Layer 1 RGB output after training](./figs/train_L1_rgb.png) | ![Layer 1 Vector output after training](./figs/train_L1_vec.png)




## Citation

```

@misc{NeuralMAG2024,
  author = {Caiyq and Ijin08},
  title = {NeuralMAG: Initial Release v1.0.0},
  year = {2024},
  publisher = {Zenodo},
  version = {1.0.0},
  doi = {10.5281/zenodo.13736224},
  url = {https://doi.org/10.5281/zenodo.13736224}
}

```


## License

This project is licensed under the MIT License - see the [`LICENSE`](./LICENSE) file for details.


