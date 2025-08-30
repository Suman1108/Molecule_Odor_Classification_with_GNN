# Molecular Odor Prediction with Graph Neural Networks

## Overview

Quantitative Structure-Odor Relationship (QSOR) modeling seeks to predict how a molecule smells based on its structure. With advancements in deep learning and graph neural networks (GNNs) that can learn directly from molecular graphs, this long standing interdisciplinary challenge spanning chemistry, neuroscience, and machine learning is gaining attention.

Human odor perception is driven by interactions between odorant molecules and hundreds of olfactory receptors in the nose. By leveraging machine learning to model this process, we can design novel synthetic fragrances, reduce dependency on natural raw materials, and improve our understanding of how the brain processes smells.

While several recent studies have explored machine learning approaches to odor prediction, there remains significant room for improvement in generalizability, interpretability, and chemical insight. Among the foundational works in this space, *Machine Learning for Scent* by Sánchez-Lengeling et al. and the *Principal Odor Map (POM)* proposed by Lee et al. have been especially influential. These studies offered strong baselines, modeling strategies, and odor representation frameworks that guided and informed this project. The open-source [POM GitHub repository](https://github.com/ARY2260/openpom) also proved useful for understanding practical implementation details.

This project builds upon these efforts by extending feature representations, incorporating functional group information, and modeling odor descriptor hierarchies. For a detailed description of methodology, experiments, and results, please refer to the research paper available in folder - Paper and Slides.

---

### Setup
```sh
# Clone the repository
git clone https://github.com/Suman1108/Molecule_Odor_Classification_with_GNN.git
cd Molecule_Odor_Classification_with_GNN

# Create a virtual environment
python -m venv myenv
source myenv/bin/activate  # On Windows use: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
``` 

## Usage
```sh
# Train and Evaluate the model
python TrainEvaluate.py
```
## References

1. Gilmer et al. (2017). *Neural Message Passing for Quantum Chemistry*.  
2. Wu et al. (2018). *MoleculeNet: A Benchmark for Molecular Machine Learning*.  
3. Duvenaud et al. (2015). *Convolutional Networks on Graphs for Learning Molecular Fingerprints*.  
4. Kipf & Welling (2016). *Semi-Supervised Classification with Graph Convolutional Networks*.  
5. Amoore & Pfaffmann (1969). *A Plan to Identify Odor Qualities of Molecules*.  
6. Buck & Axel (1991). *A Novel Multigene Family May Encode Odorant Receptors*.  
7. Mainland et al. (2014). *The Missense of Smell: Functional Variability in the Human Odorant Receptor Repertoire*.  
8. Sánchez-Lengeling et al. (2023). *Machine Learning for Scent: Learning Generalizable Perceptual Representations of Small Molecules*.

9. Lee, B. K., Mayhew, E. J., Sánchez-Lengeling, B., Wei, J. N., Qian, W. W., Little, K. A.,& Wiltschko, A. B. (2022). *Principal Odor Map unifies diverse tasks in human olfactory perception*. Nature, 601(7891), 177–182. https://doi.org/10.1038/s41586-021-04276-0

## License
This project is licensed under the Apache-2.0 License.

