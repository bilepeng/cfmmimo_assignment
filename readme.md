# Distributed Combinatorial Optimization of Downlink User Assignment in mmWave Cell-free Massive MIMO Using Graph Neural Networks

![GitHub](https://img.shields.io/github/license/bilepeng/risnet_partial_csi)
[![DOI](https://img.shields.io/badge/doi-10.1109/GLOBECOM52923.2024.10901132-informational)](https://doi.org/10.1109/GLOBECOM52923.2024.10901132)
[![arXiv](https://img.shields.io/badge/arXiv-2406.05652-informational)](https://arxiv.org/abs/2406.05652)

This repository contains the source code for the paper "Distributed Combinatorial Optimization of Downlink User Assignment in mmWave Cell-free Massive MIMO Using Graph Neural Networks"
(Bile Peng, Bihan Guo, Karl-Ludwig Besser, Luca Kunz, Ramprasad Raghunath, Anke Schmeink, Eduard A. Jorswieck, Giuseppe Caire, H. Vincent Poor, 2024 IEEE
Global Communications Conference, Dec. 2024.
([doi:10.1109/GLOBECOM52923.2024.10901132](https://doi.org/10.1109/GLOBECOM52923.2024.10901132),
[arXiv:2406.05652](https://arxiv.org/abs/2406.05652))

The data is available under https://drive.google.com/file/d/1H9g5PPtkazrdjGFMBhLszGSJbe513hXg/view?usp=share_link.


## Usage
Put the unzipped files in `data/` and run the following commands to train the model.

```bash
python3 train.py
```

A folder with the name of its creation date and time is automatically created,
where the trained models and the Tensorboard log file are saved.


## Acknowledgements
The work of B. Peng, R. Raghunath, A. Schmeink, E. Jorswieck and G. Caire
is supported by the Federal Ministry of Education and Research Germany
(BMBF) as part of the 6G Research and Innovation Cluster (6G-RIC) under
Grant 16KISK031. The work of K.-L. Besser is supported by the German
Research Foundation (DFG) under grant BE 8098/1-1. The work of L. Kunz
is supported by the BMBF under grant 16KISK074. The work of H. V. Poor
was supported in part by the U.S National Science Foundation under Grants
CNS-2128448 and ECCS-2335876.


## License and Referencing
This program is licensed under the GPLv3 license. If you in any way use this
code for research that results in publications, please cite our original
article listed above.  
You can use the following BibTeX entry

```bibtex
@inproceedings{Peng2024distributed,
  author = {Peng, Bile and Guo, Bihan and Besser, Karl-Ludwig and Kunz, Luca and Raghunath, Ramprasad and Schmeink, Anke and Jorswieck, Eduard A. and Caire, Giuseppe and Poor, H. Vincent},
  title = {Distributed Combinatorial Optimization of Downlink User Assignment in mmWave Cell-free Massive MIMO Using Graph Neural Networks},
  booktitle = {GLOBECOM 2024 -- 2024 IEEE Global Communications Conference},
  year = {2024},
  month = {12},
  pages = {462--468},
  publisher = {IEEE},
  venue = {Cape Town, South Africa},
  archiveprefix = {arXiv},
  eprint = {2406.05652},
  primaryclass = {eess.SP},
  doi = {10.1109/GLOBECOM52923.2024.10901132},
}
```
