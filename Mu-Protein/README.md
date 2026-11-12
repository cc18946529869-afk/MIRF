 <!-- ![Repo Logo](docs/logo.png) -->

 <p align="center">
  <img src="docs/logo-small.png" alt="Repo Logo" width="500">
</p>

# Introduction 
μProtein is a general framework designed to accelerate  protein engineering by integrating μFormer, a deep learning model for accurate mutational effect prediction, with μSearch, a reinforcement learning algorithm tailored for efficient navigation of the protein fitness landscape.

For more details, please refer to our [paper in Nature Machine Intelligence](https://www.nature.com/articles/s42256-025-01103-w).

This repository contains the following components:

- **`pmlm/`** – Protein language model pretraining  
- **`mu-former/`** – Fitness landscape modeling using the pretrained protein language model  
- **`mu-search/`** – Navigating the constructed fitness landscape oracle
- **`pretrained/`** – Pretrained PMLM model checkpoint (stored using Git LFS).

For more details, refer to the respective README files:  

- [μFormer](mu-former/README.md)  
- [μSearch](mu-search/README.md)  
- [PMLM Pretraining](pmlm/README.md) 

## Citation
If you are using our code or model, please cite the following paper:

```bibtex
@article{sun2025accelerating,
  title={Accelerating protein engineering with fitness landscape modelling and reinforcement learning},
  author={Sun, Haoran and He, Liang and Deng, Pan and Liu, Guoqing and Zhao, Zhiyu and Jiang, Yuliang and Cao, Chuan and Ju, Fusong and Wu, Lijun and Liu, Haiguang and others},
  journal={Nature Machine Intelligence},
  pages={1--15},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

## License

This repository is licensed under the [MIT License](LICENSE).  

---

## Contact

For questions or collaborations, please contact the authors via email or open an issue in this repository.

---