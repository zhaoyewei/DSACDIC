# Deep Subdomain Alignment for Cross-domain Image Classification
[paper](https://openaccess.thecvf.com/content/WACV2024/papers/Zhao_Deep_Subdomain_Alignment_for_Cross-Domain_Image_Classification_WACV_2024_paper.pdf) [video](https://www.youtube.com/watch?v=AM63h2LzFY8)

# Abstract

Unsupervised domain adaptation (UDA), which aims to transfer knowledge learned from a labeled source domain to an unlabeled target domain, is useful for various cross-domain image classification scenarios. A commonly used approach for UDA is to minimize the distribution differences between two domains, and subdomain alignment is found to be an effective method. However, most of the existing subdomain alignment methods are based on adversarial learning and focus on subdomain alignment procedures without considering the discriminability among individual subdomains, resulting in slow convergence and unsatisfactory adaptation results. To address these issues, we propose a novel deep subdomain alignment method for UDA in image classification, which consists of a Union Subdomain Contrastive Learning (USCL) module and a Multi-view Subdomain Alignment (MvSA) strategy. USCL can create discriminative and dispersed subdomains by bringing samples from the same subdomain closer while pushing away samples from different subdomains. MvSA makes use of labeled source domain data and easy target domain data to perform target-to-source and target-to-target alignment. Experimental results on three image classification datasets (Office-31, Office-Home, Visda-17) demonstrate that our proposed method is effective for UDA and achieves promising results in several cross-domain image classification tasks.

# Dependence
`
Torch == 1.4.0
Torchvision == 0.5.0
scipy == 1.2.3
numpy == 1.16.6
`

# Dataset

[Office-Home](https://arxiv.org/pdf/1706.07522v1)

[Office-31](https://faculty.cc.gatech.edu/~judy/domainadapt/)

[VisDA-17](https://ai.bu.edu/visda-2017/)

# Usage

`python main.py`

# Contract

mail to: zhaoyewei21@mails.ucas.ac.cn

# Citation
`
InProceedings{Zhao_2024_WACV,

    author    = {Zhao, Yewei and Han, Hu and Shan, Shiguang and Chen, Xilin},
    
    title     = {Deep Subdomain Alignment for Cross-Domain Image Classification},
    
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    
    month     = {January},
    
    year      = {2024},
    
    pages     = {2820-2829}
}
`
