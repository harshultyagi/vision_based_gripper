# References

Bibliographic details below were checked against the publishers' or arXiv listings
(October 2026). Numbers quoted from these works were obtained under *their* protocols, not ours.

1. **[Jiang 2011]** Y. Jiang, S. Moseson, A. Saxena. *Efficient grasping from RGBD images: Learning
   using a new rectangle representation.* IEEE International Conference on Robotics and Automation
   (ICRA), Shanghai, 2011. DOI 10.1109/ICRA.2011.5980145. — Introduces the grasp
   rectangle and the Cornell Grasping Dataset.
2. **[Lenz 2015]** I. Lenz, H. Lee, A. Saxena. *Deep learning for detecting robotic grasps.* The
   International Journal of Robotics Research, 2015. DOI 10.1177/0278364914549607. arXiv:1301.3592.
3. **[Redmon 2015]** J. Redmon, A. Angelova. *Real-time grasp detection using convolutional neural
   networks.* ICRA 2015. arXiv:1412.3128.
4. **[Morrison 2018]** D. Morrison, P. Corke, J. Leitner. *Closing the Loop for Robotic Grasping: A
   Real-time, Generative Grasp Synthesis Approach.* Robotics: Science and Systems XIV (RSS), 2018.
   DOI 10.15607/RSS.2018.XIV.021. arXiv:1804.05172. Code: https://github.com/dougsm/ggcnn
5. **[Kumra 2020]** S. Kumra, S. Joshi, F. Sahin. *Antipodal Robotic Grasping using Generative
   Residual Convolutional Neural Network.* IEEE/RSJ International Conference on Intelligent Robots and
   Systems (IROS), 2020. DOI 10.1109/IROS45743.2020.9340777. arXiv:1909.04810.
   Code: https://github.com/skumra/robotic-grasping
6. **[He 2016]** K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.*
   CVPR 2016. arXiv:1512.03385.
7. **[Ioffe 2015]** S. Ioffe, C. Szegedy. *Batch Normalization: Accelerating Deep Network Training
   by Reducing Internal Covariate Shift.* ICML 2015. arXiv:1502.03167.
8. **[Kingma 2015]** D. P. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization.* ICLR 2015.
   arXiv:1412.6980.
9. **[Loshchilov 2017]** I. Loshchilov, F. Hutter. *SGDR: Stochastic Gradient Descent with Warm
   Restarts.* ICLR 2017. arXiv:1608.03983. (cosine learning-rate schedule)
10. **[Girshick 2015]** R. Girshick. *Fast R-CNN.* ICCV 2015. arXiv:1504.08083. (Smooth L1 loss)
11. **[Srivastava 2014]** N. Srivastava et al. *Dropout: A Simple Way to Prevent Neural Networks from
    Overfitting.* JMLR 15:1929–1958, 2014.
12. **[Bertalmio 2001]** M. Bertalmio, A. L. Bertozzi, G. Sapiro. *Navier–Stokes, fluid dynamics, and
    image and video inpainting.* CVPR 2001. (OpenCV `INPAINT_NS`)
13. **[Micikevicius 2018]** P. Micikevicius et al. *Mixed Precision Training.* ICLR 2018.
    arXiv:1710.03740.
14. PyTorch documentation, https://pytorch.org/docs/stable/ ; installation selector
    https://pytorch.org/get-started/locally/
15. Gradio documentation, https://www.gradio.app/docs
16. Shapely documentation (polygon intersection), https://shapely.readthedocs.io/

### Published Cornell results quoted in the report

Taken from Table III of [Kumra 2020] (rectangle metric, IoU > 25 %, angle < 30°;
cross-validation as described there; split ratios are not given). Not reproduced here.

| Method | Image-wise | Object-wise |
|---|---|---|
| Jiang et al. 2011 (fast search) | 60.5 % | 58.3 % |
| Lenz et al. 2015 | 73.9 % | 75.6 % |
| Redmon & Angelova 2015 | 88.0 % | 87.1 % |
| GG-CNN (Morrison et al. 2018) | 73.0 % | 69.0 % |
| GR-ConvNet-D | 93.2 % | 94.3 % |
| GR-ConvNet-RGB | 96.6 % | 95.5 % |
| GR-ConvNet-RGB-D | 97.7 % | 96.6 % |
