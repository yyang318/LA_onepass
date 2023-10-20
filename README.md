This is the code of manuscript "A Light-weight, Effective and Efficient Model for Label Aggregation in Crowdsourcing" submitted to ACM Transactions on Knowledge Discovery from Data.

------

**Implemented methods**: we use pure python to implement MV, LA<sup>onepass</sup>,  LA<sup>twopass</sup>, MV, DS, LFC, ZC, EBCC, BWA, iCRH, TILCC and SBIC.

BiLA, LAA and TiReMGE require Tensorflow to run, we use their authors' implementation and they can be downloaded from:

- BiLA: https://github.com/ChiHong-Xtautau/BiLA-OnlineLabelAggregation
- LAA: https://github.com/coverdark/deep_laa
- TiReMGE: https://github.com/lazyloafer/TiReMGE

------

**Usage**: the scripts of the experiments are in the *experiments* package.

**Requirements**: the algorithms only require standard Python libraries to run. Numpy, pandas and matplotlib are used to summarize the results and plot.

------

**Dataset**: 20 datasets used in the experiments are in the *dataset* folder. Their links and sources are given below:

- *senti* and *fact* are collected from  CrowdScale 2013 (Josephy et al. 2014). They can be downloaded at https://sites.google.com/site/crowdscale2013/home.
- *MS*, *ZC\_in*, *ZC\_us*, *ZC\_all*, *SP*, *SP\_amt*, *CF* and *CF\_amt* are from Active Crowd Toolkit project (Venanzi et al. 2015). They can be downloaded at https://github.com/orchidproject/active-crowd-toolkit.
- *prod*, *tweet*, *dog*, *face* and *adult* are collected from Truth Inference Project (Zheng et al. 2017). They can be downloaded at http://dbgroup.cs.tsinghua.edu.cn/ligl/crowddata/.
- *bird*, *rte*, *web* and *trec* are used in (Zhang et al. 2014). They can be downloaded at https://github.com/zhangyuc/SpectralMethodsMeetEM.
- *mill* is collected from a "Who wants to be a millionaire app" (Aydin, Yilmaz, and Demirbas 2021). It can be downloaded from https://github.com/bahadiri/Millionaire.

References:

Aydin, B. I.; Yilmaz, Y. S.; and Demirbas, M. 2021. A crowdsourced “Who wants to be a millionaire?” player. Concurrency and Computation: Practice and Experience, 33(8): e4168.

Josephy, T.; Lease, M.; Paritosh, P.; Krause, M.; Georgescu, M.; Tjalve, M.; and Braga, D. 2014. Workshops held at the first aaai conference on human computation and crowdsourcing: A report. AI Magazine, 35(2): 75–78.

Venanzi, M.; Parson, O.; Rogers, A.; and Jennings, N. 2015. The activecrowdtoolkit: An open-source tool for benchmarking active learning algorithms for crowdsourcing research. In Third AAAI Conference on Human Computation and Crowdsourcing.

Zhang, Y.; Chen, X.; Zhou, D.; and Jordan, M. I. 2014. Spectral methods meet EM: A provably optimal algorithm for crowdsourcing. Advances in neural information processing systems, 27.

Zheng, Y.; Li, G.; Li, Y.; Shan, C.; and Cheng, R. 2017. Truth inference in crowdsourcing: Is the problem solved? Proceedings of the VLDB Endowment, 10(5): 541–552.

------

**Note**: the algorithmic flows of the scripts *proposed_method.py* and *proposed_method_online.py* are the same.  In  *proposed_method_online.py*, we add extra code for online evaluation.
