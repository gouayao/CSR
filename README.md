#CSR

This code implements CSR model, described in the paper Rethinking Cross-domain Semantic Relation for Few-shot Image Generation, Yao Gou, Min Li, Yilong Lv, Yusen Zhang, Yuhang Xing, APIN, 2023.

##The code borrows heavily from the PyTorch implementation of CDC:
https://github.com/utkarshojha/few-shot-gan-adaptation

##Our CSR is inspired by ContraD
https://github.com/jh-jeong/ContraD

##Train the CSR model: 
```bash
python train.py  --ckpt ./checkpoints/source_ffhq.pt --data_path ./processed_data/10-shot/ --exp 10-shot_CSR
```

## Acknowledgment

As mentioned before, the StyleGAN2 model is borrowed from this wonderful [pytorch implementation](https://github.com/rosinality/stylegan2-pytorch) by [@rosinality](https://github.com/rosinality). We are also thankful to [@mseitzer](https://github.com/mseitzer) and [@richzhang](https://github.com/richzhang) for their user friendly implementations of computing [FID score](https://github.com/mseitzer/pytorch-fid) and [LPIPS metric](https://github.com/richzhang/PerceptualSimilarity) respectively. 
