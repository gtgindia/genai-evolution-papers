# Denoising Diffusion Probabilistic Models (Ho, Jain, and Abbeel, 2020, arXiv:2006.11239)

Authors: Jonathan Ho, Ajay Jain, and Pieter Abbeel, all at UC Berkeley. The paper was posted to arXiv in June 2020 and appeared at NeurIPS 2020. It is usually shortened to DDPM.

## The story in one paragraph

This paper takes an old idea that everyone had ignored and shows that it can paint beautiful pictures. The idea is diffusion. You slowly add noise to a photo until it becomes pure static, and then you teach a neural network to run that movie backwards, removing a little noise at a time until a new photo appears. The authors make two careful choices that turn this vague idea into a stable recipe. They teach the network to predict the noise that was added rather than the clean image, and they train with a simple unweighted squared-error loss. With those changes, a straightforward U-Net trained on this denoising game reaches an Inception score of 9.46 and an FID of 3.17 on unconditional CIFAR-10, beating most GANs of the day, and produces church, bedroom, and face samples that look similar in quality to ProgressiveGAN.

## What problem was hurting before this paper

By 2020, image generation was dominated by GANs, autoregressive models, flows, and VAEs. GANs made sharp images but were famously unstable and prone to mode collapse, which means they would repeat the same few faces and ignore the full diversity of the data. Autoregressive models had excellent likelihoods but were slow and had a fixed pixel-by-pixel order. Diffusion probabilistic models had been proposed back in 2015 by Sohl-Dickstein and colleagues using ideas from non-equilibrium thermodynamics, but nobody had shown high-quality samples with them. Score-based models and energy-based models were promising, yet their sampling procedures were often hand-tuned after training. The field needed proof that a diffusion model could be simple to train, stable like a regression model, and still competitive on perceptual quality.

## The core idea, explained simply (with an analogy)

Imagine dropping a single drop of ink into a glass of water and filming it as it spreads. After many steps, the water looks uniformly gray and all memory of the drop seems lost. Now imagine learning to play that film in reverse. If you can remove just a tiny bit of cloudiness at each step, you will gradually recover a sharp drop again. Starting from pure gray static, different reverse choices lead to different final drops.

DDPM does the same thing with images. The forward process is fixed: at each of 1000 steps, it adds a tiny amount of Gaussian noise according to a schedule that grows linearly from a very small value to a larger one. After 1000 steps, the image is essentially random noise. The reverse process is learned: a neural network looks at a noisy image and the current step number and predicts how to take one small step back toward a cleaner image. Repeating that step 1000 times turns fresh random noise into a brand-new image that was never in the training set.

## How it actually works (step by step: architecture, training objective, inference — explain each piece in words; explain key equations in plain language)

The model has a forward chain and a reverse chain over the same sequence of variables, from the real image at step zero to pure noise at step T.

The forward chain is fixed and has no learned parameters. At each step, the next image is a slightly shrunken version of the previous image plus fresh Gaussian noise. Because Gaussians compose nicely, you can jump directly to any noise level in closed form. The paper writes this as sampling from a Gaussian whose mean is the original image scaled down and whose variance grows with time. In plain language, if you tell me the original photo and a noise level, I can immediately create the correctly blurred and grainy version without simulating all the intermediate steps. This trick makes training efficient.

The reverse chain is learned. It starts from standard normal noise and applies a sequence of Gaussian steps with learned means. The paper fixes the variances to simple constants and focuses all learning on the means. The key parameterization is to predict the noise epsilon rather than the mean or the clean image. In words, instead of asking the network to paint the clean photo directly, which is a hard and unstable target, the authors ask it what grain was just added. Subtracting a scaled version of that predicted grain gives the correct step toward a cleaner image. This noise-prediction view is mathematically equivalent to denoising score matching across many noise levels, and sampling looks like annealed Langevin dynamics, which is a principled way of walking toward high-probability images by following estimated gradients plus a little randomness.

The exact variational bound would weight different time steps differently, with extra emphasis on nearly clean images. The authors find that this weighting hurts perceptual quality because the network wastes capacity on invisible details. They therefore propose a simplified objective called L-simple. It picks a random image, a random time step, and random noise, forms the noisy image, and penalizes the squared difference between the true noise and the predicted noise. Every time step counts equally. In plain language, the network practices equally at light fog, medium fog, and heavy fog, instead of obsessing over the very lightest fog. Training is just repeated gradient descent on this regression loss.

The architecture is a U-Net similar to PixelCNN++ with Wide ResNet blocks and group normalization, plus self-attention at the 16 by 16 resolution. Time is communicated to every residual block through sinusoidal position embeddings like those in Transformers. The CIFAR-10 model has about 36 million parameters, while the 256 by 256 LSUN and CelebA-HQ models have about 114 million parameters, with a larger 256-million-parameter variant for bedrooms. Images are scaled to the range from negative one to positive one. For likelihood evaluation, the final step uses a discrete decoder that integrates the Gaussian over each pixel bin, so the bound is a true lossless codelength.

Sampling starts from random noise and iterates backwards. At each step, the network predicts the noise, the current image is shifted toward a cleaner version using a precise formula that depends on the noise schedule, and fresh Gaussian noise is added except at the very last step. The paper also shows that you can estimate the final clean image at any intermediate time by removing all the predicted noise at once. Early steps reveal large shapes like pose and layout, while late steps fill in hair strands and textures. The same intermediate latent can branch into several images that share high-level attributes but differ in fine details.

## Key results and numbers from the paper (with the actual scores/tables described in words)

On unconditional CIFAR-10, the simplified-objective model reaches an Inception score of 9.46 and an FID of 3.17 when FID is computed against the training set, as was standard. Against the test set, FID is 5.24, which is still better than many published training-set scores. This beats BigGAN at 14.73 FID, most energy-based and score-based models in the 25 to 38 range, and PixelCNN variants near 50 to 65. It is close to StyleGAN2 with adaptive augmentation at 3.26 FID, which is remarkable for a model with no adversarial training.

The ablation table is the heart of the paper. Predicting the posterior mean with the true variational bound and fixed variances gives an Inception score around 8.06 and FID around 13.22. Predicting noise with the true bound gives similar numbers around 7.67 and 13.51. But predicting noise with the simplified loss jumps to 9.46 and 3.17. Learning the variances instead of fixing them makes training unstable and hurts quality. Predicting the clean image directly worked worse early on. In short, neither noise prediction alone nor the simple loss alone is enough. Together they improve FID by roughly a factor of four.

On LSUN 256 by 256, the small model gets FID 6.36 on bedrooms, 7.89 on churches, and 19.75 on cats. The larger bedroom model improves to 4.90, which the authors describe as similar to ProgressiveGAN. Likelihoods are decent but not state of the art. The best-sample model uses about 1.78 bits per dimension for rate and 1.97 bits per dimension for distortion, with a root mean squared error under one gray level on a zero to 255 scale. More than half the lossless codelength describes changes humans can barely see. Rate-distortion curves show that distortion falls steeply with the first few bits, which supports the paper's claim that diffusion models are excellent lossy compressors even when they are not the best lossless compressors.

## Why this paper matters (what later work it unlocked — name the specific later papers among the 15 where relevant)

This is the paper that brought GANs down from the throne and made diffusion the default language of image generation. Within this 15-paper collection, its most direct child is paper 07, Latent Diffusion, often called Stable Diffusion. That work runs exactly this DDPM denoising process, but in a compressed latent space instead of pixels, which makes high-resolution generation fast and cheap. Paper 06, CLIP, supplies the text encoder that later lets diffusion models follow prompts, so the combination of CLIP plus latent diffusion is what powers modern text-to-image systems. The broader family that followed, including DDIM for fast sampling, classifier guidance, Imagen, DALL-E 2, and video models like Sora, all start from the noise-prediction plus L-simple recipe proven here. Conceptually, the paper also bridged variational inference, score matching, autoregressive decoding, and lossy compression in one framework, which gave researchers many entry points to improve it.

## Honest limitations

Sampling is slow. Generating one batch requires 1000 sequential network evaluations, so a batch of 256 CIFAR images already takes many seconds, and 256 by 256 sampling takes minutes. Log-likelihoods lag behind autoregressive models and flows, so if your goal is pure compression or density estimation, diffusion is not the best choice. The paper is also honest that its progressive compression scheme is a proof of concept. It relies on an idealized coding procedure called minimal random coding that is not tractable for high-dimensional images, so you cannot yet ship it as a real codec. Finally, all the headline numbers use fixed variances and a hand-chosen linear noise schedule. Later work shows that schedules, variances, and weighting can be improved substantially.

## Fun trivia and history (from your web search)

The most loved story about this paper is that it revived a five-year-old idea that everyone else had abandoned. Sohl-Dickstein's 2015 paper had already written down the full forward and reverse mathematics using thermodynamics language, but its samples never threatened GANs, so the thread went quiet. Jonathan Ho, then a PhD student in Pieter Abbeel's lab, reportedly picked it up as a side project alongside flow models, and found that a tiny change in what the network is asked to predict completely changes the outcome. The project code was only about 800 lines, yet within a month the community had multiple PyTorch reproductions.

A second striking detail is how brutally simple the winning recipe is. There is no discriminator, no annealing schedule tuned by hand at sampling time, and no complex regularization. It is mean squared error plus exponential moving average of weights plus a big U-Net. The paper reports that sampling from the moving-averaged weights alone improves FID by well over a point. That simplicity is a large part of why diffusion spread so fast. Once people saw that stable regression plus 1000 small steps could beat BigGAN without adversarial tricks, the whole field pivoted, and within two years the visual internet was filled with descendants like Stable Diffusion, Midjourney, and DALL-E 2.

## Key terms explained (glossary of 5-8 terms)

Forward process means the fixed chain that adds noise step by step until the image becomes static. It has no learned parameters and is fully described by the noise schedule.

Reverse process means the learned chain that removes noise step by step, starting from static and ending at a new image. Each step is a Gaussian whose mean comes from the neural network.

Noise schedule, written as beta values, controls how much noise is added at each step. The paper uses 1000 steps increasing linearly from a tiny value to 0.02. Larger values destroy signal faster.

Epsilon prediction means training the network to output the noise that was added, rather than the clean image. Subtracting the predicted noise is numerically calmer and connects to score matching.

L-simple is the simplified training loss. It is just the average squared error between true noise and predicted noise, with time steps sampled uniformly. It down-weights nearly clean steps that the exact bound would emphasize.

FID and Inception score are sample-quality metrics computed with an Inception classifier. Inception score rewards sharp and diverse predictions. FID measures the distance between real and generated feature distributions, so lower is better.

U-Net is a convolutional architecture with a downsampling path, an upsampling path, and skip connections. It lets the model reason at multiple scales, from coarse layout to fine texture.

Annealed Langevin dynamics is a sampling method that follows estimated gradients of the data density while gradually reducing noise. DDPM sampling resembles this, which is why the paper talks about scores and Langevin dynamics.

## Connections (which earlier paper it builds on, which later paper builds on it)

This paper builds on the 2015 Sohl-Dickstein diffusion paper for the Markov formulation and variational bound, on U-Net and ResNet ideas for the backbone, on Transformer sinusoidal embeddings from paper 01 for time conditioning, and on score-matching work by Song and Ermon for the noise-level interpretation. It is contemporary with GAN and PixelCNN work that it compares against.

It is built upon directly by paper 07, Latent Diffusion, which moves the same denoising loop into latent space to enable Stable Diffusion. It pairs naturally with paper 06, CLIP, whose text embeddings become the conditioning signal for text-to-image diffusion. Later advances like DDIM, improved DDPM with learned variances and cosine schedules, classifier guidance that finally beats GANs on ImageNet, and modern video and protein models such as Sora and AlphaFold 3 all trace their sampling loop back to Algorithms 1 and 2 of this paper.
