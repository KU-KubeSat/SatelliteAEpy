# Research image log, model progressions

Motivation: original team used different zip shenanigans to get to 3-3.1x compression, thought ML could get to 10x or even 100x

Early october 2024:

Started designing the first model, a basic VAE, a variational autoencoder, standard model for image compression:

## Stable 10x research (Fall 24-25)

MNIST test: <br>
<img width="680" height="369" alt="Pasted image 20250203232738" src="https://github.com/user-attachments/assets/83def07c-2034-468d-bcf9-63417c1261de" />

Quickly realized VAEs can retain semantic information, but do not perform well at all for reconstrution, 2 million loss at certain points, but usually in the 100-200 range
<img width="680" height="439" alt="Pasted image 20250203232804" src="https://github.com/user-attachments/assets/7ef4d685-0697-47b7-963d-0895bccd7f6b" />

switched to an autoencoder, the codebook/semantic information is jumbled, basically unuseable for other models, but the restoration itself is much better, noisy restoration:
<img width="680" height="341" alt="Pasted image 20250203233039" src="https://github.com/user-attachments/assets/f92fc5c4-9d87-48c3-8054-d737ff408bfa" />

Used the structural similarity index (SSIM), better visual alignment:
<img width="680" height="341" alt="Pasted image 20250203233127" src="https://github.com/user-attachments/assets/cb2ef8a0-fb8a-4c8c-8865-232f2d3991e4" />

Had a 1 color channel issue for a while where the model only learned green:
<img width="680" height="680" alt="Pasted image 20250203233145" src="https://github.com/user-attachments/assets/81acb913-3c29-43b2-b42e-61ebc392ef74" />

Used 3 color channels, solved the issue, first 10x compression, which is currently the main upload:
<img width="900" height="452" alt="Pasted image 20250203233213" src="https://github.com/user-attachments/assets/f029ab7b-47c6-4df4-aec1-701c5b25a880" />

Attempted classic CV tests with the old model: <br>
<img width="680" height="373" alt="Pasted image 20250203234642" src="https://github.com/user-attachments/assets/4b3f9e50-b6d4-47f8-a278-d1d0c7b64400" />
<img width="679" height="323" alt="Pasted image 20250203234701" src="https://github.com/user-attachments/assets/4d23c7d1-3596-4fd8-8fac-612d97c5299a" />

## 100x compression testing (spring 24-25)

Residuals and Skip connects need work, combine multiple stride convolutions:
<img width="680" height="680" alt="Pasted image 20250203233420" src="https://github.com/user-attachments/assets/4a102353-82d2-4420-857d-bfa0e78a46f0" />

Started working on 100x compression from here, SSIM somewhat works, initial patchwork:
<img width="680" height="173" alt="Pasted image 20250203233716" src="https://github.com/user-attachments/assets/415af748-b089-4716-b9c7-7bae2e089f71" />

25x compression:
<img width="2047" height="514" alt="Pasted image 20250203234348" src="https://github.com/user-attachments/assets/dcce51b9-2e65-41fb-8dc5-6aac1492bd0c" />

60x compression is a blur:
<img width="680" height="373" alt="Pasted image 20250203234613" src="https://github.com/user-attachments/assets/989991cb-6a01-4042-894e-89ec5b607122" />

pure SSIM, 122x comrpession:
<img width="1678" height="939" alt="Pasted image 20250203234752" src="https://github.com/user-attachments/assets/71b26a86-c788-46f4-917b-574fbd1c8994" />

The main idea behind brownian bridge testing, current version isnt using this but the math looks promising, try it again sometime:
<img width="1959" height="514" alt="Pasted image 20250203235001" src="https://github.com/user-attachments/assets/70fd2541-a57d-4713-9fa8-c352a50ff4f2" />

The model learns position and structural info before color:
<img width="794" height="394" alt="visualization_epoch_46" src="https://github.com/user-attachments/assets/285c4ad6-cb46-433c-b3be-f74da9be0ed6" />

MSE causes the square artifacts, SSIM paves them a little, but conversely ruins complicated details
<img width="1982" height="1012" alt="Pasted image 20250205130653" src="https://github.com/user-attachments/assets/88e595b1-e6fe-4c0d-839b-cc14354cebdd" />
<img width="1982" height="1012" alt="Pasted image 20250205134259" src="https://github.com/user-attachments/assets/8cc30644-c46d-4754-8128-1ef9937f4898" />

The autoencoder also learns a different color distribution to RGB, it does not like learning color jitter

<img width="794" height="394" alt="visualization_epoch_185" src="https://github.com/user-attachments/assets/d46857f9-db15-41ea-a922-bfa9c50a482c" />

## 100x compression (Fall 25-26)

13x compression, using simpler models:
<img width="1982" height="1012" alt="Pasted image 20250904183420" src="https://github.com/user-attachments/assets/6f4cf728-503d-4b20-8eee-7815414e2794" />

High noise ZFP can reach 97x compression:
<img width="1270" height="949" alt="Pasted image 20250905101205" src="https://github.com/user-attachments/assets/0ee4897e-3514-45e3-a874-31cdaa24d7df" />

(9/5/25)

trying a decompressed model to see what color range and transformations the model learns early in the training process, it appears to try to learns position and sharpness over color first
<img width="631" height="467" alt="image" src="https://github.com/user-attachments/assets/9cdbd351-9ab7-42e2-a05c-9de4c2af1bd0" />

Balancing between model and zfp, this model has very little ZFP noise, but the AE is approximating a lot and thus is still blurry
<img width="659" height="482" alt="image" src="https://github.com/user-attachments/assets/1b02e7c9-35bb-4500-94a0-dd921eca0f52" />

Increasing epochs on prev model arch, the model slowly learns to fix some color, still not a full run:
<img width="1982" height="1012" alt="image" src="https://github.com/user-attachments/assets/2c646630-b9a0-401b-97ea-6f53586bfbb4" />













